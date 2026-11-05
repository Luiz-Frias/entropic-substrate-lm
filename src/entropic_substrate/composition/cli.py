"""Strict manifest commands for H probe training and evidence inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.core.sentence.scoring import LatentUsageOperand, latent_usage_estimator
from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    normalize_d1_artifact_directory,
    normalize_d1_artifact_root,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.capture.adapters.composite import build_sink_for_run
from entropic_substrate.ladder.capture.ports.run_descriptor import TrackingRunDescriptor
from entropic_substrate.ladder.capture.reader.sinks import (
    CONFIG_HASH_TAG,
    EXP_TAG,
    git_sha,
    h_config_hash,
)
from entropic_substrate.ladder.run.generation import read_generation, run_probe_generation
from entropic_substrate.ladder.run.h_metric_extractor import HTrainingStepExtractor
from entropic_substrate.ladder.run.journal import artifact_reference, read_json
from entropic_substrate.ladder.run.runner import read_training_progress, run_probe_training

_DIRECTORY_LABEL = "H command directory"


def add_sentence_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    for name, description in (
        ("h-train-probe", "execute a file-bound H engineering training probe"),
        ("h-inspect-training", "verify an H training receipt without running training"),
        ("h-generate-probe", "generate from an H probe's sealed trained model"),
        ("h-inspect-generation", "verify generation feedback and lineage"),
    ):
        parser = sub.add_parser(name, help=description)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--manifest", required=True)


def _positive(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("command byte limits and counters must be positive integers")
    return value


def cmd_sentence(args: argparse.Namespace) -> int:
    stage = "validation"
    try:
        root = normalize_d1_artifact_root(args.root)
        _, _, digest = read_d1_artifact_bytes(
            root, args.manifest, label="H command", max_bytes=1_048_576
        )
        raw = read_json(
            root,
            ArtifactIdentity(args.manifest, ContentHash("sha256", digest)),
            max_bytes=1_048_576,
        )
        inspect = args.command.startswith("h-inspect-")
        generation = args.command in ("h-generate-probe", "h-inspect-generation")
        fields = (
            "schema profile source_root output_root max_source_bytes max_artifact_bytes "
            + ("training " if generation else "")
            + ("receipt" if inspect else "resume pause_after")
        )
        row = recovery_record(raw, fields)
        expected = {
            "h-train-probe": "h-probe-train-command-v1",
            "h-inspect-training": "h-probe-inspect-command-v1",
            "h-generate-probe": "h-probe-generate-command-v1",
            "h-inspect-generation": "h-probe-inspect-generation-command-v1",
        }[args.command]
        if row["schema"] != expected:
            raise ValueError("unsupported H command schema")
        source_limit, artifact_limit = (
            _positive(row["max_source_bytes"]),
            _positive(row["max_artifact_bytes"]),
        )
        profile = HProfile.from_manifest(
            read_json(root, artifact_from_mapping(row["profile"]), max_bytes=artifact_limit)
        )
        source, output = (
            normalize_d1_artifact_directory(root, row["source_root"], label=_DIRECTORY_LABEL),
            normalize_d1_artifact_directory(root, row["output_root"], label=_DIRECTORY_LABEL),
        )
        if generation:
            training = artifact_from_mapping(row["training"])
            if inspect:
                result = read_generation(
                    output,
                    artifact_from_mapping(row["receipt"]),
                    training,
                    profile,
                    source,
                    max_source_bytes=source_limit,
                    max_artifact_bytes=artifact_limit,
                )
            else:
                count = None if row["pause_after"] is None else _positive(row["pause_after"])
                resume = None if row["resume"] is None else artifact_from_mapping(row["resume"])
                stage = "execution"
                generated = run_probe_generation(
                    output,
                    training,
                    profile,
                    source,
                    max_source_bytes=source_limit,
                    max_artifact_bytes=artifact_limit,
                    resume=resume,
                    pause_after=count,
                )
                result = {
                    "status": generated.status,
                    "scientific_authority": "none",
                    "receipt": artifact_reference(output, generated.receipt),
                }
        elif inspect:
            result = read_training_progress(
                output,
                artifact_from_mapping(row["receipt"]),
                profile,
                source,
                max_source_bytes=source_limit,
                max_artifact_bytes=artifact_limit,
            )
        else:
            pause = None
            if row["pause_after"] is not None:
                point = recovery_record(row["pause_after"], "phase step")
                phase, step = point["phase"], _positive(point["step"])
                if phase not in ("codec_warmup", "prediction"):
                    raise ValueError("unknown pause phase")
                budget = (
                    profile.schedule.warmup_steps
                    if phase == "codec_warmup"
                    else profile.schedule.steps
                )
                if step > budget or not (
                    step == budget
                    or step % profile.schedule.checkpoint_interval == 0
                    or (phase == "prediction" and step % profile.schedule.evaluation_interval == 0)
                ):
                    raise ValueError("pause point must select a declared seal boundary")
                pause = (phase, step)
            resume = None if row["resume"] is None else artifact_from_mapping(row["resume"])
            # FOR-1644: the probe is tracked like the other H producers. The results file
            # is a SIBLING of the output root, never inside it: run_probe_training
            # refuses a fresh run whose root is non-empty, and JsonlSink creates its
            # parent. A sibling of an output root that normalization keeps under --root
            # stays under --root. JsonlSink appends, so a FRESH run over a results file a
            # previous run left is refused here (a validation refusal, exit 2) instead of
            # silently mixing two runs; a RESUME continues its paused invocation's file.
            results_path = output.parent / f"{output.name}.observability" / "metrics.jsonl"
            if resume is None and results_path.exists():
                raise ValueError("fresh probe results file already exists; resume it or remove it")
            # The sink is built only here, after every validation refusal and after the
            # stage leaves "validation": a sink-construction failure (MlflowStartupError
            # is a RuntimeError) is an execution defect, never an invalid request.
            stage = "execution"
            chash = h_config_hash(profile)
            sink = build_sink_for_run(
                TrackingRunDescriptor(
                    # `-tiny` suffix: TRACER-PARTITION (FOR-1642). The probe writes no
                    # h-run-lifecycle-v1 records, so capture_results must keep it off the
                    # H lane by default, exactly as for the local tiny tracer.
                    run_name=f"h-train-probe-{profile.identity}-tiny",
                    tiny=True,
                    params={
                        EXP_TAG: profile.name,
                        "seed": profile.seed,
                        CONFIG_HASH_TAG: chash,
                        "git_sha": git_sha(),
                        "run_class": profile.run_class,
                        "precision": profile.precision,
                    },
                    tags={
                        "schema": "h-train-probe-v1",
                        EXP_TAG: profile.name,
                        CONFIG_HASH_TAG: chash,
                    },
                ),
                results_path=results_path,
                metric_extractor=HTrainingStepExtractor(warmup_steps=profile.schedule.warmup_steps),
            )
            # DN-1643-PORT: FINISHED iff run_probe_training RETURNED; every raised exit,
            # BaseException included, finishes the sink FAILED exactly once and leaves
            # the exception to the handler below untouched.
            terminal_status = "FAILED"
            try:
                progress = run_probe_training(
                    profile,
                    source,
                    output,
                    max_source_bytes=source_limit,
                    max_artifact_bytes=artifact_limit,
                    resume=resume,
                    stop_after_seal=None
                    if pause is None
                    else lambda phase, step: (phase, step) == pause,
                    sink=sink,
                    latent_usage=latent_usage_estimator(LatentUsageOperand.posterior_mean),
                )
                terminal_status = "FINISHED"
            finally:
                sink.finish(status=terminal_status)
            result = {
                "status": progress.status,
                "scientific_authority": "none",
                "last_seal": artifact_reference(output, progress.last_seal),
                "receipt": artifact_reference(output, progress.receipt),
            }
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        status = (
            "invalid_request"
            if stage == "validation"
            else "scientific_gap"
            if isinstance(error, UndefinedDirection)
            else "execution_defect"
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "error_type": type(error).__name__,
                    "detail": str(error),
                    "scientific_authority": "none",
                },
                sort_keys=True,
            )
        )
        return 2 if stage == "validation" else 1
