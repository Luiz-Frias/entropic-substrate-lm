"""Strict manifest commands for H probe training and evidence inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from entropic_substrate.__legacy.evidence.ladder_campaign import artifact_from_mapping
from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import (
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.generation import read_generation, run_probe_generation
from entropic_substrate.ladder.run.journal import artifact_reference, read_json
from entropic_substrate.ladder.run.runner import read_training_progress, run_probe_training


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


def _directory(root: Path, value: object) -> Path:
    relative = normalize_d1_relative_artifact_path(value, label="H command directory")
    return normalize_d1_artifact_root(root / relative)


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
        source, output = _directory(root, row["source_root"]), _directory(root, row["output_root"])
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
            stage = "execution"
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
            )
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
