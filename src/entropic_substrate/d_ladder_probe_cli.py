"""Strict file-backed diagnostic command over captured predictive-state models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from entropic_substrate.analysis.ladder_probe import ProbeSource, run_verified_checkpoint_probe
from entropic_substrate.analysis.predictive_probe import ProbeConfig
from entropic_substrate.d1_artifact_snapshot import (
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
    read_d1_artifact_bytes,
)
from entropic_substrate.evidence.ladder_campaign import (
    artifact_from_mapping,
    artifact_mapping,
    object_fields,
    read_bound_artifact,
)
from entropic_substrate.experiments.ladder_profile import load_rung_profile


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate score command key")
    return result


def _source(root: Path, value: object) -> ProbeSource:
    row = object_fields(value, "artifact split split_id source_id source_offset")
    artifact = artifact_from_mapping(row["artifact"])
    relative = normalize_d1_relative_artifact_path(artifact.path, label="probe source")
    return ProbeSource(
        root,
        relative,
        artifact.content_hash,
        cast(str, row["split"]),
        cast(str, row["split_id"]),
        cast(str, row["source_id"]),
        cast(int, row["source_offset"]),
    )


def _config(value: object) -> ProbeConfig:
    row = object_fields(
        value,
        "lags train_positions eval_positions updates batch_size learning_rate weight_decay seed",
    )
    if type(row["lags"]) is not list:
        raise ValueError("probe lags must be an explicit list")
    for key in ("learning_rate", "weight_decay"):
        if type(row[key]) not in (int, float):
            raise TypeError("probe rates must be numerical values")
    return ProbeConfig(
        lags=tuple(cast(list[int], row["lags"])),
        train_positions=cast(int, row["train_positions"]),
        eval_positions=cast(int, row["eval_positions"]),
        updates=cast(int, row["updates"]),
        batch_size=cast(int, row["batch_size"]),
        learning_rate=float(cast(float, row["learning_rate"])),
        weight_decay=float(cast(float, row["weight_decay"])),
        seed=cast(int, row["seed"]),
    )


def add_score_gain_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
) -> None:
    parser = sub.add_parser(
        "d-score-gain", help="fit and capture a diagnostic over a verified D-ladder checkpoint"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--manifest", required=True, help="relative D1 or D-ladder score command JSON path"
    )


def cmd_score_gain(args: argparse.Namespace) -> int:
    """Use explicit source spans and hashes; never alter the model or launch a run."""
    root = normalize_d1_artifact_root(args.root)
    _, payload, _ = read_d1_artifact_bytes(root, args.manifest, label="score command")
    value = json.loads(payload, object_pairs_hook=_unique)
    if type(value) is not dict:
        raise ValueError("score command must be an object")
    schema = value.get("schema")
    if schema not in ("d-score-gain-command-v1", "d1-score-gain-command-v1"):
        raise ValueError("unsupported score command schema")
    native = schema == "d1-score-gain-command-v1"
    identity_field = "request" if native else "profile"
    row = object_fields(
        value, f"schema {identity_field} checkpoint training heldout probe_config output"
    )
    checkpoint = artifact_from_mapping(row["checkpoint"])
    relative = normalize_d1_relative_artifact_path(checkpoint.path, label="probe checkpoint")
    checkpoint_path = root / relative
    output = root / normalize_d1_relative_artifact_path(row["output"], label="probe output")
    if output.exists():
        raise FileExistsError("score command requires a new output directory")
    if not output.resolve().is_relative_to(root):
        raise ValueError("probe output escapes the artifact root")
    training, heldout = _source(root, row["training"]), _source(root, row["heldout"])
    config = _config(row["probe_config"])
    if native:
        from entropic_substrate.analysis.d1_probe import run_d1_checkpoint_probe
        from entropic_substrate.d1_trainer_request_codec import parse_d1_trainer_request
        from entropic_substrate.experiments.d1_config import load_d1_execution_catalog

        request = parse_d1_trainer_request(
            read_bound_artifact(root, artifact_from_mapping(row["request"])),
            catalog=load_d1_execution_catalog(),
        )
        result = run_d1_checkpoint_probe(
            request.plan,
            checkpoint_path,
            checkpoint.content_hash.digest,
            training,
            heldout,
            output,
            config,
        )
    else:
        spec = load_rung_profile(root, artifact_from_mapping(row["profile"]))
        _, manifest_bytes, _ = read_d1_artifact_bytes(
            checkpoint_path.parent, "manifest.json", label="model manifest"
        )
        manifest = json.loads(manifest_bytes, object_pairs_hook=_unique)
        if not isinstance(manifest, dict):
            raise ValueError("model manifest must be an object")
        result = run_verified_checkpoint_probe(
            spec,
            manifest,
            checkpoint_path,
            checkpoint.content_hash.digest,
            training,
            heldout,
            output,
            config,
        )
    print(
        json.dumps(
            {
                "receipt": artifact_mapping(result.receipt),
                "weights": artifact_mapping(result.weights),
                "report": artifact_mapping(result.report),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0
