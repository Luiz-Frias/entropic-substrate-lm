"""Strict public manifests for sealed-model span diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import cast

from entropic_substrate.__legacy.analysis.predictive_probe import ProbeConfig
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
from entropic_substrate.ladder.run.journal import artifact_reference, read_json
from entropic_substrate.ladder.run.span_diagnostics import (
    read_span_diagnostics,
    run_span_diagnostics,
)
from entropic_substrate.ladder.run.span_states import SpanPopulationGap


def add_span_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    for name in ("h-diagnose-spans", "h-inspect-spans"):
        parser = sub.add_parser(name, help="run or inspect report-only sentence score diagnostics")
        parser.add_argument("--root", required=True, type=Path)
        parser.add_argument("--manifest", required=True)


def _directory(root: Path, value: object) -> Path:
    return normalize_d1_artifact_root(
        root / normalize_d1_relative_artifact_path(value, label="span command directory")
    )


def _config(value: object) -> ProbeConfig:
    row = recovery_record(value, " ".join(f.name for f in fields(ProbeConfig)))
    lags = row["lags"]
    if type(lags) is not list or any(type(lag) is not int for lag in lags):
        raise ValueError("diagnostic lags require an explicit JSON array")
    if tuple(sorted(set(lags))) != tuple(lags):
        raise ValueError("diagnostic lags must be unique and increasing")

    def integer(name: str) -> int:
        value = row[name]
        if type(value) is not int:
            raise ValueError("diagnostic counts and seeds must be integers")
        return value

    def number(name: str) -> float:
        value = row[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("diagnostic fitting rates must be numeric and not boolean")
        return float(value)

    result = ProbeConfig(
        lags=tuple(cast(list[int], lags)),
        train_positions=integer("train_positions"),
        eval_positions=integer("eval_positions"),
        updates=integer("updates"),
        batch_size=integer("batch_size"),
        learning_rate=number("learning_rate"),
        weight_decay=number("weight_decay"),
        seed=integer("seed"),
    )
    if min(result.train_positions, result.eval_positions) < len(result.lags):
        raise ValueError("diagnostic populations must cover every declared lag")
    return result


def cmd_span(args: argparse.Namespace) -> int:
    stage = "validation"
    try:
        root = normalize_d1_artifact_root(args.root)
        _, _, digest = read_d1_artifact_bytes(
            root, args.manifest, label="span command", max_bytes=1_048_576
        )
        raw = read_json(
            root,
            ArtifactIdentity(args.manifest, ContentHash("sha256", digest)),
            max_bytes=1_048_576,
        )
        inspect = args.command == "h-inspect-spans"
        row = recovery_record(
            raw,
            "schema profile source_root output_root training max_source_bytes max_artifact_bytes "
            + ("receipt" if inspect else "config"),
        )
        if row["schema"] != (
            "h-span-inspect-command-v1" if inspect else "h-span-diagnostics-command-v1"
        ):
            raise ValueError("unknown span command schema")
        limits: dict[str, int] = {}
        for name in ("max_source_bytes", "max_artifact_bytes"):
            value = row[name]
            if type(value) is not int or value < 1:
                raise ValueError("diagnostic byte limits must be positive integers")
            limits[name] = value
        profile = HProfile.from_manifest(
            read_json(
                root, artifact_from_mapping(row["profile"]), max_bytes=limits["max_artifact_bytes"]
            )
        )
        source, output = _directory(root, row["source_root"]), _directory(root, row["output_root"])
        training = artifact_from_mapping(row["training"])
        if inspect:
            result = read_span_diagnostics(
                output, artifact_from_mapping(row["receipt"]), training, profile, source, **limits
            )
        else:
            config = _config(row["config"])
            stage = "execution"
            receipt = run_span_diagnostics(output, training, profile, source, config, **limits)
            result = {
                "status": "diagnostics_complete",
                "scientific_authority": "none",
                "receipt": artifact_reference(output, receipt),
            }
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError, ExceptionGroup) as error:
        status = (
            "invalid_request"
            if stage == "validation"
            else "diagnostic_gap"
            if isinstance(error, SpanPopulationGap)
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
        return 2
