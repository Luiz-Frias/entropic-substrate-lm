"""Manifest-driven model-trace capture and independent inspection commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    normalize_d1_artifact_directory,
    normalize_d1_artifact_root,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.journal import artifact_reference, read_json
from entropic_substrate.ladder.run.trace_capture import (
    capture_model_trace,
    inspect_model_custody,
    read_model_trace,
)

_DIRECTORY_LABEL = "capture command directory"


def add_capture_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    for name in ("h-capture-model", "h-inspect-model", "h-inspect-custody"):
        parser = sub.add_parser(
            name, help="capture or verify an H model trace without granting scientific authority"
        )
        parser.add_argument("--root", required=True, type=Path)
        parser.add_argument("--manifest", required=True)


def cmd_capture(args: argparse.Namespace) -> int:
    try:
        root = normalize_d1_artifact_root(args.root)
        _, _, digest = read_d1_artifact_bytes(
            root, args.manifest, label="capture command", max_bytes=1_048_576
        )
        value = read_json(
            root,
            ArtifactIdentity(args.manifest, ContentHash("sha256", digest)),
            max_bytes=1_048_576,
        )
        custody = args.command == "h-inspect-custody"
        inspect = custody or args.command == "h-inspect-model"
        version = value.get("schema")
        if version not in (
            (("h-custody-inspect-command-v1" if custody else "h-model-inspect-command-v1"),)
            if inspect
            else ("h-model-capture-command-v1", "h-model-capture-command-v2")
        ):
            raise ValueError("unknown capture command schema")
        diagnostics_enabled = version == "h-model-capture-command-v2"
        names = (
            "capture_root capture"
            if inspect
            else "profile source_root output_root destination training generation forensic_roots"
        )
        if diagnostics_enabled:
            names += " diagnostic_roots"
        row = recovery_record(
            value, "schema max_source_bytes max_bytes max_total_bytes max_artifacts " + names
        )
        limits: dict[str, int] = {}
        for name in ("max_source_bytes", "max_bytes", "max_total_bytes", "max_artifacts"):
            count = row[name]
            if type(count) is not int or count < 1:
                raise ValueError("capture budgets must be positive integers")
            limits[name] = count
        if inspect:
            reader = inspect_model_custody if custody else read_model_trace
            result = reader(
                normalize_d1_artifact_directory(root, row["capture_root"], label=_DIRECTORY_LABEL),
                artifact_from_mapping(row["capture"]),
                **limits,
            )
        else:
            profile = HProfile.from_manifest(
                read_json(
                    root, artifact_from_mapping(row["profile"]), max_bytes=limits["max_bytes"]
                )
            )
            extra = row["forensic_roots"]
            if type(extra) is not list:
                raise ValueError("forensic roots must be an explicit array")
            diagnostics = row["diagnostic_roots"] if diagnostics_enabled else []
            if type(diagnostics) is not list:
                raise ValueError("diagnostic roots must be an explicit array")
            destination = normalize_d1_artifact_directory(
                root, row["destination"], label=_DIRECTORY_LABEL
            )
            captured = capture_model_trace(
                normalize_d1_artifact_directory(root, row["output_root"], label=_DIRECTORY_LABEL),
                artifact_from_mapping(row["training"]),
                artifact_from_mapping(row["generation"]),
                profile,
                normalize_d1_artifact_directory(root, row["source_root"], label=_DIRECTORY_LABEL),
                destination,
                forensic_roots=tuple(artifact_from_mapping(item) for item in extra),
                diagnostic_roots=tuple(artifact_from_mapping(item) for item in diagnostics),
                **limits,
            )
            read_model_trace(destination, captured, **limits)
            result = {
                "status": "model_trace_captured",
                "scientific_authority": "none",
                "capture": artifact_reference(destination, captured),
            }
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "status": "capture_refused",
                    "error_type": type(error).__name__,
                    "detail": str(error),
                    "scientific_authority": "none",
                },
                sort_keys=True,
            )
        )
        return 2
