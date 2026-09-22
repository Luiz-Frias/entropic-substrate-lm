"""Native-reader receipts and portable admission of trusted executor evidence."""

import math
import time
from datetime import UTC, datetime
from pathlib import Path
from platform import python_version
from uuid import uuid4

import torch

from entropic_substrate.d1_artifact_snapshot import (
    normalize_d1_artifact_root,
    read_d1_artifact_bytes,
)
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.worker.hardware import measure_hardware
from entropic_substrate.sentence_prediction.journal import artifact_reference, read_json, write_json
from entropic_substrate.sentence_prediction.recovery_state import recovery_record
from entropic_substrate.sentence_prediction.trace_capture import (
    inspect_model_custody,
    read_model_trace,
)

SCOPE = "native-model-source-rng-and-numerical-readers"


def _request(value: ContentHash) -> None:
    if type(value) is not ContentHash or value.algorithm != "sha256":
        raise ValueError("native verification requires a SHA-256 request identity")


def _runtime(value: object, device: object) -> None:
    row = recovery_record(value, "python torch device accelerator")
    if row["device"] != device or device not in ("cpu", "cuda"):
        raise ValueError("native verification runtime device mismatch")
    if any(type(value) is not str or not value.strip() for value in (row["python"], row["torch"])):
        raise ValueError("native verification runtime versions must be recorded")
    if device == "cpu":
        if row["accelerator"] is not None:
            raise ValueError("CPU native verification cannot claim GPU execution")
    else:
        accelerator = recovery_record(row["accelerator"], "model count")
        model, count = accelerator["model"], accelerator["count"]
        if type(model) is not str or not model.strip() or type(count) is not int or count < 1:
            raise ValueError("native verification accelerator must be measured")


def _admit_verification(
    value: object,
    capture_root: Path,
    capture: ArtifactIdentity,
    *,
    expected_request: ContentHash,
    max_bytes: int,
    max_source_bytes: int,
    max_total_bytes: int,
    max_artifacts: int,
) -> dict[str, object]:
    """Shared pre-publication and reader admission for a complete receipt payload."""
    _request(expected_request)
    row = recovery_record(
        value,
        "schema status scientific_authority verification_scope request_sha256 capture run_identity runtime verified_at wall_seconds",
    )
    if (
        row["schema"] != "h-native-verification-v1"
        or row["status"] != "native_reader_passed"
        or row["verification_scope"] != SCOPE
        or row["scientific_authority"] != "none"
    ):
        raise ValueError("unknown native verification result or authority")
    if row["request_sha256"] != expected_request.digest:
        raise ValueError("native verification request mismatch")
    if row["capture"] != artifact_reference(capture_root, capture):
        raise ValueError("native verification capture binding mismatch")
    custody = inspect_model_custody(
        capture_root,
        capture,
        max_bytes=max_bytes,
        max_source_bytes=max_source_bytes,
        max_total_bytes=max_total_bytes,
        max_artifacts=max_artifacts,
    )
    if row["run_identity"] != custody["run_identity"]:
        raise ValueError("native verification run identity mismatch")
    _runtime(row["runtime"], custody["declared_training_device"])
    duration = row["wall_seconds"]
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration < 0
    ):
        raise ValueError("native verification duration is invalid")
    stamp = row["verified_at"]
    if type(stamp) is not str or datetime.fromisoformat(stamp).utcoffset() != UTC.utcoffset(None):
        raise ValueError("native verification timestamp must be UTC")
    return row


def read_native_verification(
    root: Path,
    receipt: ArtifactIdentity,
    capture_root: Path,
    capture: ArtifactIdentity,
    *,
    expected_request: ContentHash,
    max_bytes: int,
    max_source_bytes: int,
    max_total_bytes: int,
    max_artifacts: int,
) -> dict[str, object]:
    """Admit a bound executor receipt and recheck custody, without native rerun.

    This is not independent proof against a malicious executor. The controller
    must authenticate the producing execution through its source/transport contract.
    """
    return _admit_verification(
        read_json(root, receipt, max_bytes=max_bytes),
        capture_root,
        capture,
        expected_request=expected_request,
        max_bytes=max_bytes,
        max_source_bytes=max_source_bytes,
        max_total_bytes=max_total_bytes,
        max_artifacts=max_artifacts,
    )


def verify_native_capture(
    capture_root: Path,
    capture: ArtifactIdentity,
    output_root: Path,
    *,
    request_id: ContentHash,
    max_bytes: int,
    max_source_bytes: int,
    max_total_bytes: int,
    max_artifacts: int,
) -> ArtifactIdentity:
    """Publish success only after native admission; same-request recovery is idempotent."""
    _request(request_id)
    output, captured = (
        normalize_d1_artifact_root(output_root),
        normalize_d1_artifact_root(capture_root),
    )
    if output.is_relative_to(captured):
        raise ValueError("native verification output must be outside the captured inventory")
    limits = {
        "max_bytes": max_bytes,
        "max_source_bytes": max_source_bytes,
        "max_total_bytes": max_total_bytes,
        "max_artifacts": max_artifacts,
    }
    name = f"{request_id.digest}-native-verification.json"
    if (output / name).exists():
        _, _, digest = read_d1_artifact_bytes(
            output, name, label="native verification", max_bytes=max_bytes
        )
        receipt = ArtifactIdentity(name, ContentHash("sha256", digest))
        read_native_verification(
            output, receipt, captured, capture, expected_request=request_id, **limits
        )
        return receipt
    started = time.perf_counter()
    try:
        manifest = read_model_trace(captured, capture, **limits)
        custody = inspect_model_custody(captured, capture, **limits)
        device = custody["declared_training_device"]
        accelerator = None
        if device == "cuda":
            model, count = measure_hardware()
            accelerator = {"model": model, "count": count}
        runtime = {
            "python": python_version(),
            "torch": str(torch.__version__),
            "device": device,
            "accelerator": accelerator,
        }
        _runtime(runtime, device)
        payload = {
            "schema": "h-native-verification-v1",
            "status": "native_reader_passed",
            "scientific_authority": "none",
            "verification_scope": SCOPE,
            "request_sha256": request_id.digest,
            "capture": artifact_reference(captured, capture),
            "run_identity": manifest["run_identity"],
            "runtime": runtime,
            "verified_at": datetime.now(UTC).isoformat(),
            "wall_seconds": time.perf_counter() - started,
        }
        _admit_verification(payload, captured, capture, expected_request=request_id, **limits)
        return write_json(output, name, payload, max_bytes=max_bytes)
    except Exception as error:
        try:
            write_json(
                output,
                f"{request_id.digest}-{uuid4().hex}-native-failure.json",
                {
                    "schema": "h-native-verification-failure-v1",
                    "status": "native_validation_failed",
                    "scientific_authority": "none",
                    "request_sha256": request_id.digest,
                    "capture": artifact_reference(captured, capture),
                    "failed_at": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "detail": str(error),
                },
                max_bytes=max_bytes,
            )
        except Exception as recording_error:
            raise ExceptionGroup(
                "native validation and recording defects", [error, recording_error]
            ) from error
        raise
