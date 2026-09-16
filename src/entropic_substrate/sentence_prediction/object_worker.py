"""One request, checkpoint snapshots, and marker-last worker outcomes over R2 ports."""

import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from entropic_substrate.d1_trainer_mailbox_contracts import D1TrainerMailboxObjectIdentity
from entropic_substrate.d1_trainer_object_transport import D1TrainerMailboxObjectPort
from entropic_substrate.evidence.ladder_campaign import artifact_from_mapping
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.sentence_prediction.journal import artifact_reference, read_json, write_json
from entropic_substrate.sentence_prediction.native_verification import read_native_verification
from entropic_substrate.sentence_prediction.numerics import UndefinedDirection
from entropic_substrate.sentence_prediction.recovery_state import recovery_record
from entropic_substrate.sentence_prediction.runner import read_training_progress
from entropic_substrate.sentence_prediction.span_states import SpanPopulationGap
from entropic_substrate.sentence_prediction.worker_execution import execute_probe_request
from entropic_substrate.sentence_prediction.worker_request import ProbeRequest
from entropic_substrate.sentence_prediction.worker_snapshot import publish_worker_snapshot
from entropic_substrate.sentence_prediction.worker_transport import (
    materialize_probe_request,
    publish_verified_object,
    read_published_probe_request,
)


def _paths(root: Path, sections: tuple[str, ...]) -> tuple[str, ...]:
    paths = []
    for section in sections:
        selected = root / section
        if selected.is_symlink():
            raise ValueError("worker output directory cannot be a symlink")
        if not selected.exists():
            continue
        for path in selected.rglob("*"):
            if path.is_symlink():
                raise ValueError("worker outputs cannot contain symlinks")
            if path.is_file():
                paths.append(path.relative_to(root).as_posix())
    return tuple(sorted(paths))


def _event(
    objects: D1TrainerMailboxObjectPort, request: ProbeRequest, name: str, value: dict[str, object]
) -> D1TrainerMailboxObjectIdentity:
    payload = json.dumps(
        {
            "schema": "h-worker-event-v1",
            "request_sha256": request.digest,
            "run_identity": request.profile.identity,
            "scientific_authority": "none",
            "at": datetime.now(UTC).isoformat(),
            **value,
        },
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if len(payload) > request.limits.max_object_bytes:
        raise ValueError("worker event exceeds its byte budget")
    return publish_verified_object(objects, f"{request.prefix}/{name}.json", payload)


def _section(value: object, section: str) -> ArtifactIdentity:
    artifact = artifact_from_mapping(value)
    name = PurePosixPath(artifact.path).name
    if artifact.path != f"{section}/{name}":
        raise ValueError("worker result artifact belongs to another section")
    return ArtifactIdentity(name, artifact.content_hash)


def _admit_result(root: Path, request: ProbeRequest, value: object) -> dict[str, object]:
    row = recovery_record(
        value,
        "schema request_sha256 run_identity scientific_authority status runtime successful_optimizer_steps training generation diagnostics capture verification wall_seconds",
    )
    if (
        row["schema"] != "h-probe-worker-result-v1"
        or row["request_sha256"] != request.digest
        or row["run_identity"] != request.profile.identity
        or row["scientific_authority"] != "none"
    ):
        raise ValueError("worker result identity mismatch")
    duration = row["wall_seconds"]
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration < 0
    ):
        raise ValueError("worker result duration is invalid")
    maximum = request.limits.max_object_bytes
    training = _section(row["training"], "run")
    report = read_training_progress(
        root / "run",
        training,
        request.profile,
        root / "source",
        max_source_bytes=maximum,
        max_artifact_bytes=maximum,
    )
    if (
        type(row["successful_optimizer_steps"]) is not int
        or row["successful_optimizer_steps"] != report["successful_optimizer_steps"]
    ):
        raise ValueError("worker result step count differs from admitted training")
    if row["status"] == "paused":
        if report["status"] != "paused" or any(
            row[k] is not None for k in ("generation", "diagnostics", "capture", "verification")
        ):
            raise ValueError("paused worker result cannot claim later stages")
        return row
    if row["status"] != "probe_workload_complete" or report["status"] != "training_complete":
        raise ValueError("worker completion lacks complete training")
    captured, verified = (
        _section(row["capture"], "capture"),
        _section(row["verification"], "verification"),
    )
    read_native_verification(
        root / "verification",
        verified,
        root / "capture",
        captured,
        expected_request=ContentHash("sha256", request.digest),
        max_bytes=maximum,
        max_source_bytes=maximum,
        max_total_bytes=request.limits.max_total_bytes,
        max_artifacts=request.limits.max_objects,
    )
    manifest = read_json(root / "capture", captured, max_bytes=maximum)
    diagnostics = manifest["diagnostic_roots"]
    if type(diagnostics) is not list:
        raise ValueError("worker capture must declare its diagnostic roots")
    if (
        artifact_from_mapping(manifest["training"]) != training
        or artifact_from_mapping(manifest["generation"]) != _section(row["generation"], "run")
        or _section(row["diagnostics"], "run")
        not in tuple(artifact_from_mapping(a) for a in diagnostics)
    ):
        raise ValueError("worker result is not the verified capture lineage")
    return row


def run_object_worker(
    objects: D1TrainerMailboxObjectPort,
    marker: D1TrainerMailboxObjectIdentity,
    root: Path,
    *,
    max_request_bytes: int,
    repo_root: Path | None = None,
) -> D1TrainerMailboxObjectIdentity:
    """Publish terminal status only after its snapshot is externally readable.

    Invalid request markers are not trusted enough to select a response namespace.
    Once a request is admitted, failure publication preserves partial owned outputs.
    This function never provisions a pod or authorizes another experiment.
    """
    request = read_published_probe_request(objects, marker, max_request_bytes=max_request_bytes)
    maximum = request.limits.max_object_bytes
    stage = "materialization"
    try:
        materialize_probe_request(objects, marker, root, max_request_bytes=max_request_bytes)
        (root / "worker").mkdir()

        def start(runtime: dict[str, object]) -> None:
            started = write_json(
                root,
                "worker/start.json",
                {
                    "schema": "h-worker-start-v1",
                    "request_sha256": request.digest,
                    "runtime": runtime,
                    "meaning": "native_inputs_admitted_before_training_dispatch",
                },
                max_bytes=maximum,
            )
            snapshot = publish_worker_snapshot(
                objects, request, root, ("worker/start.json",), name="started"
            )
            _event(
                objects,
                request,
                "started",
                {
                    "status": "inputs_admitted",
                    "runtime": runtime,
                    "record": artifact_reference(root, started),
                    "snapshot": asdict(snapshot),
                },
            )

        def checkpoint(artifact: ArtifactIdentity) -> None:
            row = read_json(root, artifact, max_bytes=maximum)
            total = row["successful_optimizer_steps"]
            if type(total) is not int or total < 1:
                raise ValueError("worker checkpoint must report measured successful steps")
            snapshot = publish_worker_snapshot(
                objects,
                request,
                root,
                _paths(root, ("run", "worker")),
                name=f"checkpoint-{total:08d}",
            )
            _event(
                objects,
                request,
                f"checkpoint-{total:08d}",
                {
                    "status": "checkpoint",
                    "successful_optimizer_steps": total,
                    "training": artifact_reference(root, artifact),
                    "snapshot": asdict(snapshot),
                },
            )

        stage = "execution"
        result = execute_probe_request(
            request, root, repo_root=repo_root, on_checkpoint=checkpoint, on_start=start
        )
        result = _admit_result(root, request, result)
        result_artifact = write_json(root, "worker/result.json", result, max_bytes=maximum)
        stage = "final_snapshot"
        name = "paused" if result["status"] == "paused" else "complete"
        snapshot = publish_worker_snapshot(
            objects,
            request,
            root,
            _paths(root, ("run", "capture", "verification", "worker")),
            name=name,
        )
        return _event(
            objects,
            request,
            name,
            {
                "status": result["status"],
                "result": artifact_reference(root, result_artifact),
                "snapshot": asdict(snapshot),
            },
        )
    except Exception as error:
        status = (
            "scientific_gap"
            if isinstance(error, UndefinedDirection)
            else "diagnostic_gap"
            if isinstance(error, SpanPopulationGap)
            else "execution_defect"
        )
        failure = {
            "schema": "h-worker-failure-v1",
            "request_sha256": request.digest,
            "scientific_authority": "none",
            "status": status,
            "stage": stage,
            "error_type": type(error).__name__,
            "error_code": getattr(error, "code", type(error).__name__),
            "detail": str(error),
        }
        try:
            (root / "worker").mkdir(exist_ok=True)
            write_json(root, "worker/failure.json", failure, max_bytes=maximum)
            snapshot = publish_worker_snapshot(
                objects,
                request,
                root,
                _paths(root, ("run", "capture", "verification", "worker")),
                name="failure",
            )
        except Exception as delivery_error:
            try:
                _event(
                    objects,
                    request,
                    "failure",
                    {
                        "status": status,
                        "stage": stage,
                        "error_type": type(error).__name__,
                        "error_code": getattr(error, "code", type(error).__name__),
                        "snapshot": None,
                        "evidence_delivery": "failed",
                        "delivery_error_type": type(delivery_error).__name__,
                    },
                )
            except Exception as notification_error:
                raise ExceptionGroup(
                    "worker failure and unavailable evidence delivery",
                    [error, delivery_error, notification_error],
                ) from error
            raise ExceptionGroup(
                "worker failure and evidence delivery defects", [error, delivery_error]
            ) from error
        try:
            _event(
                objects,
                request,
                "failure",
                {
                    "status": status,
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error_code": getattr(error, "code", type(error).__name__),
                    "snapshot": asdict(snapshot),
                    "evidence_delivery": "captured",
                },
            )
        except Exception as notification_error:
            raise ExceptionGroup(
                "worker failure and terminal publication defect", [error, notification_error]
            ) from error
        raise
