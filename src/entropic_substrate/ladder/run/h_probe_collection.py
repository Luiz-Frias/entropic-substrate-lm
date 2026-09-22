"""Collect one terminal worker lineage without promoting partial custody to completion."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from entropic_substrate.__legacy.d1_trainer_object_transport import D1TrainerMailboxObjectPort
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import (
    normalize_d1_artifact_root,
    publish_d1_atomic_bytes,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.h_probe_events import ObservedProbeEvent, read_probe_event
from entropic_substrate.ladder.run.journal import read_json, write_json
from entropic_substrate.ladder.run.worker_completion import (
    admit_worker_runtime,
    receive_worker_completion,
)
from entropic_substrate.ladder.run.worker_request import (
    ProbeRequest,
    worker_artifact,
)
from entropic_substrate.ladder.run.worker_snapshot import (
    materialize_worker_snapshot,
)


def collect_probe_terminal(
    objects: D1TrainerMailboxObjectPort,
    request: ProbeRequest,
    destination: Path,
    *,
    expected_pod_id: str | None,
    max_marker_bytes: int,
) -> dict[str, object] | None:
    """One observation, one fresh destination; conflicts and invalid evidence fail loud.

    Completion reuses native-receipt admission. Pause/failure capture proves bytes
    and request bindings, not host replay of native model state or a scientific
    classification. A failed copy retains unadmitted partial files for inspection.
    """
    found = [
        (name, event)
        for name in ("complete", "paused", "failure")
        if (
            event := read_probe_event(
                objects, request, name, expected_pod_id=expected_pod_id, max_bytes=max_marker_bytes
            )
        )
        is not None
    ]
    if not found:
        return None
    if len(found) != 1:
        raise ValueError("conflicting terminal claims for one worker request")
    root = normalize_d1_artifact_root(destination)
    if any(root.iterdir()):
        raise ValueError("terminal collection requires a fresh empty destination")
    name, event = found[0]
    report: dict[str, object] = {
        "schema": "h-probe-terminal-collection-v1",
        "scientific_authority": "none",
        "request_sha256": request.digest,
        "terminal": asdict(event.identity),
        "reported_outcome": event.reported_status,
        "workload_verified": False,
        "reported_optimizer_steps": None,
        "native_validation": "not_replayed_by_host",
    }
    try:
        _capture(objects, request, root, name, event, report, expected_pod_id, max_marker_bytes)
    except Exception as primary:
        try:
            publish_d1_atomic_bytes(root, "terminal.json", event.payload, label="terminal evidence")
        except Exception as custody_error:
            raise ExceptionGroup(
                "terminal recovery and evidence preservation failed", [primary, custody_error]
            ) from primary
        raise
    terminal = publish_d1_atomic_bytes(
        root, "terminal.json", event.payload, label="terminal evidence"
    )
    report["terminal_artifact"] = {"path": "terminal.json", "sha256": terminal.content_hash.digest}
    write_json(root, "collection.json", report, max_bytes=max_marker_bytes)
    return report


def _capture(
    objects: D1TrainerMailboxObjectPort,
    request: ProbeRequest,
    root: Path,
    name: str,
    event: ObservedProbeEvent,
    report: dict[str, object],
    expected_pod_id: str | None,
    max_marker_bytes: int,
) -> None:
    if name == "complete":
        completion = receive_worker_completion(
            objects,
            request,
            event.identity,
            root,
            expected_pod_id=expected_pod_id,
            max_marker_bytes=max_marker_bytes,
        )
        report.update(
            workload_verified=True,
            capture_status="native_completion_receipt_verified",
            reported_optimizer_steps=completion["successful_optimizer_steps"],
            native_validation=completion["native_validation"],
            completion=completion,
        )
    elif event.snapshot is None:
        report["capture_status"] = "missing_failure_snapshot"
    else:
        snapshot = materialize_worker_snapshot(
            objects, request, event.snapshot, root, max_marker_bytes=max_marker_bytes
        )
        row: dict[str, object] = json.loads(event.payload)
        if name == "paused":
            paused = recovery_record(
                read_json(
                    root / "worker",
                    worker_artifact(row["result"], "worker"),
                    max_bytes=request.limits.max_object_bytes,
                ),
                "schema request_sha256 run_identity scientific_authority status runtime successful_optimizer_steps training generation diagnostics capture verification wall_seconds",
            )
            pause = request.pause
            if pause is None:
                raise ValueError("pause evidence requires an explicitly requested pause")
            expected_steps = pause[1] + (
                request.profile.schedule.warmup_steps if pause[0] == "prediction" else 0
            )
            if (
                paused["schema"] != "h-probe-worker-result-v1"
                or paused["request_sha256"] != request.digest
                or paused["run_identity"] != request.profile.identity
                or paused["scientific_authority"] != "none"
                or paused["status"] != "paused"
                or type(paused["successful_optimizer_steps"]) is not int
                or paused["successful_optimizer_steps"] != expected_steps
                or any(
                    paused[k] is not None
                    for k in ("generation", "diagnostics", "capture", "verification")
                )
            ):
                raise ValueError("paused result differs from its requested lineage or endpoint")
            admit_worker_runtime(paused["runtime"], request, expected_pod_id)
            read_json(
                root / "run",
                worker_artifact(paused["training"], "run"),
                max_bytes=request.limits.max_object_bytes,
            )
            report.update(
                capture_status="pause_bytes_verified",
                reported_optimizer_steps=expected_steps,
                resume_artifact=paused["training"],
            )
        else:
            files = cast(list[dict[str, object]], snapshot["files"])
            entries = [item for item in files if item["path"] == "worker/failure.json"]
            if len(entries) != 1:
                raise ValueError("failure snapshot lacks its unique failure record")
            reference = ArtifactIdentity(
                "failure.json", ContentHash("sha256", cast(str, entries[0]["sha256"]))
            )
            failure = recovery_record(
                read_json(root / "worker", reference, max_bytes=request.limits.max_object_bytes),
                "schema request_sha256 scientific_authority status stage error_type error_code detail",
            )
            if (
                failure["schema"] != "h-worker-failure-v1"
                or failure["request_sha256"] != request.digest
                or failure["scientific_authority"] != "none"
                or any(
                    failure[key] != row[key]
                    for key in ("status", "stage", "error_type", "error_code")
                )
                or type(failure["detail"]) is not str
            ):
                raise ValueError("failure record differs from its terminal event")
            report["capture_status"] = "failure_bytes_verified"
        report["snapshot"] = asdict(event.snapshot)
