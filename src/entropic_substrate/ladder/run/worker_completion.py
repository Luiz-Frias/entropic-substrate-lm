"""Host-side completion admission without pretending to rerun native CUDA readers."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import artifact_from_mapping
from entropic_substrate.kernel.lifecycle import ContentHash
from entropic_substrate.ladder.run.campaign.d1_operator_values import require_d1_pod_id
from entropic_substrate.ladder.run.journal import read_json
from entropic_substrate.ladder.run.native_verification import read_native_verification
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker_request import ProbeRequest, worker_artifact
from entropic_substrate.ladder.run.worker_snapshot import materialize_worker_snapshot


def admit_worker_runtime(
    value: object, request: ProbeRequest, expected_pod_id: str | None
) -> dict[str, object]:
    """Compare reported runtime identity to the caller's independently owned allocation."""
    if type(value) is not dict:
        raise ValueError("worker completion requires runtime provenance")
    runtime = cast(dict[str, object], value)
    if request.profile.run_class == "local_tiny":
        if (
            expected_pod_id is not None
            or runtime.get("kind") != "local_tiny"
            or runtime.get("pod_id") is not None
        ):
            raise ValueError("local completion cannot claim a provider allocation")
    else:
        require_d1_pod_id(expected_pod_id)
        if (
            runtime.get("kind") != "remote_tiny"
            or runtime.get("pod_id") != expected_pod_id
            or runtime.get("source_attestation") != "exact_git_tree_match"
            or runtime.get("code_commit") != request.code_commit
            or runtime.get("requested_image") != request.image
        ):
            raise ValueError("worker runtime differs from the allocated construction")
    return runtime


def receive_worker_completion(
    objects: D1TrainerMailboxObjectPort,
    request: ProbeRequest,
    terminal: D1TrainerMailboxObjectIdentity,
    destination: Path,
    *,
    expected_pod_id: str | None,
    max_marker_bytes: int,
) -> dict[str, object]:
    """Bind transferred evidence to the caller's provider allocation and request.

    Admission trusts the authenticated executor's native receipt and rechecks
    its exact capture bytes; it neither reruns CUDA nor files a scientific verdict.
    Rejected completion may retain a byte-verified snapshot for investigation.
    Only this function's successful return admits completion; snapshot.json does not.
    """
    if (
        type(max_marker_bytes) is not int
        or max_marker_bytes < 1
        or terminal.size_bytes > max_marker_bytes
    ):
        raise ValueError("worker completion exceeds its marker byte budget")
    if terminal.key != f"{request.prefix}/complete.json":
        raise ValueError("worker completion marker identity mismatch")
    received = objects.read(terminal.key)
    if type(received) is not D1TrainerMailboxObjectRead or received.identity != terminal:
        raise ValueError("worker completion object identity mismatch")
    event = recovery_record(
        json.loads(received.payload),
        "schema request_sha256 run_identity scientific_authority at status result snapshot",
    )
    if (
        event["schema"] != "h-worker-event-v1"
        or event["status"] != "probe_workload_complete"
        or event["request_sha256"] != request.digest
        or event["run_identity"] != request.profile.identity
        or event["scientific_authority"] != "none"
        or json.dumps(event, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        != received.payload
    ):
        raise ValueError("worker completion identity, authority or encoding mismatch")
    at = event["at"]
    if type(at) is not str or datetime.fromisoformat(at).utcoffset() != UTC.utcoffset(None):
        raise ValueError("worker completion time must be UTC")
    saved = recovery_record(event["snapshot"], "key size_bytes sha256")
    key, size, digest = saved["key"], saved["size_bytes"], saved["sha256"]
    if type(key) is not str or type(size) is not int or type(digest) is not str:
        raise ValueError("worker completion snapshot identity is malformed")
    snapshot = D1TrainerMailboxObjectIdentity(key, size, digest)
    materialize_worker_snapshot(
        objects, request, snapshot, destination, max_marker_bytes=max_marker_bytes
    )
    result_ref = worker_artifact(event["result"], "worker")
    result = recovery_record(
        read_json(destination / "worker", result_ref, max_bytes=request.limits.max_object_bytes),
        "schema request_sha256 run_identity scientific_authority status runtime successful_optimizer_steps training generation diagnostics capture verification wall_seconds",
    )
    if (
        result["schema"] != "h-probe-worker-result-v1"
        or result["request_sha256"] != request.digest
        or result["run_identity"] != request.profile.identity
        or result["scientific_authority"] != "none"
        or result["status"] != "probe_workload_complete"
        or type(result["successful_optimizer_steps"]) is not int
        or result["successful_optimizer_steps"]
        != request.profile.schedule.warmup_steps + request.profile.schedule.steps
    ):
        raise ValueError("worker result does not establish the complete requested workload")
    runtime = admit_worker_runtime(result["runtime"], request, expected_pod_id)
    captured = worker_artifact(result["capture"], "capture")
    verified = worker_artifact(result["verification"], "verification")
    maximum = request.limits.max_object_bytes
    read_native_verification(
        destination / "verification",
        verified,
        destination / "capture",
        captured,
        expected_request=ContentHash("sha256", request.digest),
        max_bytes=maximum,
        max_source_bytes=maximum,
        max_total_bytes=request.limits.max_total_bytes,
        max_artifacts=request.limits.max_objects,
    )
    manifest = read_json(destination / "capture", captured, max_bytes=maximum)
    diagnostics = manifest["diagnostic_roots"]
    if (
        type(diagnostics) is not list
        or artifact_from_mapping(manifest["training"]) != worker_artifact(result["training"], "run")
        or artifact_from_mapping(manifest["generation"])
        != worker_artifact(result["generation"], "run")
        or worker_artifact(result["diagnostics"], "run")
        not in tuple(artifact_from_mapping(v) for v in diagnostics)
    ):
        raise ValueError("worker completion substituted a different captured lineage")
    return {
        "schema": "h-host-probe-completion-v1",
        "status": "probe_completion_verified",
        "scientific_authority": "none",
        "native_validation": "bound_executor_receipt_not_host_rerun",
        "request_sha256": request.digest,
        "successful_optimizer_steps": result["successful_optimizer_steps"],
        "capture": result["capture"],
        "verification": result["verification"],
        "runtime": runtime,
    }
