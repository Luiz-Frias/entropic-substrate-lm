"""Host-side completion admission without pretending to rerun native CUDA readers."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import artifact_from_mapping
from entropic_substrate.kernel.lifecycle import ContentHash
from entropic_substrate.ladder.run.admission import RUN_ADMISSION, RUNTIME_KIND
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
from entropic_substrate.ladder.run.worker_request import (
    RESULT_SCHEMA_V1,
    RESULT_SCHEMA_V2,
    AnyProbeRequest,
    PairProbeRequest,
    parse_pair_result,
    refuse_pair_result_for_single,
    worker_artifact,
    worker_cell_artifact,
)
from entropic_substrate.ladder.run.worker_snapshot import materialize_worker_snapshot

# FOR-1567/pre-for-1567c, DN-1567-C: the `publication` block records per-leg
# seconds and object/retry counts on a worker event. It first appeared on
# checkpoint-* events (design-gate ruling DG-C) and now also appears on the
# `complete` event (final-snapshot leg). The validator is defined ONCE here
# -- the reader-neutral site the lane measured as lowest-coupling, since
# `h_probe_events.py` already imports :func:`admit_worker_runtime` from this
# module (no new edge), while the reverse edge (this module importing
# `h_probe_events`) would close a two-module cycle. Door: read-old tolerated
# for one rollback window (an event without the block still admits),
# write-new always (the worker emits it on every checkpoint/paused/stopped/
# complete terminal event).
_PUBLICATION_SECONDS = ("l1_seconds", "l2_seconds", "l3_seconds")
_PUBLICATION_COUNTS = ("objects_put", "objects_skipped", "verification_reads", "retries")
_PUBLICATION_FIELDS = " ".join((*_PUBLICATION_SECONDS, *_PUBLICATION_COUNTS))


def admit_publication(value: object) -> None:
    row = recovery_record(value, _PUBLICATION_FIELDS)
    for key in _PUBLICATION_SECONDS:
        seconds = row[key]
        if type(seconds) is not float and type(seconds) is not int:
            raise ValueError("worker checkpoint publication seconds must be a number")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("worker checkpoint publication seconds must be finite and >= 0")
    for key in _PUBLICATION_COUNTS:
        count = row[key]
        if type(count) is not int or count < 0:
            raise ValueError("worker checkpoint publication counts must be nonnegative integers")


def admit_worker_runtime(
    value: object, request: AnyProbeRequest, expected_pod_id: str | None
) -> dict[str, object]:
    """Compare reported runtime identity to the caller's independently owned allocation."""
    if type(value) is not dict:
        raise ValueError("worker completion requires runtime provenance")
    runtime = cast(dict[str, object], value)
    # a pair's two cells share one run class (PairProbeRequest refuses two)
    run_class = (
        request.donor.run_class
        if isinstance(request, PairProbeRequest)
        else request.profile.run_class
    )
    kind = RUNTIME_KIND.get(run_class)
    # Rung B: the runtime echoes the request's sealed authority, and a smoke runtime
    # echoes none (an absent key reads as ``None``).
    authority = None if request.authority is None else request.authority.digest
    if RUN_ADMISSION.worker_surface(run_class) == "local_worker":
        if (
            expected_pod_id is not None
            or runtime.get("kind") != kind
            or runtime.get("pod_id") is not None
            or runtime.get("authority") != authority
        ):
            raise ValueError("local completion cannot claim a provider allocation")
    else:
        require_d1_pod_id(expected_pod_id)
        if (
            runtime.get("kind") != kind
            or runtime.get("pod_id") != expected_pod_id
            or runtime.get("source_attestation") != "exact_git_tree_match"
            or runtime.get("code_commit") != request.code_commit
            or runtime.get("requested_image") != request.image
            or runtime.get("authority") != authority
        ):
            raise ValueError("worker runtime differs from the allocated construction")
    return runtime


def receive_worker_completion(
    objects: D1TrainerMailboxObjectPort,
    request: AnyProbeRequest,
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

    FOR-1693 (DH-44): a pair request is admitted by :func:`_receive_pair_completion` and answers
    ``h-host-probe-completion-v2``; a single-profile request keeps ``h-host-probe-completion-v1``
    and a v1 result byte for byte, and refuses a result v2 by name.
    """
    if isinstance(request, PairProbeRequest):
        return _receive_pair_completion(
            objects,
            request,
            terminal,
            destination,
            expected_pod_id=expected_pod_id,
            max_marker_bytes=max_marker_bytes,
        )
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
    raw = json.loads(received.payload)
    # Read-old/write-new (DN-1567-C): a `complete` event recorded before this
    # door existed carries no `publication` key at all and still admits; one
    # written by the current worker always carries it. Mirrors
    # h_probe_events.py's own extra-key pattern for the same door.
    extra = " publication" if type(raw) is dict and "publication" in raw else ""
    event = recovery_record(
        raw,
        "schema request_sha256 run_identity scientific_authority at status result snapshot" + extra,
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
    if "publication" in event:
        admit_publication(event["publication"])
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
    raw_result = read_json(
        destination / "worker", result_ref, max_bytes=request.limits.max_object_bytes
    )
    if raw_result.get("schema") == RESULT_SCHEMA_V2:
        refuse_pair_result_for_single()
    result = recovery_record(
        raw_result,
        "schema request_sha256 run_identity scientific_authority status runtime successful_optimizer_steps training generation diagnostics capture verification wall_seconds",
    )
    if (
        result["schema"] != RESULT_SCHEMA_V1
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


# ---------------------------------------------------------------- the pair (v3): completion v2
COMPLETION_SCHEMA_V2 = "h-host-probe-completion-v2"
# Terminal markers a pair may be received under: the full workload, or the scientific
# exclusion (a pair whose cell stopped on a recorded scientific outcome). A failure terminal is
# never admitted here; the host reader of a result v2 under it is FOR-1694's.
_PAIR_TERMINALS = {
    "complete.json": "probe_workload_complete",
    "excluded.json": "pair_excluded",
}


def _receive_pair_completion(
    objects: D1TrainerMailboxObjectPort,
    request: PairProbeRequest,
    terminal: D1TrainerMailboxObjectIdentity,
    destination: Path,
    *,
    expected_pod_id: str | None,
    max_marker_bytes: int,
) -> dict[str, object]:
    """``h-host-probe-completion-v2``: one entry per cell, the pair verdict, the Q11 counts.

    Admits the ``complete`` terminal and the ``excluded`` terminal only, and a result v2 only: a
    v1 result presented for a pair is refused by name with the minimum supported version. Every
    cell that produced a capture is re-verified against ITS OWN ``capture/<cell>/`` and
    ``verification/<cell>/``, and its steps are checked against its schedule. The publication
    COUNTS (never the seconds) of the terminal event ride here and nowhere else (Q11).
    """
    if (
        type(max_marker_bytes) is not int
        or max_marker_bytes < 1
        or terminal.size_bytes > max_marker_bytes
    ):
        raise ValueError("worker completion exceeds its marker byte budget")
    expected_status = next(
        (
            status
            for name, status in _PAIR_TERMINALS.items()
            if terminal.key == f"{request.prefix}/{name}"
        ),
        None,
    )
    if expected_status is None:
        raise ValueError("worker completion marker identity mismatch")
    received = objects.read(terminal.key)
    if type(received) is not D1TrainerMailboxObjectRead or received.identity != terminal:
        raise ValueError("worker completion object identity mismatch")
    event = recovery_record(
        json.loads(received.payload),
        "schema request_sha256 run_identity scientific_authority at status result snapshot publication",
    )
    if (
        event["schema"] != "h-worker-event-v1"
        or event["status"] != expected_status
        or event["request_sha256"] != request.digest
        or event["run_identity"] != request.run_identity
        or event["scientific_authority"] != "none"
        or json.dumps(event, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        != received.payload
    ):
        raise ValueError("worker completion identity, authority or encoding mismatch")
    admit_publication(event["publication"])
    at = event["at"]
    if type(at) is not str or datetime.fromisoformat(at).utcoffset() != UTC.utcoffset(None):
        raise ValueError("worker completion time must be UTC")
    saved = recovery_record(event["snapshot"], "key size_bytes sha256")
    key, size, digest = saved["key"], saved["size_bytes"], saved["sha256"]
    if type(key) is not str or type(size) is not int or type(digest) is not str:
        raise ValueError("worker completion snapshot identity is malformed")
    materialize_worker_snapshot(
        objects,
        request,
        D1TrainerMailboxObjectIdentity(key, size, digest),
        destination,
        max_marker_bytes=max_marker_bytes,
    )
    result_ref = worker_artifact(event["result"], "worker")
    result = parse_pair_result(
        read_json(destination / "worker", result_ref, max_bytes=request.limits.max_object_bytes),
        request,
    )
    if result["status"] != expected_status:
        raise ValueError("worker result status differs from its terminal event")
    runtime = admit_worker_runtime(result["runtime"], request, expected_pod_id)
    maximum = request.limits.max_object_bytes
    entries: list[dict[str, object]] = []
    for raw, cell in zip(
        cast(list[dict[str, object]], result["cells"]), request.cells, strict=True
    ):
        token, schedule = cell.token, cell.profile.schedule
        outcome = cast(str, raw["outcome"])
        steps = cast(int, raw["successful_optimizer_steps"])
        capture = verification = None
        if outcome == "complete":
            if steps != schedule.warmup_steps + schedule.steps:
                raise ValueError("worker result does not establish the complete requested workload")
            captured = worker_cell_artifact(raw["capture"], "capture", token)
            verified = worker_cell_artifact(raw["verification"], "verification", token)
            read_native_verification(
                destination / "verification" / token,
                verified,
                destination / "capture" / token,
                captured,
                expected_request=ContentHash("sha256", request.digest),
                max_bytes=maximum,
                max_source_bytes=maximum,
                max_total_bytes=request.limits.max_total_bytes,
                max_artifacts=request.limits.max_objects,
            )
            manifest = read_json(destination / "capture" / token, captured, max_bytes=maximum)
            diagnostics = manifest["diagnostic_roots"]
            if (
                type(diagnostics) is not list
                or artifact_from_mapping(manifest["training"])
                != worker_cell_artifact(raw["training"], "run", token)
                or artifact_from_mapping(manifest["generation"])
                != worker_cell_artifact(raw["generation"], "run", token)
                or worker_cell_artifact(raw["diagnostics"], "run", token)
                not in tuple(artifact_from_mapping(v) for v in diagnostics)
            ):
                raise ValueError("worker completion substituted a different captured lineage")
            capture, verification = raw["capture"], raw["verification"]
        elif expected_status == "probe_workload_complete":
            raise ValueError("a complete pair completes both cells")
        entries.append(
            {
                "cell": token,
                "run_identity": raw["run_identity"],
                "outcome": outcome,
                "successful_optimizer_steps": steps,
                "capture": capture,
                "verification": verification,
            }
        )
    publication = cast(dict[str, object], event["publication"])
    return {
        "schema": COMPLETION_SCHEMA_V2,
        "status": "probe_completion_verified",
        "scientific_authority": "none",
        "native_validation": "bound_executor_receipt_not_host_rerun",
        "request_sha256": request.digest,
        "run_identity": request.run_identity,
        "verdict": result["verdict"],
        "cells": entries,
        "runtime": runtime,
        "publication": {key: publication[key] for key in _PUBLICATION_COUNTS},
    }
