"""Host observations of H worker events; artifact verification remains separate."""

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker_completion import admit_publication, admit_worker_runtime
from entropic_substrate.ladder.run.worker_request import (
    ProbeRequest,
    worker_artifact,
)

_COMMON = "schema request_sha256 run_identity scientific_authority at status snapshot"
_FIELDS = {
    "started": "runtime record",
    "checkpoint": "successful_optimizer_steps training",
    "complete": "result",
    "paused": "result",
    "stopped": "result successful_optimizer_steps",
    "failure": "stage error_type error_code evidence_delivery",
}
_STATUS = {
    "started": "inputs_admitted",
    "checkpoint": "checkpoint",
    "complete": "probe_workload_complete",
    "paused": "paused",
    "stopped": "stopped_by_operator",
}
# FOR-1567/pre-for-1567c, design-gate rulings DG-C and DN-1567-C: a
# `publication` block records per-leg seconds and object/retry counts. It
# first appeared on checkpoint-* events only (DG-C); DN-1567-C (operator,
# 2026-09-29) widened the door to every terminal event the final-snapshot
# call site emits (`complete`, `paused`, `stopped`), since one call site
# serves all three and the worker writes the block on all of them
# unconditionally. Door: read-old tolerated for one rollback window (an
# event of any of these kinds WITHOUT the block still admits), write-new
# always. `started` and `failure` are NOT in this set -- an event of either
# kind carrying `publication` still raises (the exact-key admission below
# has no `extra` branch for them). The validator itself lives in
# `worker_completion.py` (reused, not copied): this module already imports
# `admit_worker_runtime` from there, so importing `admit_publication` too
# adds no new edge, while the reverse import would close a cycle.
_PUBLICATION_KINDS = ("checkpoint", "complete", "paused", "stopped")


@dataclass(frozen=True, slots=True)
class ObservedProbeEvent:
    """Immutable reported evidence, never a completion or scientific certificate."""

    identity: D1TrainerMailboxObjectIdentity
    payload: bytes
    reported_status: str
    snapshot: D1TrainerMailboxObjectIdentity | None
    reported_optimizer_steps: int | None

    @property
    def scientific_authority(self) -> Literal["none"]:
        return "none"

    @property
    def completion_verified(self) -> Literal[False]:
        return False


def _snapshot(value: object, request: ProbeRequest, name: str) -> D1TrainerMailboxObjectIdentity:
    row = recovery_record(value, "key size_bytes sha256")
    key, size, digest = row["key"], row["size_bytes"], row["sha256"]
    if (
        type(key) is not str
        or key != f"{request.prefix}/snapshots/{name}.json"
        or type(size) is not int
        or size > request.limits.max_object_bytes
        or type(digest) is not str
    ):
        raise ValueError("worker event snapshot identity or budget mismatch")
    return D1TrainerMailboxObjectIdentity(key, size, digest)


def _admit_failure(row: dict[str, object]) -> None:
    if (
        row["status"] not in ("execution_defect", "scientific_gap", "diagnostic_gap")
        or row["stage"] not in ("materialization", "execution", "final_snapshot")
        or row["evidence_delivery"] not in ("captured", "failed")
    ):
        raise ValueError("worker failure classification is invalid")
    fields = ["error_type", "error_code"]
    if row["evidence_delivery"] == "failed":
        if row["snapshot"] is not None:
            raise ValueError("failed evidence delivery cannot claim a snapshot")
        fields.append("delivery_error_type")
    for field in fields:
        value = row[field]
        if type(value) is not str or not value.strip():
            raise ValueError("worker failure requires an explicit error classification")


def read_probe_event(
    objects: D1TrainerMailboxObjectPort,
    request: ProbeRequest,
    name: str,
    *,
    expected_pod_id: str | None,
    max_bytes: int,
) -> ObservedProbeEvent | None:
    """Read one bounded immutable event from the request's exact namespace.

    Missing objects return None, not readiness. A reported completion must still
    pass receive_worker_completion before the host admits workload completion.
    The object adapter must also bound network reads; this reader validates the
    returned bytes and never interprets a snapshot marker as verified contents.
    """
    if type(request) is not ProbeRequest:
        raise ValueError("worker observation requires a validated probe request")
    if type(name) is not str:
        raise ValueError("worker event name must be text")
    checkpoint = re.fullmatch(r"checkpoint-([0-9]{8,})", name)
    kind = "checkpoint" if checkpoint else name
    if kind not in _FIELDS or name == "checkpoint":
        raise ValueError("worker event name is unsupported")
    if type(max_bytes) is not int or not 1 <= max_bytes <= request.limits.max_object_bytes:
        raise ValueError("worker event requires a positive bounded byte budget")
    key = f"{request.prefix}/{name}.json"
    received = objects.read(key)
    if received is None:
        return None
    if (
        type(received) is not D1TrainerMailboxObjectRead
        or received.identity.key != key
        or received.identity.size_bytes > max_bytes
    ):
        raise ValueError("worker event object identity or budget mismatch")
    value = json.loads(received.payload)
    extra = ""
    if type(value) is dict and kind == "failure" and value.get("evidence_delivery") == "failed":
        extra = " delivery_error_type"
    if type(value) is dict and kind in _PUBLICATION_KINDS and "publication" in value:
        extra += " publication"
    row = recovery_record(value, f"{_COMMON} {_FIELDS[kind]}{extra}")
    if (
        row["schema"] != "h-worker-event-v1"
        or row["request_sha256"] != request.digest
        or row["run_identity"] != request.profile.identity
        or row["scientific_authority"] != "none"
        or json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        != received.payload
    ):
        raise ValueError("worker event identity, authority or encoding mismatch")
    at = row["at"]
    if type(at) is not str or datetime.fromisoformat(at).utcoffset() != UTC.utcoffset(None):
        raise ValueError("worker event time must be UTC")
    status = row["status"]
    if type(status) is not str or (kind != "failure" and status != _STATUS[kind]):
        raise ValueError("worker event status differs from its role")
    steps = None
    if kind == "failure":
        _admit_failure(row)
    elif kind == "started":
        admit_worker_runtime(row["runtime"], request, expected_pod_id)
        worker_artifact(row["record"], "worker")
    elif kind == "checkpoint":
        count = row["successful_optimizer_steps"]
        if (
            type(count) is not int
            or not 1
            <= count
            <= request.profile.schedule.warmup_steps + request.profile.schedule.steps
            or name != f"checkpoint-{count:08d}"
        ):
            raise ValueError("worker checkpoint count differs from event or workload")
        steps = count
        worker_artifact(row["training"], "run")
    elif kind == "stopped":
        count = row["successful_optimizer_steps"]
        if (
            type(count) is not int
            or not 1
            <= count
            <= request.profile.schedule.warmup_steps + request.profile.schedule.steps
        ):
            raise ValueError("worker stopped step differs from workload bounds")
        steps = count
        worker_artifact(row["result"], "worker")
    else:
        worker_artifact(row["result"], "worker")
    if kind in _PUBLICATION_KINDS and "publication" in row:
        admit_publication(row["publication"])
    snapshot = (
        None
        if kind == "failure" and row["evidence_delivery"] == "failed"
        else _snapshot(row["snapshot"], request, name)
    )
    return ObservedProbeEvent(received.identity, received.payload, status, snapshot, steps)
