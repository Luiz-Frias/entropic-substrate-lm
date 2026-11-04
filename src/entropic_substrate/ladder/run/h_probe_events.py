"""Host observations of H worker events; artifact verification remains separate."""

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

from entropic_substrate.core.sentence.profile import HProfile
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
    AnyProbeRequest,
    PairProbeRequest,
    ProbeRequest,
    worker_artifact,
    worker_cell_artifact,
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
# FOR-1694: a PAIR's events (written by ``object_worker``). The set adds ``excluded`` (the pair's
# one non-failure scientific terminal, status ``pair_excluded``); a stopped pair event carries no
# ``successful_optimizer_steps`` (each cell has its own, inside the result v2 it cites), and a
# checkpoint event is named for its CELL: ``checkpoint-<cell>-<total:08d>``.
_PAIR_FIELDS = {**_FIELDS, "stopped": "result", "excluded": "result"}
_PAIR_STATUS = {**_STATUS, "excluded": "pair_excluded"}
_PAIR_CHECKPOINT = re.compile(r"checkpoint-([a-z0-9][a-z0-9-]*)-([0-9]{8,})")
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
_PUBLICATION_KINDS = ("checkpoint", "complete", "paused", "stopped", "excluded")


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


def _snapshot(value: object, request: AnyProbeRequest, name: str) -> D1TrainerMailboxObjectIdentity:
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
    request: AnyProbeRequest,
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

    FOR-1694: a pair request reads the pair's event vocabulary (``_PAIR_FIELDS``): its own
    terminal names and cell-named checkpoints, with the run identity each event carries
    (pair-level events the pair id, a checkpoint its cell's profile identity, a failure the pair
    id or the failing cell's). A single-profile request keeps every byte of its reader.
    """
    if type(request) is not ProbeRequest and type(request) is not PairProbeRequest:
        raise ValueError("worker observation requires a validated probe request")
    if type(name) is not str:
        raise ValueError("worker event name must be text")
    pair = request if isinstance(request, PairProbeRequest) else None
    fields_by_kind = _FIELDS if pair is None else _PAIR_FIELDS
    cell: str | None = None
    if pair is None:
        checkpoint = re.fullmatch(r"checkpoint-([0-9]{8,})", name)
    else:
        checkpoint = _PAIR_CHECKPOINT.fullmatch(name)
        if checkpoint is not None:
            cell = checkpoint.group(1)
            if cell not in pair.tokens:
                raise ValueError("worker event names a cell outside its pair")
    kind = "checkpoint" if checkpoint else name
    if kind not in fields_by_kind or name == "checkpoint":
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
    row = recovery_record(value, f"{_COMMON} {fields_by_kind[kind]}{extra}")
    if pair is None:
        run_identity_ok = row["run_identity"] == _single_profile(request).identity
    elif kind == "checkpoint":
        run_identity_ok = (
            row["run_identity"] == pair.cells[pair.tokens.index(cast(str, cell))].profile.identity
        )
    elif kind == "failure":
        # the failing cell's identity (a hand-off failure is the recipient's) or the pair's own
        run_identity_ok = row["run_identity"] in (
            pair.run_identity,
            pair.donor.identity,
            pair.recipient.identity,
        )
    else:
        run_identity_ok = row["run_identity"] == pair.run_identity
    if (
        row["schema"] != "h-worker-event-v1"
        or row["request_sha256"] != request.digest
        or not run_identity_ok
        or row["scientific_authority"] != "none"
        or json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        != received.payload
    ):
        raise ValueError("worker event identity, authority or encoding mismatch")
    at = row["at"]
    if type(at) is not str or datetime.fromisoformat(at).utcoffset() != UTC.utcoffset(None):
        raise ValueError("worker event time must be UTC")
    status = row["status"]
    statuses = _STATUS if pair is None else _PAIR_STATUS
    if type(status) is not str or (kind != "failure" and status != statuses[kind]):
        raise ValueError("worker event status differs from its role")
    steps = None
    if kind == "failure":
        _admit_failure(row)
    elif kind == "started":
        admit_worker_runtime(row["runtime"], request, expected_pod_id)
        worker_artifact(row["record"], "worker")
    elif kind == "checkpoint":
        count = row["successful_optimizer_steps"]
        schedule = (
            _single_profile(request).schedule
            if pair is None
            else pair.cells[pair.tokens.index(cast(str, cell))].profile.schedule
        )
        if (
            type(count) is not int
            or not 1 <= count <= schedule.warmup_steps + schedule.steps
            or name
            != (f"checkpoint-{count:08d}" if cell is None else f"checkpoint-{cell}-{count:08d}")
        ):
            raise ValueError("worker checkpoint count differs from event or workload")
        steps = count
        if pair is None:
            worker_artifact(row["training"], "run")
        else:
            worker_cell_artifact(row["training"], "run", cast(str, cell))
    elif kind == "stopped" and pair is None:
        count = row["successful_optimizer_steps"]
        schedule = _single_profile(request).schedule
        if type(count) is not int or not 1 <= count <= schedule.warmup_steps + schedule.steps:
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


def _single_profile(request: AnyProbeRequest) -> HProfile:
    """The one profile of a single-profile request (never called for a pair)."""
    if not isinstance(request, ProbeRequest):
        raise ValueError("a pair has no single profile")
    return request.profile
