"""Canonical wire codec for D1 fresh, resume, and durable-failure handoffs."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import NoReturn, cast

from entropic_substrate.__legacy.evidence.d1_journal import JournalTailQuarantineReceipt
from entropic_substrate.__legacy.evidence.d1_journal_values import (
    D1JournalEvent,
    JournalCheckpointState,
    parse_event_line,
)
from entropic_substrate.__legacy.evidence.resource_ledger_codec import (
    ResourceLedgerCodecGap,
    resource_ledger_state_from_mapping,
    resource_ledger_state_to_mapping,
)
from entropic_substrate.__legacy.evidence.resource_work_codec import (
    ResourceWorkCodecGap,
    resource_snapshot_from_mapping,
    resource_snapshot_to_mapping,
)
from entropic_substrate.kernel.artifacts import (
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
    relative_d1_artifact_path,
    snapshot_d1_artifact,
)
from entropic_substrate.kernel.evidence_class.resource_provenance_codec import (
    ResourceProvenanceCodecGap,
    run_start_from_mapping,
    run_start_to_mapping,
)
from entropic_substrate.ladder.run.trainer.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
)
from entropic_substrate.ladder.run.trainer.d1_dual_failure import secondary_failure_from_mapping
from entropic_substrate.ladder.run.trainer.d1_evaluation_codec import (
    d1_evaluation_from_mapping,
    d1_evaluation_to_mapping,
)
from entropic_substrate.ladder.run.trainer.d1_resume_trainer import D1ResumeRunCompletion
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import (
    D1FreshRunCompletion,
    D1TrainerFailureHandoff,
    D1TrainerTermination,
)

__all__ = [
    "D1TrainerHandoffCodecError",
    "parse_d1_trainer_handoff",
    "serialize_d1_trainer_handoff",
    "trainer_handoff_from_mapping",
    "trainer_handoff_to_mapping",
]

_SCHEMA = "d1-trainer-handoff-wire-v1"
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_TERMINATION_KEYS = frozenset(
    {
        "execution_segment_id",
        "execution_segment_index",
        "journal_state",
        "logical_run_id",
        "model_successful_step",
        "resource_ledger_state",
        "schema_version",
        "successful_step",
        "trainer_ended_at",
    }
)
_FRESH_KEYS = frozenset(
    {
        "data_order",
        "final_checkpoint",
        "final_evaluation",
        "final_resource_work",
        "kind",
        "operator_finalization_required",
        "schema_version",
        "termination",
    }
)
_FAILURE_KEYS = frozenset(
    {
        "error_message",
        "error_type",
        "kind",
        "operator_finalization_required",
        "rollback_claimed",
        "schema_version",
        "secondary_failure",
        "termination",
        "trainer_event_kind",
    }
)
_RESUME_KEYS = frozenset(
    {
        "final_checkpoint",
        "final_evaluation",
        "final_resource_work",
        "kind",
        "operator_finalization_required",
        "resume",
        "schema_version",
        "termination",
    }
)
_RESUME_VALUE_KEYS = frozenset(
    {
        "resumed_from_sha256",
        "run_start_provenance",
        "segment_data_order",
        "tail_quarantine",
    }
)


class D1TrainerHandoffCodecError(ValueError):
    """A trainer handoff wire value is incomplete, ambiguous, or noncanonical."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1TrainerHandoffCodecError(message)
    raise D1TrainerHandoffCodecError(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _exact(value: dict[str, object], keys: frozenset[str], *, label: str) -> None:
    if frozenset(value) != keys:
        _fail(f"{label} keys must be exact")


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _fail(f"{label} must be a nonempty trimmed string")
    return value


def _nonblank_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        _fail(f"{label} must be a nonblank string")
    return value


def _integer(value: object, *, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer at least {minimum}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be a JSON boolean")
    return value


def _sha256(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return text


def _checkpoint_to_mapping(
    receipt: D1CheckpointArtifactReceipt,
    *,
    artifact_root: Path,
) -> dict[str, object]:
    if type(receipt) is not D1CheckpointArtifactReceipt:
        raise TypeError("fresh trainer handoff requires a checkpoint receipt")
    relative = relative_d1_artifact_path(
        artifact_root,
        receipt.path,
        label="trainer handoff checkpoint",
    )
    _, size, digest = snapshot_d1_artifact(
        artifact_root,
        relative,
        label="trainer handoff checkpoint",
    )
    if (size, digest) != (receipt.size_bytes, receipt.sha256):
        _fail("trainer handoff checkpoint bytes differ from its receipt")
    return {
        "path": relative,
        "sha256": receipt.sha256,
        "size_bytes": receipt.size_bytes,
    }


def _checkpoint_from_mapping(
    value: object,
    *,
    artifact_root: Path,
) -> D1CheckpointArtifactReceipt:
    item = _mapping(value, label="trainer handoff checkpoint")
    _exact(item, frozenset({"path", "sha256", "size_bytes"}), label="trainer handoff checkpoint")
    relative = normalize_d1_relative_artifact_path(
        item["path"],
        label="trainer handoff checkpoint",
    )
    path, observed_size, observed_digest = snapshot_d1_artifact(
        artifact_root,
        relative,
        label="trainer handoff checkpoint",
    )
    expected_size = _integer(item["size_bytes"], label="checkpoint size_bytes", minimum=1)
    expected_digest = _sha256(item["sha256"], label="checkpoint sha256")
    if (observed_size, observed_digest) != (expected_size, expected_digest):
        _fail("trainer handoff checkpoint bytes differ from the wire receipt")
    return D1CheckpointArtifactReceipt(path, expected_size, expected_digest)


def _stable_artifact_bytes(
    *,
    artifact_root: Path,
    relative_path: object,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes]:
    relative = normalize_d1_relative_artifact_path(relative_path, label=label)
    path, size, digest = snapshot_d1_artifact(artifact_root, relative, label=label)
    if (size, digest) != (expected_size, expected_sha256):
        _fail(f"{label} bytes differ from their receipt")
    try:
        data = path.read_bytes()
    except OSError as error:
        _fail(f"{label} bytes could not be read", error)
    if (len(data), hashlib.sha256(data).hexdigest()) != (size, digest):
        _fail(f"{label} changed after its descriptor-bound snapshot")
    return path, data


def _tail_events_from_bytes(value: bytes) -> tuple[D1JournalEvent, ...]:
    if type(value) is not bytes or not value:
        _fail("trainer handoff quarantine tail must contain event bytes")
    try:
        return tuple(parse_event_line(line) for line in value.splitlines(keepends=True))
    except (TypeError, ValueError) as error:
        _fail(f"trainer handoff quarantine tail events are invalid: {error}", error)


def _event_from_mapping(value: object) -> D1JournalEvent:
    item = _mapping(value, label="trainer handoff quarantine event")
    try:
        line = (
            json.dumps(
                item,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            + b"\n"
        )
        return parse_event_line(line)
    except (TypeError, ValueError) as error:
        _fail(f"trainer handoff quarantine event is invalid: {error}", error)


def _quarantine_to_mapping(
    value: JournalTailQuarantineReceipt | None,
    *,
    artifact_root: Path,
) -> dict[str, object] | None:
    if value is None:
        return None
    if type(value) is not JournalTailQuarantineReceipt:
        raise TypeError("resume handoff tail quarantine is invalid")
    relative = relative_d1_artifact_path(
        artifact_root,
        value.quarantine_path,
        label="trainer handoff quarantine",
    )
    _, data = _stable_artifact_bytes(
        artifact_root=artifact_root,
        relative_path=relative,
        expected_size=value.quarantined_journal_byte_count,
        expected_sha256=value.quarantined_journal_sha256,
        label="trainer handoff quarantine",
    )
    prefix = data[: value.recovered_prefix_byte_count]
    if hashlib.sha256(prefix).hexdigest() != value.recovered_prefix_sha256:
        _fail("trainer handoff quarantine retained prefix differs")
    tail_events = _tail_events_from_bytes(data[value.recovered_prefix_byte_count :])
    if tail_events != value.tail_events:
        _fail("trainer handoff quarantine tail events differ from artifact bytes")
    return {
        "artifact_path": relative,
        "evidence": value.manifest,
        "tail_events": [event.manifest for event in value.tail_events],
    }


def _quarantine_from_mapping(
    value: object,
    *,
    artifact_root: Path,
) -> JournalTailQuarantineReceipt | None:
    if value is None:
        return None
    item = _mapping(value, label="trainer handoff quarantine")
    _exact(
        item,
        frozenset({"artifact_path", "evidence", "tail_events"}),
        label="trainer handoff quarantine",
    )
    evidence = _mapping(item["evidence"], label="trainer handoff quarantine evidence")
    _exact(
        evidence,
        frozenset({"schema_version", "kind", "artifact", "retained_prefix", "tail"}),
        label="trainer handoff quarantine evidence",
    )
    if evidence["schema_version"] != "d1-tail-quarantine/v1" or evidence["kind"] != (
        "quarantined_tail"
    ):
        _fail("trainer handoff quarantine evidence schema or kind is unsupported")
    artifact = _mapping(evidence["artifact"], label="trainer handoff quarantine artifact")
    retained = _mapping(
        evidence["retained_prefix"], label="trainer handoff quarantine retained prefix"
    )
    tail = _mapping(evidence["tail"], label="trainer handoff quarantine tail")
    _exact(
        artifact,
        frozenset({"role", "file_name", "sha256", "byte_count"}),
        label="trainer handoff quarantine artifact",
    )
    _exact(
        retained,
        frozenset({"sha256", "byte_count"}),
        label="trainer handoff quarantine retained prefix",
    )
    _exact(
        tail,
        frozenset({"event_count", "first_event_sequence", "last_event_sequence"}),
        label="trainer handoff quarantine tail",
    )
    relative = normalize_d1_relative_artifact_path(
        item["artifact_path"], label="trainer handoff quarantine"
    )
    file_name = _text(artifact["file_name"], label="quarantine artifact file_name")
    if relative.rsplit("/", 1)[-1] != file_name or artifact["role"] != (
        "pre_resume_authoritative_journal"
    ):
        _fail("trainer handoff quarantine artifact identity differs")
    artifact_size = _integer(
        artifact["byte_count"], label="quarantine artifact byte_count", minimum=1
    )
    artifact_digest = _sha256(artifact["sha256"], label="quarantine artifact sha256")
    path, data = _stable_artifact_bytes(
        artifact_root=artifact_root,
        relative_path=relative,
        expected_size=artifact_size,
        expected_sha256=artifact_digest,
        label="trainer handoff quarantine",
    )
    prefix_size = _integer(retained["byte_count"], label="quarantine prefix byte_count", minimum=1)
    prefix_digest = _sha256(retained["sha256"], label="quarantine prefix sha256")
    if prefix_size >= len(data) or hashlib.sha256(data[:prefix_size]).hexdigest() != prefix_digest:
        _fail("trainer handoff quarantine retained prefix differs")
    artifact_events = _tail_events_from_bytes(data[prefix_size:])
    raw_events = item["tail_events"]
    if type(raw_events) is not list:
        _fail("trainer handoff quarantine tail_events must be a JSON array")
    wire_events = tuple(_event_from_mapping(event) for event in raw_events)
    if wire_events != artifact_events:
        _fail("trainer handoff quarantine wire events differ from artifact bytes")
    try:
        return JournalTailQuarantineReceipt(
            quarantine_path=path,
            quarantined_journal_sha256=artifact_digest,
            quarantined_journal_byte_count=artifact_size,
            recovered_prefix_sha256=prefix_digest,
            recovered_prefix_byte_count=prefix_size,
            tail_event_count=_integer(
                tail["event_count"], label="quarantine tail event_count", minimum=1
            ),
            first_tail_event_sequence=_integer(
                tail["first_event_sequence"],
                label="quarantine first_event_sequence",
                minimum=0,
            ),
            last_tail_event_sequence=_integer(
                tail["last_event_sequence"],
                label="quarantine last_event_sequence",
                minimum=0,
            ),
            tail_events=wire_events,
        )
    except (TypeError, ValueError) as error:
        _fail(f"trainer handoff quarantine receipt is invalid: {error}", error)


def _termination_to_mapping(value: D1TrainerTermination) -> dict[str, object]:
    if type(value) is not D1TrainerTermination:
        raise TypeError("trainer handoff requires a D1TrainerTermination")
    return {
        "execution_segment_id": value.execution_segment_id,
        "execution_segment_index": value.execution_segment_index,
        "journal_state": value.journal_state.manifest,
        "logical_run_id": value.logical_run_id,
        "model_successful_step": value.model_successful_step,
        "resource_ledger_state": resource_ledger_state_to_mapping(value.resource_ledger_state),
        "schema_version": value.schema_version,
        "successful_step": value.successful_step,
        "trainer_ended_at": value.trainer_ended_at.isoformat().replace("+00:00", "Z"),
    }


def _termination_from_mapping(value: object) -> D1TrainerTermination:
    item = _mapping(value, label="trainer handoff termination")
    _exact(item, _TERMINATION_KEYS, label="trainer handoff termination")
    timestamp = item["trainer_ended_at"]
    if type(timestamp) is not str or _UTC.fullmatch(timestamp) is None:
        _fail("trainer handoff termination timestamp must be canonical UTC Z time")
    try:
        return D1TrainerTermination(
            logical_run_id=_text(item["logical_run_id"], label="termination logical_run_id"),
            execution_segment_index=_integer(
                item["execution_segment_index"],
                label="termination execution_segment_index",
                minimum=0,
            ),
            execution_segment_id=_text(
                item["execution_segment_id"], label="termination execution_segment_id"
            ),
            trainer_ended_at=datetime.fromisoformat(timestamp[:-1] + "+00:00"),
            successful_step=_integer(
                item["successful_step"], label="termination successful_step", minimum=0
            ),
            model_successful_step=_integer(
                item["model_successful_step"],
                label="termination model_successful_step",
                minimum=0,
            ),
            journal_state=JournalCheckpointState.from_mapping(item["journal_state"]),
            resource_ledger_state=resource_ledger_state_from_mapping(item["resource_ledger_state"]),
            schema_version=_text(item["schema_version"], label="termination schema_version"),
        )
    except D1TrainerHandoffCodecError:
        raise
    except (TypeError, ValueError, ResourceLedgerCodecGap) as error:
        _fail(f"trainer handoff termination is invalid: {error}", error)


def trainer_handoff_to_mapping(
    handoff: D1FreshRunCompletion | D1ResumeRunCompletion | D1TrainerFailureHandoff,
    *,
    artifact_root: Path,
) -> dict[str, object]:
    root = normalize_d1_artifact_root(artifact_root)
    if type(handoff) is D1FreshRunCompletion:
        return {
            "data_order": {
                "batches_attested": handoff.data_order_batches_attested,
                "sha256": handoff.data_order_sha256,
            },
            "final_checkpoint": _checkpoint_to_mapping(
                handoff.final_checkpoint,
                artifact_root=root,
            ),
            "final_evaluation": d1_evaluation_to_mapping(handoff.final_evaluation),
            "final_resource_work": resource_snapshot_to_mapping(handoff.final_resource_work),
            "kind": "fresh_completion",
            "operator_finalization_required": handoff.operator_finalization_required,
            "schema_version": _SCHEMA,
            "termination": _termination_to_mapping(handoff.termination),
        }
    if type(handoff) is D1ResumeRunCompletion:
        return {
            "final_checkpoint": _checkpoint_to_mapping(
                handoff.final_checkpoint,
                artifact_root=root,
            ),
            "final_evaluation": d1_evaluation_to_mapping(handoff.final_evaluation),
            "final_resource_work": resource_snapshot_to_mapping(handoff.final_resource_work),
            "kind": "resume_completion",
            "operator_finalization_required": handoff.operator_finalization_required,
            "resume": {
                "resumed_from_sha256": handoff.resumed_from_sha256,
                "run_start_provenance": run_start_to_mapping(handoff.run_start_provenance),
                "segment_data_order": {
                    "batches_attested": handoff.segment_data_order_batches_attested,
                    "sha256": handoff.segment_data_order_sha256,
                },
                "tail_quarantine": _quarantine_to_mapping(
                    handoff.tail_quarantine,
                    artifact_root=root,
                ),
            },
            "schema_version": _SCHEMA,
            "termination": _termination_to_mapping(handoff.termination),
        }
    if type(handoff) is D1TrainerFailureHandoff:
        return {
            "error_message": handoff.error_message,
            "error_type": handoff.error_type,
            "kind": "trainer_failure",
            "operator_finalization_required": handoff.operator_finalization_required,
            "rollback_claimed": handoff.rollback_claimed,
            "schema_version": _SCHEMA,
            "secondary_failure": (
                None if handoff.secondary_failure is None else handoff.secondary_failure.manifest
            ),
            "termination": _termination_to_mapping(handoff.termination),
            "trainer_event_kind": handoff.trainer_event_kind,
        }
    raise TypeError("trainer handoff codec requires a fresh, resume, or failure outcome")


def trainer_handoff_from_mapping(
    value: object,
    *,
    artifact_root: Path,
) -> D1FreshRunCompletion | D1ResumeRunCompletion | D1TrainerFailureHandoff:
    root = normalize_d1_artifact_root(artifact_root)
    item = _mapping(value, label="trainer handoff")
    if item.get("schema_version") != _SCHEMA:
        _fail(f"trainer handoff schema_version must be {_SCHEMA}")
    kind = item.get("kind")
    try:
        if kind == "fresh_completion":
            _exact(item, _FRESH_KEYS, label="fresh trainer handoff")
            order = _mapping(item["data_order"], label="fresh trainer data_order")
            _exact(
                order,
                frozenset({"batches_attested", "sha256"}),
                label="fresh trainer data_order",
            )
            return D1FreshRunCompletion(
                termination=_termination_from_mapping(item["termination"]),
                final_checkpoint=_checkpoint_from_mapping(
                    item["final_checkpoint"],
                    artifact_root=root,
                ),
                final_evaluation=d1_evaluation_from_mapping(item["final_evaluation"]),
                final_resource_work=resource_snapshot_from_mapping(item["final_resource_work"]),
                data_order_sha256=_sha256(order["sha256"], label="data_order sha256"),
                data_order_batches_attested=_integer(
                    order["batches_attested"],
                    label="data_order batches_attested",
                    minimum=1,
                ),
                operator_finalization_required=_boolean(
                    item["operator_finalization_required"],
                    label="fresh operator_finalization_required",
                ),
            )
        if kind == "resume_completion":
            _exact(item, _RESUME_KEYS, label="resume trainer handoff")
            resume = _mapping(item["resume"], label="resume trainer values")
            _exact(resume, _RESUME_VALUE_KEYS, label="resume trainer values")
            order = _mapping(
                resume["segment_data_order"],
                label="resume trainer segment_data_order",
            )
            _exact(
                order,
                frozenset({"batches_attested", "sha256"}),
                label="resume trainer segment_data_order",
            )
            return D1ResumeRunCompletion(
                termination=_termination_from_mapping(item["termination"]),
                final_checkpoint=_checkpoint_from_mapping(
                    item["final_checkpoint"],
                    artifact_root=root,
                ),
                final_evaluation=d1_evaluation_from_mapping(item["final_evaluation"]),
                final_resource_work=resource_snapshot_from_mapping(item["final_resource_work"]),
                resumed_from_sha256=_sha256(
                    resume["resumed_from_sha256"], label="resumed_from_sha256"
                ),
                run_start_provenance=run_start_from_mapping(resume["run_start_provenance"]),
                segment_data_order_sha256=_sha256(
                    order["sha256"], label="segment_data_order sha256"
                ),
                segment_data_order_batches_attested=_integer(
                    order["batches_attested"],
                    label="segment_data_order batches_attested",
                    minimum=1,
                ),
                tail_quarantine=_quarantine_from_mapping(
                    resume["tail_quarantine"],
                    artifact_root=root,
                ),
                operator_finalization_required=_boolean(
                    item["operator_finalization_required"],
                    label="resume operator_finalization_required",
                ),
            )
        if kind == "trainer_failure":
            _exact(item, _FAILURE_KEYS, label="failure trainer handoff")
            secondary = item["secondary_failure"]
            return D1TrainerFailureHandoff(
                termination=_termination_from_mapping(item["termination"]),
                error_type=_nonblank_text(item["error_type"], label="failure error_type"),
                error_message=_nonblank_text(item["error_message"], label="failure error_message"),
                secondary_failure=(
                    None if secondary is None else secondary_failure_from_mapping(secondary)
                ),
                trainer_event_kind=_text(
                    item["trainer_event_kind"], label="failure trainer_event_kind"
                ),
                rollback_claimed=_boolean(
                    item["rollback_claimed"], label="failure rollback_claimed"
                ),
                operator_finalization_required=_boolean(
                    item["operator_finalization_required"],
                    label="failure operator_finalization_required",
                ),
            )
        _fail("trainer handoff kind is unsupported")
    except D1TrainerHandoffCodecError:
        raise
    except (
        TypeError,
        ValueError,
        ResourceProvenanceCodecGap,
        ResourceWorkCodecGap,
    ) as error:
        _fail(f"trainer handoff {kind or 'unknown'} is invalid: {error}", error)


def serialize_d1_trainer_handoff(
    handoff: D1FreshRunCompletion | D1ResumeRunCompletion | D1TrainerFailureHandoff,
    *,
    artifact_root: Path,
) -> bytes:
    return json.dumps(
        trainer_handoff_to_mapping(handoff, artifact_root=artifact_root),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"trainer handoff contains duplicate key {key!r}")
        result[key] = value
    return result


def parse_d1_trainer_handoff(
    payload: bytes,
    *,
    artifact_root: Path,
) -> D1FreshRunCompletion | D1ResumeRunCompletion | D1TrainerFailureHandoff:
    if type(payload) is not bytes or not payload:
        _fail("trainer handoff payload must be nonempty bytes")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda constant: _fail(f"non-finite JSON constant {constant}"),
        )
    except D1TrainerHandoffCodecError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("trainer handoff payload must be UTF-8 JSON", error)
    handoff = trainer_handoff_from_mapping(value, artifact_root=artifact_root)
    if serialize_d1_trainer_handoff(handoff, artifact_root=artifact_root) != payload:
        _fail("trainer handoff payload must use canonical JSON")
    return handoff
