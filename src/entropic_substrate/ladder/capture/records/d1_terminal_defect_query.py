"""Strict interpretation of a TRAINER_ABORTED terminal numerical defect."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from entropic_substrate.ladder.capture.reader.d1_numerical_defect import (
    D1CheckpointReferenceKind,
    D1TerminalNumericalDefectRecord,
    terminal_numerical_defect_from_mapping,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    JournalEventKind,
)
from entropic_substrate.ladder.capture.records.d1_telemetry_values import (
    D1RunEvidence,
    D1RunStatus,
    D1TelemetryGap,
)

__all__ = ["validate_d1_terminal_defect", "validate_d1_terminal_defect_event"]


def _gap(code: str, path: str, detail: str) -> D1TelemetryGap:
    return D1TelemetryGap(code=code, path=path, detail=detail)


def _plan_value(plan: Mapping[str, object], field: str) -> object:
    if field not in plan:
        raise _gap("terminal_defect_identity", f"plan.{field}", "is absent")
    return plan[field]


def validate_d1_terminal_defect_event(
    run: D1RunEvidence,
    event: D1JournalEvent,
) -> D1TerminalNumericalDefectRecord:
    """Validate one retained TRAINER_ABORTED event against admitted run evidence."""
    if type(run) is not D1RunEvidence:
        raise TypeError("terminal defect validation requires D1RunEvidence")
    if run.status is not D1RunStatus.TRAINER_ABORTED:
        raise _gap(
            "terminal_defect_status",
            "event_kind",
            "terminal numerical defect requires trainer_aborted run evidence",
        )
    if (
        type(event) is not D1JournalEvent
        or event.event_kind is not JournalEventKind.TRAINER_ABORTED
    ):
        raise _gap(
            "terminal_defect_event",
            "event_kind",
            "requires one retained TRAINER_ABORTED event",
        )
    raw = event.payload.get("numerical_defect")
    if raw is None:
        raise _gap(
            "terminal_defect_missing",
            "trainer_aborted.numerical_defect",
            "trainer abort does not carry canonical numerical_defect evidence",
        )
    try:
        record = terminal_numerical_defect_from_mapping(raw)
    except (TypeError, ValueError) as error:
        raise _gap(
            "terminal_defect_payload", "trainer_aborted.numerical_defect", str(error)
        ) from error
    identity = run.identity
    plan = cast(Mapping[str, object], run.plan_manifest)
    expected = (
        identity.logical_run_id,
        identity.run_class,
        identity.family_digest,
        identity.construction_digest,
        identity.execution_digest,
        identity.aggregation_digest,
        run.plan_digest,
        _plan_value(plan, "arm"),
        _plan_value(plan, "seed"),
        _plan_value(plan, "pairing_id"),
    )
    observed = (
        record.logical_run_id,
        record.run_class,
        record.family_digest,
        record.construction_digest,
        record.execution_digest,
        record.aggregation_digest,
        record.plan_digest,
        record.arm.value,
        record.seed,
        record.pairing_id,
    )
    if observed != expected or record.successful_step != event.successful_step:
        raise _gap(
            "terminal_defect_identity",
            "trainer_aborted.numerical_defect",
            "record identity or cursor differs from its admitted journal",
        )
    if not any(
        segment.execution_segment_index == record.execution_segment_index
        and segment.execution_segment_id == record.execution_segment_id
        for segment in run.segments
    ):
        raise _gap(
            "terminal_defect_segment",
            "trainer_aborted.numerical_defect.execution_segment_id",
            "does not name an admitted execution segment",
        )
    if record.checkpoint.kind is D1CheckpointReferenceKind.RECEIPT and not any(
        sha256 == record.checkpoint.sha256 for _, sha256 in run.checkpoint_receipts
    ):
        raise _gap(
            "terminal_defect_checkpoint",
            "trainer_aborted.numerical_defect.checkpoint",
            "receipt is absent from the admitted journal prefix",
        )
    return record


def validate_d1_terminal_defect(
    path: str | Path,
    run: D1RunEvidence,
) -> D1TerminalNumericalDefectRecord:
    """Return the retained defect without reopening the admitted journal path."""
    target = Path(path)
    if type(run) is not D1RunEvidence:
        raise TypeError("terminal defect validation requires D1RunEvidence")
    if run.path != target:
        raise _gap("terminal_defect_identity", str(target), "run evidence belongs to another path")
    if run.trainer_abort is None:
        raise _gap(
            "terminal_defect_status",
            "event_kind",
            "terminal numerical defect requires trainer_aborted run evidence",
        )
    record = validate_d1_terminal_defect_event(run, run.trainer_abort.event)
    if run.trainer_abort.numerical_defect != record:
        raise _gap(
            "terminal_defect_identity",
            "trainer_aborted.numerical_defect",
            "retained numerical defect differs from its event",
        )
    return record
