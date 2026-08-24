"""Strict interpretation of a TRAINER_ABORTED terminal numerical defect."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from entropic_substrate.d1_numerical_defect import (
    D1CheckpointReferenceKind,
    D1TerminalNumericalDefectRecord,
    terminal_numerical_defect_from_mapping,
)
from entropic_substrate.evidence.d1_journal import D1JournalError, read_journal
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.evidence.d1_telemetry_values import (
    D1RunEvidence,
    D1RunStatus,
    D1TelemetryGap,
)

__all__ = ["validate_d1_terminal_defect"]


def _gap(code: str, path: str, detail: str) -> D1TelemetryGap:
    return D1TelemetryGap(code=code, path=path, detail=detail)


def _plan_value(plan: Mapping[str, object], field: str) -> object:
    if field not in plan:
        raise _gap("terminal_defect_identity", f"plan.{field}", "is absent")
    return plan[field]


def validate_d1_terminal_defect(
    path: str | Path,
    run: D1RunEvidence,
) -> D1TerminalNumericalDefectRecord:
    """Validate the canonical nested defect after whole-journal admission."""
    target = Path(path)
    if type(run) is not D1RunEvidence:
        raise TypeError("terminal defect validation requires D1RunEvidence")
    if run.path != target:
        raise _gap("terminal_defect_identity", str(target), "run evidence belongs to another path")
    if run.status is not D1RunStatus.TRAINER_ABORTED:
        raise _gap(
            "terminal_defect_status",
            "event_kind",
            "terminal numerical defect requires trainer_aborted run evidence",
        )
    try:
        events = read_journal(target)
    except D1JournalError as error:
        raise _gap(f"journal_{error.code}", error.path, str(error)) from error
    aborted = tuple(
        event for event in events if event.event_kind is JournalEventKind.TRAINER_ABORTED
    )
    if len(aborted) != 1:
        raise _gap(
            "terminal_defect_count",
            "event_kind",
            "requires exactly one TRAINER_ABORTED event",
        )
    raw = aborted[0].payload.get("numerical_defect")
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
    if observed != expected or record.successful_step != aborted[0].successful_step:
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
