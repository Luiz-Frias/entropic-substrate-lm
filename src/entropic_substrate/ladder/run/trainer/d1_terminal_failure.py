"""Join a typed numeric source observation to the trainer lifecycle cursor.

Also owns the one terminal failure sequence the fresh and resume trainers share.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from entropic_substrate.kernel.evidence_class.resource_provenance import RunStartProvenance
from entropic_substrate.ladder.capture.reader.d1_numerical_defect import (
    D1DefectCheckpointReference,
    D1NonfiniteMetricError,
    D1TerminalNumericalDefectRecord,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    FrozenJson,
    JournalEventKind,
)
from entropic_substrate.ladder.capture.resource.resource_ledger import (
    ResourceLedger,
    ResourceLedgerState,
)
from entropic_substrate.ladder.register.d1_plan import D1RunPlan
from entropic_substrate.ladder.run.trainer.d1_dual_failure import (
    D1PrimarySecondaryFailure,
    D1SecondaryFailureEvidence,
    D1SecondaryFailureStage,
)
from entropic_substrate.ladder.run.trainer.d1_segment_engine import D1SegmentJournal
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import D1TrainerClock
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import (
    D1TrainerFailureHandoff,
    capture_d1_trainer_termination,
)

__all__ = [
    "_trainer_failure",
    "next_trainer_event",
    "terminal_numerical_defect_for_failure",
    "trainer_event",
]


def terminal_numerical_defect_for_failure(
    error: Exception,
    *,
    plan: D1RunPlan,
    start: RunStartProvenance,
    resource_state: ResourceLedgerState,
    successful_step: int,
    checkpoint_sha256: str | None,
) -> D1TerminalNumericalDefectRecord | None:
    """Enrich only a typed defect observed at the registered terminal cursor."""
    if type(error) is not D1NonfiniteMetricError:
        return None
    maximum = plan.profile.draft.execution_protocol.optimization.max_steps
    if successful_step != maximum:
        return None
    if resource_state.logical_run_id != plan.logical_run_id:
        raise ValueError("terminal defect resource ledger belongs to another run")
    if resource_state.execution_segment_index != start.execution_segment_index:
        raise ValueError("terminal defect resource ledger belongs to another segment")
    checkpoint = (
        D1DefectCheckpointReference.receipt(checkpoint_sha256)
        if checkpoint_sha256 is not None
        else D1DefectCheckpointReference.gap("no_prior_checkpoint")
    )
    return D1TerminalNumericalDefectRecord.from_error(
        error,
        plan=plan,
        start=start,
        successful_step=successful_step,
        eval_index=resource_state.next_eval_index,
        checkpoint=checkpoint,
    )


def trainer_event(
    plan: D1RunPlan,
    kind: JournalEventKind,
    *,
    sequence: int,
    successful_step: int,
    payload: Mapping[str, object],
) -> D1JournalEvent:
    return D1JournalEvent(
        event_kind=kind,
        event_seq=sequence,
        logical_run_id=plan.logical_run_id,
        execution_digest=plan.run.execution_digest,
        aggregation_digest=plan.run.aggregation_digest,
        construction_digest=plan.run.construction_digest,
        family_digest=plan.profile.family.family_digest,
        run_class=plan.run.run_class,
        successful_step=successful_step,
        payload=cast(Mapping[str, FrozenJson], payload),
    )


def next_trainer_event(
    journal: D1SegmentJournal,
    plan: D1RunPlan,
    kind: JournalEventKind,
    *,
    successful_step: int,
    payload: Mapping[str, object],
) -> D1JournalEvent:
    return trainer_event(
        plan,
        kind,
        sequence=journal.state.last_event_sequence + 1,
        successful_step=successful_step,
        payload=payload,
    )


def _trainer_failure(
    plan: D1RunPlan,
    journal: D1SegmentJournal,
    ledger: ResourceLedger,
    start: RunStartProvenance,
    clock: D1TrainerClock,
    step: int,
    model_step: int,
    error: Exception,
    checkpoint_sha256: str | None,
) -> tuple[D1TrainerFailureHandoff, BaseException]:
    trainer_ended_at = clock.utc_now()
    secondary_failure: D1SecondaryFailureEvidence | None = None
    failure_cause: BaseException = error
    try:
        ledger.record_meter_observation(monotonic_ns=clock.monotonic_ns())
    except Exception as meter_error:
        secondary_failure = D1SecondaryFailureEvidence.from_error(
            D1SecondaryFailureStage.TERMINAL_METER_REFRESH,
            meter_error,
        )
        failure_cause = D1PrimarySecondaryFailure(
            error,
            meter_error,
            secondary_stage=D1SecondaryFailureStage.TERMINAL_METER_REFRESH,
        )
    journal_cursor = journal.state
    resource_state = ledger.current_state()
    error_message = str(error) or repr(error)
    payload: dict[str, object] = {
        "error_message": error_message,
        "error_type": type(error).__name__,
        "execution_segment_id": start.execution_segment_id,
        "execution_segment_index": start.execution_segment_index,
        "journal_cursor": journal_cursor.manifest,
        "logical_run_id": start.logical_run_id,
        "model_successful_step": model_step,
        "operator_finalization_required": True,
        "resource_ledger_state": resource_state.manifest,
        "rollback_claimed": False,
        "secondary_failure": (None if secondary_failure is None else secondary_failure.manifest),
        "trainer_ended_at": trainer_ended_at.isoformat().replace("+00:00", "Z"),
    }
    numerical_defect = terminal_numerical_defect_for_failure(
        error,
        plan=plan,
        start=start,
        resource_state=resource_state,
        successful_step=step,
        checkpoint_sha256=checkpoint_sha256,
    )
    if numerical_defect is not None:
        payload["numerical_defect"] = numerical_defect.manifest
    try:
        journal.append(
            next_trainer_event(
                journal,
                plan,
                JournalEventKind.TRAINER_ABORTED,
                successful_step=step,
                payload=payload,
            )
        )
    except Exception as append_error:
        raise D1PrimarySecondaryFailure(
            failure_cause,
            append_error,
            secondary_stage=D1SecondaryFailureStage.TRAINER_ABORT_APPEND,
        ) from error
    termination = capture_d1_trainer_termination(
        start=start,
        trainer_ended_at=trainer_ended_at,
        successful_step=step,
        model_successful_step=model_step,
        journal_state=journal.state,
        resource_ledger_state=resource_state,
    )
    return (
        D1TrainerFailureHandoff.from_error(
            termination,
            error,
            secondary_failure=secondary_failure,
        ),
        failure_cause,
    )
