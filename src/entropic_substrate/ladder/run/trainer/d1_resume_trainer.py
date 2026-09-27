"""D1 continuation loop over a journal-first exact-resume admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import entropic_substrate.ladder.run.trainer.d1_segment_engine as segment_engine
from entropic_substrate.kernel.evidence_class.resource_provenance import (
    RunStartProvenance,
    SegmentStartMode,
)
from entropic_substrate.ladder.capture.adapters.d1_observability import (
    D1JournalObserver,
    ObservedD1Journal,
)
from entropic_substrate.ladder.capture.records.d1_journal import JournalTailQuarantineReceipt
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    FrozenJson,
    JournalCheckpointState,
    JournalEventKind,
)
from entropic_substrate.ladder.capture.resource.resource_ledger import (
    ResourceLedger,
)
from entropic_substrate.ladder.capture.resource.resource_work import ResourceWorkSnapshot
from entropic_substrate.ladder.register.d1_plan import D1RunPlan
from entropic_substrate.ladder.run.resource.d1_resource_activation import (
    D1ResourceActivationFactory,
)
from entropic_substrate.ladder.run.trainer.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
)
from entropic_substrate.ladder.run.trainer.d1_dual_failure import (
    D1PrimarySecondaryFailure,
    D1SecondaryFailureEvidence,
    D1SecondaryFailureStage,
)
from entropic_substrate.ladder.run.trainer.d1_evaluation import D1EvalResult
from entropic_substrate.ladder.run.trainer.d1_resume_runtime import (
    D1ResumeRunLocations,
    open_d1_resume,
    restore_d1_resume_runtime,
)
from entropic_substrate.ladder.run.trainer.d1_terminal_failure import (
    terminal_numerical_defect_for_failure,
)
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import (
    D1TrainerClock,
    D1TrainerWallOrigin,
    SystemD1TrainerClock,
)
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import (
    D1TrainerFailureHandoff,
    D1TrainerTermination,
    capture_d1_trainer_termination,
)

__all__ = [
    "D1ResumeRunCompletion",
    "D1ResumeRunFailure",
    "D1ResumeRunLocations",
    "run_d1_resume",
]


def _require_lower_sha256(value: str, *, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class D1ResumeRunCompletion:
    termination: D1TrainerTermination
    final_checkpoint: D1CheckpointArtifactReceipt
    final_evaluation: D1EvalResult
    final_resource_work: ResourceWorkSnapshot
    resumed_from_sha256: str
    run_start_provenance: RunStartProvenance
    segment_data_order_sha256: str
    segment_data_order_batches_attested: int
    tail_quarantine: JournalTailQuarantineReceipt | None
    operator_finalization_required: bool = True

    def __post_init__(self) -> None:
        if type(self.termination) is not D1TrainerTermination:
            raise TypeError("D1 resume completion requires trainer termination")
        if self.successful_step < 1:
            raise ValueError("D1 resume completion successful_step must be positive")
        if self.journal_state.successful_step != self.successful_step:
            raise ValueError("D1 resume completion journal cursor differs")
        if self.termination.model_successful_step != self.successful_step:
            raise ValueError("D1 resume completion model and durable cursors differ")
        ledger = self.termination.resource_ledger_state
        if ledger.last_training_step != self.successful_step:
            raise ValueError("D1 resume completion resource ledger cursor differs")
        if type(self.final_checkpoint) is not D1CheckpointArtifactReceipt:
            raise TypeError("D1 resume completion requires checkpoint receipt")
        if type(self.final_evaluation) is not D1EvalResult:
            raise TypeError("D1 resume completion requires final evaluation")
        if type(self.final_resource_work) is not ResourceWorkSnapshot:
            raise TypeError("D1 resume completion requires final resource work")
        work = self.final_resource_work
        if (
            work.logical_run_id != self.termination.logical_run_id
            or work.execution_segment_index != self.termination.execution_segment_index
            or work.training_step != self.successful_step
            or self.termination.trainer_ended_at < work.observed_at
        ):
            raise ValueError("D1 resume completion resource snapshot cursor differs")
        for field in (
            "committed_quality_source_bytes",
            "committed_quality_model_tokens",
            "online_train_token_positions",
            "target_train_token_positions",
            "baseline_calibration_positions",
            "eval_positions",
        ):
            if getattr(work, field) != getattr(ledger, field):
                raise ValueError(f"D1 resume completion resource evidence differs at {field}")
        start = self.run_start_provenance
        if type(start) is not RunStartProvenance:
            raise TypeError("D1 resume completion requires run-start provenance")
        if (
            start.segment_start_mode is not SegmentStartMode.RESUME
            or start.execution_segment_index < 1
            or (
                start.logical_run_id,
                start.execution_segment_index,
                start.execution_segment_id,
            )
            != (
                self.termination.logical_run_id,
                self.termination.execution_segment_index,
                self.termination.execution_segment_id,
            )
            or self.termination.trainer_ended_at < start.trainer_started_at
        ):
            raise ValueError("D1 resume completion segment identity differs")
        _require_lower_sha256(self.resumed_from_sha256, label="resumed_from_sha256")
        if start.resume_checkpoint_sha256 != self.resumed_from_sha256:
            raise ValueError("D1 resume completion checkpoint lineage differs")
        _require_lower_sha256(
            self.segment_data_order_sha256,
            label="segment_data_order_sha256",
        )
        if (
            type(self.segment_data_order_batches_attested) is not int
            or self.segment_data_order_batches_attested < 1
        ):
            raise ValueError("D1 resume completion must attest a data-order batch")
        if (
            self.tail_quarantine is not None
            and type(self.tail_quarantine) is not JournalTailQuarantineReceipt
        ):
            raise TypeError("D1 resume completion tail quarantine is invalid")
        if self.operator_finalization_required is not True:
            raise ValueError("D1 resumed trainer cannot finalize provider billing or RUN_END")

    @property
    def successful_step(self) -> int:
        return self.termination.successful_step

    @property
    def journal_state(self) -> JournalCheckpointState:
        return self.termination.journal_state


class D1ResumeRunFailure(RuntimeError):
    """Typed resumed-run failure carrying the finalizable trainer handoff."""

    def __init__(self, handoff: D1TrainerFailureHandoff) -> None:
        if type(handoff) is not D1TrainerFailureHandoff:
            raise TypeError("D1ResumeRunFailure requires D1TrainerFailureHandoff")
        self.handoff = handoff
        super().__init__(
            f"D1 resumed trainer failed after step {handoff.termination.successful_step}: "
            f"{handoff.error_type}: {handoff.error_message}"
        )


def _event(
    plan: D1RunPlan,
    journal: ObservedD1Journal,
    kind: JournalEventKind,
    step: int,
    payload: Mapping[str, object],
) -> D1JournalEvent:
    return D1JournalEvent(
        event_kind=kind,
        event_seq=journal.state.last_event_sequence + 1,
        logical_run_id=plan.logical_run_id,
        execution_digest=plan.run.execution_digest,
        aggregation_digest=plan.run.aggregation_digest,
        construction_digest=plan.run.construction_digest,
        family_digest=plan.profile.family.family_digest,
        run_class=plan.run.run_class,
        successful_step=step,
        payload=cast(Mapping[str, FrozenJson], payload),
    )


def _trainer_failure(
    plan: D1RunPlan,
    journal: ObservedD1Journal,
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
            _event(
                plan,
                journal,
                JournalEventKind.TRAINER_ABORTED,
                step,
                payload,
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


def run_d1_resume(
    *,
    plan: D1RunPlan,
    start: RunStartProvenance,
    origin: D1TrainerWallOrigin,
    locations: D1ResumeRunLocations,
    observer: D1JournalObserver,
    clock: D1TrainerClock | None = None,
    resource_activation_factory: D1ResourceActivationFactory | None = None,
    checkpoint_sync: segment_engine.D1CheckpointSyncPort | None = None,
) -> D1ResumeRunCompletion:
    """Record the segment, restore exact state, and stop before billing finalization."""
    active_clock = clock or SystemD1TrainerClock()
    if not isinstance(cast(object, active_clock), D1TrainerClock):
        raise TypeError("D1 resume clock must implement D1TrainerClock")
    admission = open_d1_resume(
        plan=plan,
        start=start,
        origin=origin,
        locations=locations,
        observer=observer,
        resource_activation_factory=resource_activation_factory,
    )
    step = admission.successful_step
    model_step = step
    ledger = admission.resource_ledger
    latest_checkpoint_sha256 = admission.checkpoint_receipt.sha256
    with admission.journal as journal, admission.resource_activation:
        try:
            restored = restore_d1_resume_runtime(admission)
            components = restored.components
            training = segment_engine.run_d1_training_segment(
                components=components,
                journal=journal,
                clock=active_clock,
                meter=admission.resource_activation.meter,
                baseline=restored.baseline,
                journal_path=locations.journal_path,
                checkpoint_path=locations.checkpoint_path,
                checkpoint_sync=checkpoint_sync,
                successful_step=step,
                best_nats_per_byte=restored.best_nats_per_byte,
            )
            step = training.successful_step
            model_step = training.successful_step
            latest_checkpoint_sha256 = training.final_checkpoint.sha256
            termination = capture_d1_trainer_termination(
                start=start,
                trainer_ended_at=active_clock.utc_now(),
                successful_step=step,
                model_successful_step=model_step,
                journal_state=journal.state,
                resource_ledger_state=ledger.current_state(),
            )
            return D1ResumeRunCompletion(
                termination=termination,
                final_checkpoint=training.final_checkpoint,
                final_evaluation=training.final_evaluation,
                final_resource_work=training.final_resource_work,
                resumed_from_sha256=admission.checkpoint_receipt.sha256,
                run_start_provenance=start,
                segment_data_order_sha256=training.data_order_sha256,
                segment_data_order_batches_attested=training.data_order_batches_attested,
                tail_quarantine=journal.tail_quarantine,
            )
        except Exception as caught:
            if isinstance(caught, segment_engine.D1TrainingInterrupted):
                step = caught.durable_successful_step
                model_step = caught.model_successful_step
                latest_checkpoint_sha256 = caught.latest_checkpoint_sha256
                error = caught.primary
            else:
                error = caught
            handoff, failure_cause = _trainer_failure(
                plan,
                journal,
                ledger,
                start,
                active_clock,
                step,
                model_step,
                error,
                latest_checkpoint_sha256,
            )
            raise D1ResumeRunFailure(handoff) from failure_cause
