"""D1 continuation loop over a journal-first exact-resume admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from entropic_substrate.d1_calibration import D1BaselineCalibrationAttestation
from entropic_substrate.d1_evaluation import D1EvalResult, D1Evaluator
from entropic_substrate.d1_observability import D1JournalObserver, ObservedD1Journal
from entropic_substrate.d1_resume_runtime import (
    D1ResumeRunLocations,
    open_d1_resume,
    restore_d1_resume_runtime,
)
from entropic_substrate.d1_trainer_handoff import (
    D1TrainerFailureHandoff,
    D1TrainerTermination,
    capture_d1_trainer_termination,
)
from entropic_substrate.d1_trainer_runtime import (
    D1FreshComponents,
    D1TrainerClock,
    D1TrainerWallOrigin,
    SystemD1TrainerClock,
)
from entropic_substrate.d1_training_step import apply_d1_training_step
from entropic_substrate.evidence.d1_journal import JournalTailQuarantineReceipt
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalEvent,
    FrozenJson,
    JournalCheckpointState,
    JournalEventKind,
)
from entropic_substrate.evidence.resource_ledger import (
    ResourceLedger,
    ResourceMeterStrategy,
)
from entropic_substrate.evidence.resource_provenance import RunStartProvenance
from entropic_substrate.evidence.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.evidence.resource_work import ResourceWorkSnapshot
from entropic_substrate.evidence.resource_work_codec import resource_snapshot_to_mapping
from entropic_substrate.experiments.d1_plan import D1RunPlan
from entropic_substrate.predictive_state.checkpoint_envelope import build_d1_checkpoint
from entropic_substrate.predictive_state.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
    save_atomic,
)

__all__ = [
    "D1ResumeRunCompletion",
    "D1ResumeRunFailure",
    "D1ResumeRunLocations",
    "run_d1_resume",
]


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


def _evaluate(
    components: D1FreshComponents,
    journal: ObservedD1Journal,
    clock: D1TrainerClock,
    step: int,
) -> tuple[D1EvalResult, ResourceWorkSnapshot]:
    result = D1Evaluator().evaluate(system=components.system, sampler=components.eval_sampler)
    batching = components.plan.profile.draft.execution_protocol.batching
    target_positions = (
        result.batch_count * batching.batch_size * (batching.sequence_length + batching.horizon)
    )
    work = components.resource_ledger.record_eval(
        online_positions=result.online_positions,
        target_positions=target_positions,
        training_step=step,
        observed_at=clock.utc_now(),
        monotonic_ns=clock.monotonic_ns(),
    )
    journal.append(
        _event(
            components.plan,
            journal,
            JournalEventKind.EVAL,
            step,
            {
                "eval_index": work.eval_index,
                "quality": result.telemetry_manifest,
                "resource_work": resource_snapshot_to_mapping(work),
            },
        )
    )
    return result, work


def _save_checkpoint(
    components: D1FreshComponents,
    journal: ObservedD1Journal,
    path: Path,
    baseline: D1BaselineCalibrationAttestation,
    step: int,
    best_nats: float,
) -> D1CheckpointArtifactReceipt:
    state = build_d1_checkpoint(
        system=components.system,
        optimizer=components.optimizer.optimizer,
        sampler=components.train_sampler,
        run=components.plan.run,
        protocol=components.plan.profile.draft.execution_protocol,
        successful_step=step,
        baseline_calibration=baseline,
        resource_ledger=components.resource_ledger.current_state(),
        journal_state=journal.state,
        lifecycle={
            "schema_version": "d1-trainer-lifecycle-v1",
            "state": "running",
            "completion_reason": None,
            "abort_reason": None,
        },
        best_nats_per_byte=best_nats,
    )
    receipt = save_atomic(path, state)
    journal.append_checkpoint(receipt.sha256)
    return receipt


def _trainer_failure(
    plan: D1RunPlan,
    journal: ObservedD1Journal,
    ledger: ResourceLedger,
    start: RunStartProvenance,
    clock: D1TrainerClock,
    step: int,
    error: Exception,
) -> D1TrainerFailureHandoff:
    trainer_ended_at = clock.utc_now()
    journal_cursor = journal.state
    resource_state = ledger.current_state()
    error_message = str(error) or repr(error)
    journal.append(
        _event(
            plan,
            journal,
            JournalEventKind.TRAINER_ABORTED,
            step,
            {
                "error_message": error_message,
                "error_type": type(error).__name__,
                "execution_segment_id": start.execution_segment_id,
                "execution_segment_index": start.execution_segment_index,
                "journal_cursor": journal_cursor.manifest,
                "logical_run_id": start.logical_run_id,
                "operator_finalization_required": True,
                "resource_ledger_state": resource_state.manifest,
                "rollback_claimed": False,
                "trainer_ended_at": trainer_ended_at.isoformat().replace("+00:00", "Z"),
            },
        )
    )
    termination = capture_d1_trainer_termination(
        start=start,
        trainer_ended_at=trainer_ended_at,
        successful_step=step,
        journal_state=journal.state,
        resource_ledger_state=resource_state,
    )
    return D1TrainerFailureHandoff.from_error(termination, error)


def run_d1_resume(
    *,
    plan: D1RunPlan,
    start: RunStartProvenance,
    origin: D1TrainerWallOrigin,
    locations: D1ResumeRunLocations,
    observer: D1JournalObserver,
    clock: D1TrainerClock | None = None,
    meter_strategy: ResourceMeterStrategy | None = None,
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
        meter_strategy=meter_strategy,
    )
    step = admission.successful_step
    ledger = admission.resource_ledger
    with admission.journal as journal:
        try:
            restored = restore_d1_resume_runtime(admission)
            components = restored.components
            best_nats = restored.best_nats_per_byte
            protocol = plan.profile.draft.execution_protocol
            final_eval: D1EvalResult | None = None
            final_work: ResourceWorkSnapshot | None = None
            final_receipt: D1CheckpointArtifactReceipt | None = None
            while step < protocol.optimization.max_steps:
                batch = components.train_sampler.predictive_batch(protocol.batching.horizon)
                online, target = (
                    batch.online_observations.numel(),
                    batch.target_encoder_context.numel(),
                )
                try:
                    result = apply_d1_training_step(
                        system=components.system,
                        optimizer_bundle=components.optimizer,
                        batch=batch,
                        spec=protocol.optimization,
                        successful_step=step,
                    )
                except BaseException:
                    ledger.record_train_attempt(
                        online_positions=online,
                        target_positions=target,
                        quality_source_byte_exposures=online,
                        committed=False,
                    )
                    raise
                step = result.successful_step
                ledger.record_train_attempt(
                    online_positions=online,
                    target_positions=target,
                    quality_source_byte_exposures=online,
                    committed=True,
                )
                order = components.train_sampler.data_order_digest()
                if order is None:
                    raise RuntimeError("D1 resumed segment has no data-order attestation")
                journal.append(
                    _event(
                        plan,
                        journal,
                        JournalEventKind.TRAIN,
                        step,
                        {
                            "data_order": {"batches_attested": order[1], "sha256": order[0]},
                            "execution_segment": run_start_to_mapping(start),
                            "scheduled_log_boundary": step % protocol.evaluation.log_interval == 0,
                            "train": result.manifest,
                        },
                    )
                )
                terminal = step == protocol.optimization.max_steps
                if step % protocol.evaluation.eval_interval == 0 or terminal:
                    final_eval, final_work = _evaluate(components, journal, active_clock, step)
                    best_nats = min(best_nats, final_eval.nats_per_byte)
                    scheduled = protocol.evaluation.save_interval > 0 and (
                        step % protocol.evaluation.save_interval == 0
                    )
                    if scheduled or terminal:
                        final_receipt = _save_checkpoint(
                            components,
                            journal,
                            locations.checkpoint_path,
                            restored.baseline,
                            step,
                            best_nats,
                        )
            if final_eval is None or final_work is None or final_receipt is None:
                raise RuntimeError("D1 resumed terminal boundary lacks evaluation/checkpoint")
            order = components.train_sampler.data_order_digest()
            if order is None:
                raise RuntimeError("D1 resumed segment lacks data-order evidence")
            termination = capture_d1_trainer_termination(
                start=start,
                trainer_ended_at=active_clock.utc_now(),
                successful_step=step,
                journal_state=journal.state,
                resource_ledger_state=ledger.current_state(),
            )
            return D1ResumeRunCompletion(
                termination=termination,
                final_checkpoint=final_receipt,
                final_evaluation=final_eval,
                final_resource_work=final_work,
                resumed_from_sha256=admission.checkpoint_receipt.sha256,
                run_start_provenance=start,
                segment_data_order_sha256=order[0],
                segment_data_order_batches_attested=order[1],
                tail_quarantine=journal.tail_quarantine,
            )
        except Exception as error:
            handoff = _trainer_failure(
                plan,
                journal,
                ledger,
                start,
                active_clock,
                step,
                error,
            )
            raise D1ResumeRunFailure(handoff) from error
