"""Fresh-only D1 trainer application service with authoritative evidence ordering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import entropic_substrate.ladder.run.trainer.d1_segment_engine as segment_engine
from entropic_substrate.kernel.evidence_class.resource_provenance import RunStartProvenance
from entropic_substrate.kernel.evidence_class.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.ladder.capture.records.d1_journal import AuthoritativeRunJournal
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    FrozenJson,
    JournalCheckpointState,
    JournalEventKind,
)
from entropic_substrate.ladder.register.d1_plan import D1RunPlan
from entropic_substrate.ladder.run.resource.d1_resource_activation import (
    D1ResourceActivationFactory,
    UnavailableResourceActivationFactory,
)
from entropic_substrate.ladder.run.trainer.d1_calibration import D1BaselineCalibrator
from entropic_substrate.ladder.run.trainer.d1_dual_failure import (
    D1PrimarySecondaryFailure,
    D1SecondaryFailureEvidence,
    D1SecondaryFailureStage,
)
from entropic_substrate.ladder.run.trainer.d1_terminal_failure import (
    terminal_numerical_defect_for_failure,
)
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import (
    D1TrainerClock,
    D1TrainerWallOrigin,
    SystemD1TrainerClock,
    build_d1_fresh_components,
    create_d1_resource_ledger,
)
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import (
    D1FreshRunCompletion,
    D1FreshRunFailure,
    D1TrainerFailureHandoff,
    capture_d1_trainer_termination,
)


@dataclass(frozen=True, slots=True)
class D1FreshRunLocations:
    journal_path: Path
    checkpoint_path: Path

    def __post_init__(self) -> None:
        for label, path in (
            ("journal_path", self.journal_path),
            ("checkpoint_path", self.checkpoint_path),
        ):
            if not isinstance(cast(object, path), Path):
                raise TypeError(f"D1 {label} must be a Path")
            if not path.parent.is_dir() or path.parent.is_symlink():
                raise ValueError(f"D1 {label} parent must be a real directory")
        if self.journal_path == self.checkpoint_path:
            raise ValueError("D1 journal and checkpoint paths must differ")
        if self.checkpoint_path.exists():
            raise FileExistsError(
                f"D1 fresh checkpoint target already exists: {self.checkpoint_path}"
            )


class D1FreshJournal(Protocol):
    @property
    def state(self) -> JournalCheckpointState: ...
    @property
    def latest_checkpoint_sha256(self) -> str | None: ...
    def append(self, event: D1JournalEvent) -> JournalCheckpointState: ...
    def append_checkpoint(self, checkpoint_sha256: str) -> JournalCheckpointState: ...
    def __enter__(self) -> D1FreshJournal: ...
    def __exit__(self, *_: object) -> None: ...


type D1FreshJournalFactory = Callable[[Path, D1JournalEvent], D1FreshJournal]


def _event(
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


def _next_event(
    journal: D1FreshJournal,
    plan: D1RunPlan,
    kind: JournalEventKind,
    *,
    successful_step: int,
    payload: Mapping[str, object],
) -> D1JournalEvent:
    return _event(
        plan,
        kind,
        sequence=journal.state.last_event_sequence + 1,
        successful_step=successful_step,
        payload=payload,
    )


def run_d1_fresh(
    *,
    plan: D1RunPlan,
    start: RunStartProvenance,
    origin: D1TrainerWallOrigin,
    locations: D1FreshRunLocations,
    clock: D1TrainerClock | None = None,
    resource_activation_factory: D1ResourceActivationFactory | None = None,
    journal_factory: D1FreshJournalFactory = AuthoritativeRunJournal.fresh,
    checkpoint_sync: segment_engine.D1CheckpointSyncPort | None = None,
) -> D1FreshRunCompletion:
    """Execute one fresh D1 run and stop at the operator-finalization boundary."""
    if type(locations) is not D1FreshRunLocations:
        raise TypeError("D1 fresh trainer requires D1FreshRunLocations")
    active_clock = clock or SystemD1TrainerClock()
    if not isinstance(cast(object, active_clock), D1TrainerClock):
        raise TypeError("D1 fresh trainer clock must implement D1TrainerClock")
    if not callable(journal_factory):
        raise TypeError("D1 fresh trainer journal_factory must be callable")
    activation_factory = resource_activation_factory or UnavailableResourceActivationFactory()
    if not isinstance(cast(object, activation_factory), D1ResourceActivationFactory):
        raise TypeError("D1 resource activation factory is invalid")
    if activation_factory.registration != start.resource_activation:
        raise ValueError("D1 resource activation differs from run-start provenance")
    activation = activation_factory.fresh(start)
    ledger = create_d1_resource_ledger(
        plan=plan,
        start=start,
        origin=origin,
        meter_strategy=activation.meter,
    )
    start_event = _event(
        plan,
        JournalEventKind.RUN_START,
        sequence=0,
        successful_step=0,
        payload={
            "plan": plan.plan_manifest,
            "plan_digest": plan.plan_digest,
            "run_start_provenance": run_start_to_mapping(start),
            "trainer_wall_origin": origin.manifest,
        },
    )
    with journal_factory(locations.journal_path, start_event) as journal, activation:
        successful_step = 0
        model_successful_step = 0
        latest_checkpoint_sha256: str | None = None
        try:
            components = build_d1_fresh_components(
                plan=plan,
                start=start,
                origin=origin,
                resource_ledger=ledger,
            )
            baseline = D1BaselineCalibrator().calibrate(
                system=components.system,
                split=components.corpus.train,
                successful_step=0,
            )
            ledger.record_baseline(target_positions=baseline.source_span_positions)
            journal.append(
                _next_event(
                    journal,
                    plan,
                    JournalEventKind.BASELINE_CALIBRATED,
                    successful_step=0,
                    payload={
                        "attestation": baseline.manifest,
                        "resource_ledger": ledger.current_state().manifest,
                    },
                )
            )
            initial_eval, _initial_work = segment_engine.record_d1_segment_evaluation(
                components=components,
                journal=journal,
                clock=active_clock,
                meter=activation.meter,
                successful_step=0,
            )
            training = segment_engine.run_d1_training_segment(
                components=components,
                journal=journal,
                clock=active_clock,
                meter=activation.meter,
                baseline=baseline,
                journal_path=locations.journal_path,
                checkpoint_path=locations.checkpoint_path,
                checkpoint_sync=checkpoint_sync,
                successful_step=successful_step,
                best_nats_per_byte=initial_eval.nats_per_byte,
            )
            successful_step = training.successful_step
            model_successful_step = training.successful_step
            latest_checkpoint_sha256 = training.final_checkpoint.sha256
            termination = capture_d1_trainer_termination(
                start=start,
                trainer_ended_at=active_clock.utc_now(),
                successful_step=successful_step,
                model_successful_step=model_successful_step,
                journal_state=journal.state,
                resource_ledger_state=ledger.current_state(),
            )
            return D1FreshRunCompletion(
                termination=termination,
                final_checkpoint=training.final_checkpoint,
                final_evaluation=training.final_evaluation,
                final_resource_work=training.final_resource_work,
                data_order_sha256=training.data_order_sha256,
                data_order_batches_attested=training.data_order_batches_attested,
            )
        except Exception as caught:
            if isinstance(caught, segment_engine.D1TrainingInterrupted):
                successful_step = caught.durable_successful_step
                model_successful_step = caught.model_successful_step
                latest_checkpoint_sha256 = caught.latest_checkpoint_sha256
                error = caught.primary
            else:
                error = caught
            trainer_ended_at = active_clock.utc_now()
            secondary_failure: D1SecondaryFailureEvidence | None = None
            failure_cause: BaseException = error
            try:
                ledger.record_meter_observation(monotonic_ns=active_clock.monotonic_ns())
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
            resource_ledger_state = ledger.current_state()
            error_message = str(error) or repr(error)
            failure_payload: dict[str, object] = {
                "error_message": error_message,
                "error_type": type(error).__name__,
                "execution_segment_id": start.execution_segment_id,
                "execution_segment_index": start.execution_segment_index,
                "journal_cursor": journal_cursor.manifest,
                "logical_run_id": start.logical_run_id,
                "model_successful_step": model_successful_step,
                "operator_finalization_required": True,
                "resource_ledger_state": resource_ledger_state.manifest,
                "rollback_claimed": False,
                "secondary_failure": (
                    None if secondary_failure is None else secondary_failure.manifest
                ),
                "trainer_ended_at": trainer_ended_at.isoformat().replace("+00:00", "Z"),
            }
            numerical_defect = terminal_numerical_defect_for_failure(
                error,
                plan=plan,
                start=start,
                resource_state=resource_ledger_state,
                successful_step=successful_step,
                checkpoint_sha256=latest_checkpoint_sha256,
            )
            if numerical_defect is not None:
                failure_payload["numerical_defect"] = numerical_defect.manifest
            try:
                journal.append(
                    _next_event(
                        journal,
                        plan,
                        JournalEventKind.TRAINER_ABORTED,
                        successful_step=successful_step,
                        payload=failure_payload,
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
                successful_step=successful_step,
                model_successful_step=model_successful_step,
                journal_state=journal.state,
                resource_ledger_state=resource_ledger_state,
            )
            handoff = D1TrainerFailureHandoff.from_error(
                termination,
                error,
                secondary_failure=secondary_failure,
            )
            raise D1FreshRunFailure(handoff) from failure_cause
