"""Fresh-only D1 trainer application service with authoritative evidence ordering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import entropic_substrate.ladder.run.trainer.d1_segment_engine as segment_engine
from entropic_substrate.kernel.evidence_class.resource_provenance import RunStartProvenance
from entropic_substrate.kernel.evidence_class.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.ladder.capture.records.d1_journal import AuthoritativeRunJournal
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    JournalCheckpointState,
    JournalEventKind,
)
from entropic_substrate.ladder.register.d1_plan import D1RunPlan
from entropic_substrate.ladder.run.resource.d1_resource_activation import (
    D1ResourceActivationFactory,
    UnavailableResourceActivationFactory,
)
from entropic_substrate.ladder.run.trainer.d1_calibration import D1BaselineCalibrator
from entropic_substrate.ladder.run.trainer.d1_terminal_failure import (
    _trainer_failure,
    next_trainer_event,
    trainer_event,
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
    start_event = trainer_event(
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
                next_trainer_event(
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
            handoff, failure_cause = _trainer_failure(
                plan,
                journal,
                ledger,
                start,
                active_clock,
                successful_step,
                model_successful_step,
                error,
                latest_checkpoint_sha256,
            )
            raise D1FreshRunFailure(handoff) from failure_cause
