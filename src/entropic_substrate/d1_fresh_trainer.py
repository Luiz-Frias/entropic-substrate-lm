"""Fresh-only D1 trainer application service with authoritative evidence ordering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from entropic_substrate.d1_calibration import (
    D1BaselineCalibrationAttestation,
    D1BaselineCalibrator,
)
from entropic_substrate.d1_dual_failure import (
    D1PrimarySecondaryFailure,
    D1SecondaryFailureEvidence,
    D1SecondaryFailureStage,
)
from entropic_substrate.d1_evaluation import D1EvalResult, D1Evaluator
from entropic_substrate.d1_resource_activation import (
    D1ActiveResourceMeter,
    D1ResourceActivationFactory,
    UnavailableResourceActivationFactory,
)
from entropic_substrate.d1_terminal_failure import terminal_numerical_defect_for_failure
from entropic_substrate.d1_trainer_handoff import (
    D1FreshRunCompletion,
    D1FreshRunFailure,
    D1TrainerFailureHandoff,
    capture_d1_trainer_termination,
)
from entropic_substrate.d1_trainer_runtime import (
    D1FreshComponents,
    D1TrainerClock,
    D1TrainerWallOrigin,
    SystemD1TrainerClock,
    build_d1_fresh_components,
    create_d1_resource_ledger,
)
from entropic_substrate.d1_training_step import apply_d1_training_step
from entropic_substrate.evidence.d1_journal import AuthoritativeRunJournal
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalEvent,
    FrozenJson,
    JournalCheckpointState,
    JournalEventKind,
)
from entropic_substrate.evidence.resource_meter_models import ResourcePhase
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


def _record_evaluation(
    *,
    components: D1FreshComponents,
    evaluator: D1Evaluator,
    journal: D1FreshJournal,
    clock: D1TrainerClock,
    meter: D1ActiveResourceMeter,
    successful_step: int,
) -> tuple[D1EvalResult, ResourceWorkSnapshot]:
    with meter.measure(ResourcePhase.EVAL):
        result = evaluator.evaluate(system=components.system, sampler=components.eval_sampler)
    protocol = components.plan.profile.draft.execution_protocol
    batching = protocol.batching
    target_positions = (
        result.batch_count * batching.batch_size * (batching.sequence_length + batching.horizon)
    )
    meter.record_positions(
        ResourcePhase.EVAL,
        online_positions=result.online_positions,
        target_positions=target_positions,
    )
    snapshot = components.resource_ledger.record_eval(
        online_positions=result.online_positions,
        target_positions=target_positions,
        training_step=successful_step,
        observed_at=clock.utc_now(),
        monotonic_ns=clock.monotonic_ns(),
    )
    journal.append(
        _next_event(
            journal,
            components.plan,
            JournalEventKind.EVAL,
            successful_step=successful_step,
            payload={
                "eval_index": snapshot.eval_index,
                "quality": result.telemetry_manifest,
                "resource_work": resource_snapshot_to_mapping(snapshot),
            },
        )
    )
    return result, snapshot


def _checkpoint(
    *,
    components: D1FreshComponents,
    journal: D1FreshJournal,
    locations: D1FreshRunLocations,
    baseline: D1BaselineCalibrationAttestation,
    successful_step: int,
    best_nats_per_byte: float,
) -> D1CheckpointArtifactReceipt:
    checkpoint = build_d1_checkpoint(
        system=components.system,
        optimizer=components.optimizer.optimizer,
        sampler=components.train_sampler,
        run=components.plan.run,
        protocol=components.plan.profile.draft.execution_protocol,
        successful_step=successful_step,
        baseline_calibration=baseline,
        resource_ledger=components.resource_ledger.current_state(),
        journal_state=journal.state,
        lifecycle={
            "schema_version": "d1-trainer-lifecycle-v1",
            "state": "running",
            "completion_reason": None,
            "abort_reason": None,
        },
        best_nats_per_byte=best_nats_per_byte,
    )
    receipt = save_atomic(locations.checkpoint_path, checkpoint)
    journal.append_checkpoint(receipt.sha256)
    return receipt


def _data_order(components: D1FreshComponents) -> tuple[str, int]:
    observed = components.train_sampler.data_order_digest()
    if observed is None:
        raise RuntimeError("D1 fresh trainer has no observed training data-order digest")
    return observed


def run_d1_fresh(
    *,
    plan: D1RunPlan,
    start: RunStartProvenance,
    origin: D1TrainerWallOrigin,
    locations: D1FreshRunLocations,
    clock: D1TrainerClock | None = None,
    resource_activation_factory: D1ResourceActivationFactory | None = None,
    journal_factory: D1FreshJournalFactory = AuthoritativeRunJournal.fresh,
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
            evaluator = D1Evaluator()
            final_eval, final_work = _record_evaluation(
                components=components,
                evaluator=evaluator,
                journal=journal,
                clock=active_clock,
                meter=activation.meter,
                successful_step=0,
            )
            best_nats = final_eval.nats_per_byte
            final_receipt: D1CheckpointArtifactReceipt | None = None
            protocol = plan.profile.draft.execution_protocol
            for _ in range(protocol.optimization.max_steps):
                batch = components.train_sampler.predictive_batch(protocol.batching.horizon)
                online_positions = batch.online_observations.numel()
                target_positions = batch.target_encoder_context.numel()
                try:
                    with activation.meter.measure(ResourcePhase.TRAIN):
                        train_result = apply_d1_training_step(
                            system=components.system,
                            optimizer_bundle=components.optimizer,
                            batch=batch,
                            spec=protocol.optimization,
                            successful_step=successful_step,
                        )
                except BaseException:
                    ledger.record_train_attempt(
                        online_positions=online_positions,
                        target_positions=target_positions,
                        quality_source_byte_exposures=online_positions,
                        committed=False,
                    )
                    raise
                activation.meter.record_positions(
                    ResourcePhase.TRAIN,
                    online_positions=online_positions,
                    target_positions=target_positions,
                )
                successful_step = train_result.successful_step
                ledger.record_train_attempt(
                    online_positions=online_positions,
                    target_positions=target_positions,
                    quality_source_byte_exposures=online_positions,
                    committed=True,
                )
                order_sha256, order_batches = _data_order(components)
                journal.append(
                    _next_event(
                        journal,
                        plan,
                        JournalEventKind.TRAIN,
                        successful_step=successful_step,
                        payload={
                            "data_order": {
                                "batches_attested": order_batches,
                                "sha256": order_sha256,
                            },
                            "scheduled_log_boundary": (
                                successful_step % protocol.evaluation.log_interval == 0
                            ),
                            "train": train_result.manifest,
                        },
                    )
                )
                terminal = successful_step == protocol.optimization.max_steps
                if successful_step % protocol.evaluation.eval_interval == 0 or terminal:
                    final_eval, final_work = _record_evaluation(
                        components=components,
                        evaluator=evaluator,
                        journal=journal,
                        clock=active_clock,
                        meter=activation.meter,
                        successful_step=successful_step,
                    )
                    best_nats = min(best_nats, final_eval.nats_per_byte)
                    scheduled_save = (
                        protocol.evaluation.save_interval > 0
                        and successful_step % protocol.evaluation.save_interval == 0
                    )
                    if scheduled_save or terminal:
                        final_receipt = _checkpoint(
                            components=components,
                            journal=journal,
                            locations=locations,
                            baseline=baseline,
                            successful_step=successful_step,
                            best_nats_per_byte=best_nats,
                        )
                        latest_checkpoint_sha256 = final_receipt.sha256
            if final_receipt is None:
                raise RuntimeError("D1 terminal evaluation did not produce a checkpoint")
            order_sha256, order_batches = _data_order(components)
            termination = capture_d1_trainer_termination(
                start=start,
                trainer_ended_at=active_clock.utc_now(),
                successful_step=successful_step,
                journal_state=journal.state,
                resource_ledger_state=ledger.current_state(),
            )
            return D1FreshRunCompletion(
                termination=termination,
                final_checkpoint=final_receipt,
                final_evaluation=final_eval,
                final_resource_work=final_work,
                data_order_sha256=order_sha256,
                data_order_batches_attested=order_batches,
            )
        except Exception as error:
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
                failure_cause = D1PrimarySecondaryFailure(error, meter_error)
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
            journal.append(
                _next_event(
                    journal,
                    plan,
                    JournalEventKind.TRAINER_ABORTED,
                    successful_step=successful_step,
                    payload=failure_payload,
                )
            )
            termination = capture_d1_trainer_termination(
                start=start,
                trainer_ended_at=trainer_ended_at,
                successful_step=successful_step,
                journal_state=journal.state,
                resource_ledger_state=resource_ledger_state,
            )
            handoff = D1TrainerFailureHandoff.from_error(
                termination,
                error,
                secondary_failure=secondary_failure,
            )
            raise D1FreshRunFailure(handoff) from failure_cause
