"""Shared evaluation and checkpoint primitives for every D1 training segment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from entropic_substrate.__legacy.evidence.d1_journal_values import (
    D1JournalEvent,
    FrozenJson,
    JournalCheckpointState,
    JournalEventKind,
)
from entropic_substrate.__legacy.evidence.resource_meter_models import ResourcePhase
from entropic_substrate.__legacy.evidence.resource_work import ResourceWorkSnapshot
from entropic_substrate.__legacy.evidence.resource_work_codec import resource_snapshot_to_mapping
from entropic_substrate.ladder.run.resource.d1_resource_activation import D1ActiveResourceMeter
from entropic_substrate.ladder.run.trainer.checkpoint_envelope import build_d1_checkpoint
from entropic_substrate.ladder.run.trainer.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
    save_atomic,
)
from entropic_substrate.ladder.run.trainer.d1_calibration import D1BaselineCalibrationAttestation
from entropic_substrate.ladder.run.trainer.d1_evaluation import D1EvalResult, D1Evaluator
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import (
    D1FreshComponents,
    D1TrainerClock,
)
from entropic_substrate.ladder.run.trainer.d1_training_step import apply_d1_training_step

__all__ = [
    "D1CheckpointSyncPort",
    "D1SegmentJournal",
    "D1TrainingInterrupted",
    "D1TrainingResult",
    "record_d1_segment_evaluation",
    "run_d1_training_segment",
    "save_d1_segment_checkpoint",
]


class D1SegmentJournal(Protocol):
    """Journal capabilities needed by model-segment execution."""

    @property
    def state(self) -> JournalCheckpointState: ...
    @property
    def latest_checkpoint_sha256(self) -> str | None: ...
    def append(self, event: D1JournalEvent) -> JournalCheckpointState: ...
    def append_checkpoint(self, checkpoint_sha256: str) -> JournalCheckpointState: ...


class D1CheckpointSyncPort(Protocol):
    """Bound remote sync invoked after every locally durable checkpoint event."""

    def sync(
        self,
        *,
        successful_step: int,
        run_root: Path,
        checkpoint: D1CheckpointArtifactReceipt,
        journal_path: Path,
    ) -> None: ...


def _event(
    components: D1FreshComponents,
    journal: D1SegmentJournal,
    kind: JournalEventKind,
    *,
    successful_step: int,
    payload: Mapping[str, object],
) -> D1JournalEvent:
    plan = components.plan
    return D1JournalEvent(
        event_kind=kind,
        event_seq=journal.state.last_event_sequence + 1,
        logical_run_id=plan.logical_run_id,
        execution_digest=plan.run.execution_digest,
        aggregation_digest=plan.run.aggregation_digest,
        construction_digest=plan.run.construction_digest,
        family_digest=plan.profile.family.family_digest,
        run_class=plan.run.run_class,
        successful_step=successful_step,
        payload=cast(Mapping[str, FrozenJson], payload),
    )


def record_d1_segment_evaluation(
    *,
    components: D1FreshComponents,
    journal: D1SegmentJournal,
    clock: D1TrainerClock,
    meter: D1ActiveResourceMeter,
    successful_step: int,
) -> tuple[D1EvalResult, ResourceWorkSnapshot]:
    """Evaluate one segment boundary and journal its measured resource work."""
    with meter.measure(ResourcePhase.EVAL):
        result = D1Evaluator().evaluate(
            system=components.system,
            sampler=components.eval_sampler,
        )
    batching = components.plan.profile.draft.execution_protocol.batching
    target_positions = (
        result.batch_count * batching.batch_size * (batching.sequence_length + batching.horizon)
    )
    meter.record_positions(
        ResourcePhase.EVAL,
        online_positions=result.online_positions,
        target_positions=target_positions,
    )
    work = components.resource_ledger.record_eval(
        online_positions=result.online_positions,
        target_positions=target_positions,
        training_step=successful_step,
        observed_at=clock.utc_now(),
        monotonic_ns=clock.monotonic_ns(),
    )
    journal.append(
        _event(
            components,
            journal,
            JournalEventKind.EVAL,
            successful_step=successful_step,
            payload={
                "eval_index": work.eval_index,
                "quality": result.telemetry_manifest,
                "resource_work": resource_snapshot_to_mapping(work),
            },
        )
    )
    return result, work


def save_d1_segment_checkpoint(
    *,
    components: D1FreshComponents,
    journal: D1SegmentJournal,
    checkpoint_path: Path,
    baseline: D1BaselineCalibrationAttestation,
    successful_step: int,
    best_nats_per_byte: float,
) -> D1CheckpointArtifactReceipt:
    """Build, atomically persist, and journal one resumable segment checkpoint."""
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
    receipt = save_atomic(checkpoint_path, checkpoint)
    journal.append_checkpoint(receipt.sha256)
    return receipt


def _require_sha256(value: str, *, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class D1TrainingResult:
    """Terminal model and data-order evidence from one D1 training segment."""

    successful_step: int
    final_evaluation: D1EvalResult
    final_resource_work: ResourceWorkSnapshot
    final_checkpoint: D1CheckpointArtifactReceipt
    data_order_sha256: str
    data_order_batches_attested: int

    def __post_init__(self) -> None:
        if type(self.successful_step) is not int or self.successful_step < 1:
            raise ValueError("D1 training result successful_step must be positive")
        if type(self.final_evaluation) is not D1EvalResult:
            raise TypeError("D1 training result requires a final evaluation")
        if type(self.final_resource_work) is not ResourceWorkSnapshot:
            raise TypeError("D1 training result requires final resource work")
        if type(self.final_checkpoint) is not D1CheckpointArtifactReceipt:
            raise TypeError("D1 training result requires a final checkpoint")
        if self.final_resource_work.training_step != self.successful_step:
            raise ValueError("D1 training result resource cursor differs")
        _require_sha256(self.data_order_sha256, label="data_order_sha256")
        if (
            type(self.data_order_batches_attested) is not int
            or self.data_order_batches_attested < 1
        ):
            raise ValueError("D1 training result data-order batch count must be positive")


class D1TrainingInterrupted(RuntimeError):
    """Ordinary segment failure plus the latest authoritative progress cursor."""

    def __init__(
        self,
        primary: Exception,
        *,
        model_successful_step: int,
        durable_successful_step: int,
        latest_checkpoint_sha256: str | None,
    ) -> None:
        if not isinstance(cast(object, primary), Exception):
            raise TypeError("D1 training interruption requires an Exception")
        for label, value in (
            ("model_successful_step", model_successful_step),
            ("durable_successful_step", durable_successful_step),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"D1 training interruption {label} must be nonnegative")
        if model_successful_step not in {
            durable_successful_step,
            durable_successful_step + 1,
        }:
            raise ValueError("D1 training interruption model/durable cursor gap is invalid")
        if latest_checkpoint_sha256 is not None:
            _require_sha256(latest_checkpoint_sha256, label="latest_checkpoint_sha256")
        self.primary = primary
        self.model_successful_step = model_successful_step
        self.durable_successful_step = durable_successful_step
        self.latest_checkpoint_sha256 = latest_checkpoint_sha256
        super().__init__(
            f"D1 training segment interrupted at model step {model_successful_step}, "
            f"durable step {durable_successful_step}: "
            f"{type(primary).__name__}: {str(primary) or repr(primary)}"
        )


def _data_order(components: D1FreshComponents) -> tuple[str, int]:
    observed = components.train_sampler.data_order_digest()
    if observed is None:
        raise RuntimeError("D1 training segment has no data-order attestation")
    return observed


def run_d1_training_segment(
    *,
    components: D1FreshComponents,
    journal: D1SegmentJournal,
    clock: D1TrainerClock,
    meter: D1ActiveResourceMeter,
    baseline: D1BaselineCalibrationAttestation,
    journal_path: Path,
    checkpoint_path: Path,
    checkpoint_sync: D1CheckpointSyncPort | None,
    successful_step: int,
    best_nats_per_byte: float,
) -> D1TrainingResult:
    """Run the shared train/eval/checkpoint loop after segment-specific admission."""
    if type(components) is not D1FreshComponents:
        raise TypeError("D1 training segment requires materialized components")
    if type(baseline) is not D1BaselineCalibrationAttestation:
        raise TypeError("D1 training segment requires baseline calibration")
    protocol = components.plan.profile.draft.execution_protocol
    if (
        type(successful_step) is not int
        or successful_step < 0
        or successful_step >= protocol.optimization.max_steps
    ):
        raise ValueError("D1 training segment requires remaining optimization budget")
    if type(best_nats_per_byte) is not float or not math.isfinite(best_nats_per_byte):
        raise ValueError("D1 training segment best nats/byte must be finite")
    if checkpoint_sync is not None and not callable(getattr(checkpoint_sync, "sync", None)):
        raise TypeError("D1 training segment checkpoint_sync must implement sync")
    step = successful_step
    final_evaluation: D1EvalResult | None = None
    final_work: ResourceWorkSnapshot | None = None
    final_checkpoint: D1CheckpointArtifactReceipt | None = None
    try:
        while step < protocol.optimization.max_steps:
            batch = components.train_sampler.predictive_batch(protocol.batching.horizon)
            online_positions = batch.online_observations.numel()
            target_positions = batch.target_encoder_context.numel()
            try:
                with meter.measure(ResourcePhase.TRAIN):
                    train_result = apply_d1_training_step(
                        system=components.system,
                        optimizer_bundle=components.optimizer,
                        batch=batch,
                        spec=protocol.optimization,
                        successful_step=step,
                    )
            except BaseException:
                components.resource_ledger.record_train_attempt(
                    online_positions=online_positions,
                    target_positions=target_positions,
                    quality_source_byte_exposures=online_positions,
                    committed=False,
                )
                raise
            meter.record_positions(
                ResourcePhase.TRAIN,
                online_positions=online_positions,
                target_positions=target_positions,
            )
            step = train_result.successful_step
            components.resource_ledger.record_train_attempt(
                online_positions=online_positions,
                target_positions=target_positions,
                quality_source_byte_exposures=online_positions,
                committed=True,
            )
            order_sha256, order_batches = _data_order(components)
            journal.append(
                _event(
                    components,
                    journal,
                    JournalEventKind.TRAIN,
                    successful_step=step,
                    payload={
                        "data_order": {
                            "batches_attested": order_batches,
                            "sha256": order_sha256,
                        },
                        "scheduled_log_boundary": step % protocol.evaluation.log_interval == 0,
                        "train": train_result.manifest,
                    },
                )
            )
            terminal = step == protocol.optimization.max_steps
            if step % protocol.evaluation.eval_interval == 0 or terminal:
                final_evaluation, final_work = record_d1_segment_evaluation(
                    components=components,
                    journal=journal,
                    clock=clock,
                    meter=meter,
                    successful_step=step,
                )
                best_nats_per_byte = min(
                    best_nats_per_byte,
                    final_evaluation.nats_per_byte,
                )
                scheduled_save = protocol.evaluation.save_interval > 0 and (
                    step % protocol.evaluation.save_interval == 0
                )
                if scheduled_save or terminal:
                    final_checkpoint = save_d1_segment_checkpoint(
                        components=components,
                        journal=journal,
                        checkpoint_path=checkpoint_path,
                        baseline=baseline,
                        successful_step=step,
                        best_nats_per_byte=best_nats_per_byte,
                    )
                    if checkpoint_sync is not None:
                        checkpoint_sync.sync(
                            successful_step=step,
                            run_root=checkpoint_path.parent,
                            checkpoint=final_checkpoint,
                            journal_path=journal_path,
                        )
        if final_evaluation is None or final_work is None or final_checkpoint is None:
            raise RuntimeError("D1 terminal training boundary lacks evaluation/checkpoint")
        order_sha256, order_batches = _data_order(components)
        return D1TrainingResult(
            successful_step=step,
            final_evaluation=final_evaluation,
            final_resource_work=final_work,
            final_checkpoint=final_checkpoint,
            data_order_sha256=order_sha256,
            data_order_batches_attested=order_batches,
        )
    except Exception as primary:
        raise D1TrainingInterrupted(
            primary,
            model_successful_step=step,
            durable_successful_step=journal.state.successful_step,
            latest_checkpoint_sha256=journal.latest_checkpoint_sha256,
        ) from primary
