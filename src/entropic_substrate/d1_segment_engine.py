"""Shared evaluation and checkpoint primitives for every D1 training segment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

from entropic_substrate.d1_calibration import D1BaselineCalibrationAttestation
from entropic_substrate.d1_evaluation import D1EvalResult, D1Evaluator
from entropic_substrate.d1_resource_activation import D1ActiveResourceMeter
from entropic_substrate.d1_trainer_runtime import D1FreshComponents, D1TrainerClock
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalEvent,
    FrozenJson,
    JournalCheckpointState,
    JournalEventKind,
)
from entropic_substrate.evidence.resource_meter_models import ResourcePhase
from entropic_substrate.evidence.resource_work import ResourceWorkSnapshot
from entropic_substrate.evidence.resource_work_codec import resource_snapshot_to_mapping
from entropic_substrate.predictive_state.checkpoint_envelope import build_d1_checkpoint
from entropic_substrate.predictive_state.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
    save_atomic,
)

__all__ = [
    "D1SegmentJournal",
    "record_d1_segment_evaluation",
    "save_d1_segment_checkpoint",
]


class D1SegmentJournal(Protocol):
    """Journal capabilities needed by model-segment execution."""

    @property
    def state(self) -> JournalCheckpointState: ...
    def append(self, event: D1JournalEvent) -> JournalCheckpointState: ...
    def append_checkpoint(self, checkpoint_sha256: str) -> JournalCheckpointState: ...


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
