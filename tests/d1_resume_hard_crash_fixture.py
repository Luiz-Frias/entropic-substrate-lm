"""Hard process-loss fixture that leaves real post-checkpoint journal work."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import entropic_substrate.d1_segment_engine as segment_engine
from d1_resume_fixtures import InterruptedD1Case, origin
from d1_trainer_fixtures import DeterministicTrainerClock, micro_plan, run_start
from entropic_substrate.d1_calibration import D1BaselineCalibrationAttestation
from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.d1_trainer_runtime import D1FreshComponents
from entropic_substrate.evidence.d1_journal import read_journal
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.experiments.d1_primitives import D1Arm
from entropic_substrate.predictive_state.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
)


def hard_crash_tail_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> InterruptedD1Case:
    plan = micro_plan(D1Arm.BOTH)
    journal_path = tmp_path / "crashed.jsonl"
    checkpoint_path = tmp_path / "crashed.pt"
    real_checkpoint = segment_engine.save_d1_segment_checkpoint

    def crash_before_terminal_checkpoint(
        *,
        components: D1FreshComponents,
        journal: segment_engine.D1SegmentJournal,
        checkpoint_path: Path,
        baseline: D1BaselineCalibrationAttestation,
        successful_step: int,
        best_nats_per_byte: float,
    ) -> D1CheckpointArtifactReceipt:
        if successful_step == 2:
            raise KeyboardInterrupt("simulated hard loss after terminal eval")
        return real_checkpoint(
            components=components,
            journal=journal,
            checkpoint_path=checkpoint_path,
            baseline=baseline,
            successful_step=successful_step,
            best_nats_per_byte=best_nats_per_byte,
        )

    with monkeypatch.context() as crash_patch:
        crash_patch.setattr(
            segment_engine,
            "save_d1_segment_checkpoint",
            crash_before_terminal_checkpoint,
        )
        with pytest.raises(KeyboardInterrupt, match="hard loss"):
            run_d1_fresh(
                plan=plan,
                start=run_start(plan),
                origin=origin(),
                locations=D1FreshRunLocations(journal_path, checkpoint_path),
                clock=DeterministicTrainerClock(),
            )
    events = read_journal(journal_path)
    assert [event.event_kind for event in events[-3:]] == [
        JournalEventKind.CHECKPOINT,
        JournalEventKind.TRAIN,
        JournalEventKind.EVAL,
    ]
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return InterruptedD1Case(plan, journal_path, checkpoint_path, digest, None)
