"""Real checkpoint-boundary and hard-crash fixtures for D1 resume tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

import entropic_substrate.d1_fresh_trainer as fresh_module
from d1_trainer_fixtures import (
    ORIGIN_NS,
    STARTED_AT,
    DeterministicTrainerClock,
    micro_plan,
    run_start,
)
from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.d1_resource_activation import D1ResourceActivationFactory
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.evidence.d1_journal import AuthoritativeRunJournal, read_journal
from entropic_substrate.evidence.d1_journal_values import D1JournalEvent
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    SegmentStartMode,
    execution_segment_id,
)
from entropic_substrate.experiments.d1_plan import SmokeD1RunPlan
from entropic_substrate.experiments.d1_primitives import D1Arm
from entropic_substrate.predictive_state.checkpoint_repository import save_atomic


@dataclass(frozen=True, slots=True)
class InterruptedD1Case:
    plan: SmokeD1RunPlan
    journal_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    uninterrupted_terminal: dict[str, object] | None


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[D1JournalEvent] = []

    def observe(self, event: D1JournalEvent) -> None:
        self.events.append(event)


def origin() -> D1TrainerWallOrigin:
    return D1TrainerWallOrigin(started_at=STARTED_AT, monotonic_ns=ORIGIN_NS)


def resume_start(
    plan: SmokeD1RunPlan,
    digest: str,
    *,
    segment: int = 1,
) -> RunStartProvenance:
    start = run_start(plan)
    return replace(
        start,
        execution_segment_index=segment,
        execution_segment_id=execution_segment_id(start.logical_run_id, segment),
        segment_start_mode=SegmentStartMode.RESUME,
        resume_checkpoint_sha256=digest,
    )


def captured_step_one_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_activation_factory: D1ResourceActivationFactory | None = None,
) -> InterruptedD1Case:
    plan = micro_plan(D1Arm.BOTH)
    reference_journal = tmp_path / "reference.jsonl"
    reference_checkpoint = tmp_path / "reference.pt"
    resume_journal = tmp_path / "resume.jsonl"
    resume_checkpoint = tmp_path / "resume.pt"
    real_append = AuthoritativeRunJournal.append_checkpoint

    def capture_checkpoint(target: str | Path, state: dict[str, object]):
        receipt = save_atomic(target, state)
        if state["successful_step"] == 1:
            resume_checkpoint.write_bytes(receipt.path.read_bytes())
        return receipt

    def capture_journal(
        journal: AuthoritativeRunJournal,
        checkpoint_sha256: str,
    ):
        state = real_append(journal, checkpoint_sha256)
        if state.successful_step == 1:
            resume_journal.write_bytes(reference_journal.read_bytes())
        return state

    monkeypatch.setattr(fresh_module, "save_atomic", capture_checkpoint)
    monkeypatch.setattr(AuthoritativeRunJournal, "append_checkpoint", capture_journal)
    start = run_start(plan)
    if resource_activation_factory is not None:
        start = replace(start, resource_activation=resource_activation_factory.registration)
    completion = run_d1_fresh(
        plan=plan,
        start=start,
        origin=origin(),
        locations=D1FreshRunLocations(reference_journal, reference_checkpoint),
        clock=DeterministicTrainerClock(),
        resource_activation_factory=resource_activation_factory,
    )
    terminal = torch.load(
        completion.final_checkpoint.path,
        map_location="cpu",
        weights_only=True,
    )
    digest = hashlib.sha256(resume_checkpoint.read_bytes()).hexdigest()
    assert read_journal(resume_journal)[-1].payload["checkpoint_sha256"] == digest
    return InterruptedD1Case(plan, resume_journal, resume_checkpoint, digest, terminal)
