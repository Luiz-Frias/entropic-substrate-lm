"""Failure and exhausted-budget guard teeth for resumed D1 execution."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

import entropic_substrate.d1_segment_engine as segment_engine
from d1_resume_fixtures import RecordingObserver, captured_step_one_case, origin, resume_start
from d1_trainer_fixtures import DeterministicTrainerClock, micro_plan, run_start
from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.d1_resume_runtime import D1ResumeRunLocations, open_d1_resume
from entropic_substrate.d1_resume_trainer import D1ResumeRunFailure, run_d1_resume
from entropic_substrate.evidence.d1_journal import read_journal
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.experiments.d1_primitives import D1Arm

pytestmark = pytest.mark.unit


def test_failed_attempt_appends_one_trainer_aborted_and_returns_typed_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)

    def fail_step(**_kwargs: object) -> object:
        raise RuntimeError("resume step defect")

    monkeypatch.setattr(segment_engine, "apply_d1_training_step", fail_step)
    with pytest.raises(D1ResumeRunFailure, match="resume step defect") as captured:
        run_d1_resume(
            plan=case.plan,
            start=resume_start(case.plan, case.checkpoint_sha256),
            origin=origin(),
            locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
            observer=RecordingObserver(),
            clock=DeterministicTrainerClock(),
        )

    events = read_journal(case.journal_path)
    assert events[-1].event_kind is JournalEventKind.TRAINER_ABORTED
    assert sum(event.event_kind is JournalEventKind.TRAINER_ABORTED for event in events) == 1
    assert all(
        event.event_kind not in {JournalEventKind.RUN_ABORT, JournalEventKind.RUN_END}
        for event in events
    )
    handoff = captured.value.handoff
    assert type(captured.value.__cause__) is RuntimeError
    assert handoff.trainer_event_kind == "TRAINER_ABORTED"
    assert handoff.rollback_claimed is False
    assert handoff.operator_finalization_required is True
    assert handoff.termination.execution_segment_index == 1
    assert handoff.termination.journal_state.last_event_sequence == events[-1].event_seq
    ledger = handoff.termination.resource_ledger_state
    assert ledger.committed_quality_source_bytes == 4
    assert ledger.online_train_token_positions == 8
    assert ledger.target_train_token_positions == 10
    abort = cast(dict[str, object], events[-1].payload)
    assert abort["resource_ledger_state"] == ledger.manifest
    assert abort["rollback_claimed"] is False


def test_terminal_budget_refuses_before_resume_event(tmp_path: Path) -> None:
    plan = micro_plan(D1Arm.NONE)
    journal_path = tmp_path / "terminal.jsonl"
    checkpoint_path = tmp_path / "terminal.pt"
    completion = run_d1_fresh(
        plan=plan,
        start=run_start(plan),
        origin=origin(),
        locations=D1FreshRunLocations(journal_path, checkpoint_path),
        clock=DeterministicTrainerClock(),
    )
    before = journal_path.read_bytes()

    with pytest.raises(ValueError, match="optimization budget is exhausted"):
        open_d1_resume(
            plan=plan,
            start=resume_start(plan, completion.final_checkpoint.sha256),
            origin=origin(),
            locations=D1ResumeRunLocations(journal_path, checkpoint_path),
            observer=RecordingObserver(),
        )
    assert journal_path.read_bytes() == before
