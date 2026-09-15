"""D1 failure custody distinguishes model progress from durable journal progress."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

import d1_resume_fixtures as resume_fixtures
import d1_trainer_fixtures as trainer_fixtures
import entropic_substrate.d1_segment_engine as segment_engine
from entropic_substrate.d1_dual_failure import (
    D1PrimarySecondaryFailure,
    D1SecondaryFailureStage,
)
from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.d1_observability import D1JournalObserver
from entropic_substrate.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.d1_resume_trainer import D1ResumeRunFailure, run_d1_resume
from entropic_substrate.d1_trainer_handoff import D1FreshRunFailure
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.evidence.d1_journal import AuthoritativeRunJournal, read_journal
from entropic_substrate.evidence.d1_journal_values import D1JournalEvent, JournalEventKind
from entropic_substrate.experiments.d1_primitives import D1Arm
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem

pytestmark = pytest.mark.unit


class _Sentinel(RuntimeError):
    pass


def _capture_committed_model_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    real_step = segment_engine.apply_d1_training_step
    observed: list[int] = []

    def record_step(**kwargs: object):
        result = real_step(**kwargs)  # type: ignore[arg-type]
        system = cast(D1PredictiveSystem, kwargs["system"])
        cursor = system.target_policy_state().get("update_cursor")
        assert type(cursor) is int
        observed.append(cursor)
        return result

    monkeypatch.setattr(segment_engine, "apply_d1_training_step", record_step)
    return observed


def _capture_interruptions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[segment_engine.D1TrainingInterrupted]:
    real_run = segment_engine.run_d1_training_segment
    observed: list[segment_engine.D1TrainingInterrupted] = []

    def record_run(**kwargs: object):
        try:
            return real_run(**kwargs)  # type: ignore[arg-type]
        except segment_engine.D1TrainingInterrupted as interruption:
            observed.append(interruption)
            raise

    monkeypatch.setattr(segment_engine, "run_d1_training_segment", record_run)
    return observed


def _execute(
    tmp_path: Path,
    case: resume_fixtures.InterruptedD1Case | None,
    observer: D1JournalObserver | None = None,
) -> Path:
    if case is None:
        plan = trainer_fixtures.micro_plan(D1Arm.BOTH)
        journal, checkpoint = trainer_fixtures.output_paths(tmp_path, D1Arm.BOTH)
        run_d1_fresh(
            plan=plan,
            start=trainer_fixtures.run_start(plan),
            origin=D1TrainerWallOrigin(trainer_fixtures.STARTED_AT, trainer_fixtures.ORIGIN_NS),
            locations=D1FreshRunLocations(journal, checkpoint),
            clock=trainer_fixtures.DeterministicTrainerClock(),
        )
        return journal
    run_d1_resume(
        plan=case.plan,
        start=resume_fixtures.resume_start(case.plan, case.checkpoint_sha256),
        origin=resume_fixtures.origin(),
        locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
        observer=observer or resume_fixtures.RecordingObserver(),
        clock=trainer_fixtures.DeterministicTrainerClock(),
    )
    return case.journal_path


@pytest.mark.parametrize(
    ("mode", "expected_model_step", "expected_durable_step"),
    [("fresh", 1, 0), ("resume", 2, 1)],
)
def test_preappend_train_failure_preserves_model_and_durable_cursors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_model_step: int,
    expected_durable_step: int,
) -> None:
    case = (
        resume_fixtures.captured_step_one_case(tmp_path, monkeypatch) if mode == "resume" else None
    )
    model_steps = _capture_committed_model_steps(monkeypatch)
    interruptions = _capture_interruptions(monkeypatch)
    primary = _Sentinel("TRAIN append sentinel")
    real_append = AuthoritativeRunJournal.append
    failed = False

    def fail_before_train_append(
        journal: AuthoritativeRunJournal,
        event: D1JournalEvent,
    ):
        nonlocal failed
        if event.event_kind is JournalEventKind.TRAIN and not failed:
            failed = True
            raise primary
        return real_append(journal, event)

    monkeypatch.setattr(AuthoritativeRunJournal, "append", fail_before_train_append)
    journal_path = case.journal_path if case is not None else tmp_path / "both.jsonl"
    if case is None:
        with pytest.raises(D1FreshRunFailure) as caught:
            _execute(tmp_path, case)
        expected_checkpoint = None
    else:
        with pytest.raises(D1ResumeRunFailure) as caught:
            _execute(tmp_path, case)
        expected_checkpoint = case.checkpoint_sha256
    failure = caught.value

    interruption = interruptions[-1]
    termination = failure.handoff.termination
    assert model_steps[-1] == expected_model_step
    assert interruption.model_successful_step == expected_model_step
    assert interruption.durable_successful_step == expected_durable_step
    assert interruption.latest_checkpoint_sha256 == expected_checkpoint
    assert termination.successful_step == expected_durable_step
    assert termination.model_successful_step == expected_model_step
    assert termination.journal_state.successful_step == expected_durable_step
    assert termination.resource_ledger_state.last_training_step == expected_durable_step
    assert failure.handoff.error_type == "_Sentinel"
    assert failure.handoff.error_message == "TRAIN append sentinel"
    assert failure.handoff.rollback_claimed is False
    assert failure.__cause__ is primary
    assert read_journal(journal_path)[-1].event_kind is JournalEventKind.TRAINER_ABORTED


class _FailAfterDurableTrain(D1JournalObserver):
    def __init__(self, primary: Exception) -> None:
        self.primary = primary
        self.failed = False

    def observe(self, event: D1JournalEvent) -> None:
        if event.event_kind is JournalEventKind.TRAIN and not self.failed:
            self.failed = True
            raise self.primary


def test_postcommit_resume_observer_failure_uses_durable_train_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = resume_fixtures.captured_step_one_case(tmp_path, monkeypatch)
    model_steps = _capture_committed_model_steps(monkeypatch)
    interruptions = _capture_interruptions(monkeypatch)
    primary = _Sentinel("post-commit observer sentinel")

    with pytest.raises(D1ResumeRunFailure) as caught:
        _execute(tmp_path, case, _FailAfterDurableTrain(primary))

    interruption = interruptions[-1]
    termination = caught.value.handoff.termination
    assert model_steps[-1] == 2
    assert interruption.model_successful_step == 2
    assert interruption.durable_successful_step == 2
    assert interruption.latest_checkpoint_sha256 == case.checkpoint_sha256
    assert termination.successful_step == 2
    assert caught.value.handoff.error_message == "post-commit observer sentinel"
    assert caught.value.__cause__ is primary
    assert [event.event_kind for event in read_journal(case.journal_path)[-2:]] == [
        JournalEventKind.TRAIN,
        JournalEventKind.TRAINER_ABORTED,
    ]


@pytest.mark.parametrize("mode", ["fresh", "resume"])
def test_abort_append_failure_preserves_primary_without_false_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    case = (
        resume_fixtures.captured_step_one_case(tmp_path, monkeypatch) if mode == "resume" else None
    )
    primary = _Sentinel("primary trainer sentinel")
    secondary = _Sentinel("abort append sentinel")

    def fail_step(**_kwargs: object) -> object:
        raise primary

    real_append = AuthoritativeRunJournal.append

    def fail_abort_append(
        journal: AuthoritativeRunJournal,
        event: D1JournalEvent,
    ):
        if event.event_kind is JournalEventKind.TRAINER_ABORTED:
            raise secondary
        return real_append(journal, event)

    monkeypatch.setattr(segment_engine, "apply_d1_training_step", fail_step)
    monkeypatch.setattr(AuthoritativeRunJournal, "append", fail_abort_append)
    journal_path = case.journal_path if case is not None else tmp_path / "both.jsonl"
    with pytest.raises(D1PrimarySecondaryFailure) as caught:
        _execute(tmp_path, case)
    assert caught.value.primary is primary
    assert caught.value.secondary is secondary
    assert caught.value.secondary_evidence.stage is D1SecondaryFailureStage.TRAINER_ABORT_APPEND
    assert caught.value.__cause__ is primary
    assert all(
        event.event_kind is not JournalEventKind.TRAINER_ABORTED
        for event in read_journal(journal_path)
    )
