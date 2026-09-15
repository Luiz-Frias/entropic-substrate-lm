"""Checkpoint authority advances only from typed postcommit journal evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import entropic_substrate.d1_segment_engine as segment_engine
from d1_resume_fixtures import RecordingObserver, captured_step_one_case, origin, resume_start
from d1_trainer_fixtures import DeterministicTrainerClock
from entropic_substrate.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.d1_resume_trainer import D1ResumeRunFailure, run_d1_resume
from entropic_substrate.evidence.d1_journal import AuthoritativeRunJournal, read_journal
from entropic_substrate.evidence.d1_journal_values import D1JournalEvent, JournalEventKind

pytestmark = pytest.mark.unit


class _Sentinel(RuntimeError):
    pass


class _FailCheckpointObserver:
    def __init__(self, primary: Exception) -> None:
        self.primary = primary
        self.failed = False

    def observe(self, event: D1JournalEvent) -> None:
        if event.event_kind is JournalEventKind.CHECKPOINT and not self.failed:
            self.failed = True
            raise self.primary


def _capture_interruptions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[segment_engine.D1TrainingInterrupted]:
    real_run = segment_engine.run_d1_training_segment
    observed: list[segment_engine.D1TrainingInterrupted] = []

    def capture(**kwargs: object):
        try:
            return real_run(**kwargs)  # type: ignore[arg-type]
        except segment_engine.D1TrainingInterrupted as interruption:
            observed.append(interruption)
            raise

    monkeypatch.setattr(segment_engine, "run_d1_training_segment", capture)
    return observed


def test_resume_postcommit_checkpoint_observer_advances_authoritative_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)
    interruptions = _capture_interruptions(monkeypatch)
    primary = _Sentinel("checkpoint observer sentinel")

    with pytest.raises(D1ResumeRunFailure) as caught:
        run_d1_resume(
            plan=case.plan,
            start=resume_start(case.plan, case.checkpoint_sha256),
            origin=origin(),
            locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
            observer=_FailCheckpointObserver(primary),
            clock=DeterministicTrainerClock(),
        )

    events = read_journal(case.journal_path)
    newest = events[-2]
    assert newest.event_kind is JournalEventKind.CHECKPOINT
    newest_sha256 = newest.payload["checkpoint_sha256"]
    assert interruptions[-1].latest_checkpoint_sha256 == newest_sha256
    assert newest_sha256 != case.checkpoint_sha256
    assert newest_sha256 == hashlib.sha256(case.checkpoint_path.read_bytes()).hexdigest()
    assert interruptions[-1].model_successful_step == 2
    assert interruptions[-1].durable_successful_step == 2
    assert caught.value.handoff.termination.successful_step == 2
    assert caught.value.__cause__ is primary


def test_resume_precommit_checkpoint_append_keeps_prior_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)
    interruptions = _capture_interruptions(monkeypatch)
    primary = _Sentinel("checkpoint append sentinel")

    def fail_precommit(
        _journal: AuthoritativeRunJournal,
        _checkpoint_sha256: str,
    ) -> object:
        raise primary

    monkeypatch.setattr(AuthoritativeRunJournal, "append_checkpoint", fail_precommit)
    with pytest.raises(D1ResumeRunFailure) as caught:
        run_d1_resume(
            plan=case.plan,
            start=resume_start(case.plan, case.checkpoint_sha256),
            origin=origin(),
            locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
            observer=RecordingObserver(),
            clock=DeterministicTrainerClock(),
        )

    checkpoints = [
        event
        for event in read_journal(case.journal_path)
        if event.event_kind is JournalEventKind.CHECKPOINT
    ]
    assert checkpoints[-1].payload["checkpoint_sha256"] == case.checkpoint_sha256
    assert interruptions[-1].latest_checkpoint_sha256 == case.checkpoint_sha256
    assert caught.value.__cause__ is primary
