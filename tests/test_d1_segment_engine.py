"""Shared D1 segment evaluation and checkpoint ownership contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

import entropic_substrate.d1_segment_engine as segment_engine
from d1_resume_fixtures import RecordingObserver, captured_step_one_case, origin, resume_start
from d1_trainer_fixtures import (
    ORIGIN_NS,
    STARTED_AT,
    DeterministicTrainerClock,
    micro_plan,
    output_paths,
    run_start,
)
from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.d1_resume_trainer import run_d1_resume
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.experiments.d1_primitives import D1Arm

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("save_interval", "expected_checkpoint_steps"),
    [(1, [1, 2]), (3, [2])],
)
def test_fresh_routes_evaluation_and_checkpoint_through_shared_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    save_interval: int,
    expected_checkpoint_steps: list[int],
) -> None:
    plan = micro_plan(D1Arm.NONE, save_interval=save_interval)
    journal, checkpoint = output_paths(tmp_path, D1Arm.NONE)
    real_eval = segment_engine.record_d1_segment_evaluation
    real_checkpoint = segment_engine.save_d1_segment_checkpoint
    eval_steps: list[int] = []
    checkpoint_steps: list[int] = []

    def record_eval(**kwargs: object):
        eval_steps.append(kwargs["successful_step"])  # type: ignore[arg-type]
        return real_eval(**kwargs)  # type: ignore[arg-type]

    def record_checkpoint(**kwargs: object):
        checkpoint_steps.append(kwargs["successful_step"])  # type: ignore[arg-type]
        return real_checkpoint(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(segment_engine, "record_d1_segment_evaluation", record_eval)
    monkeypatch.setattr(segment_engine, "save_d1_segment_checkpoint", record_checkpoint)
    run_d1_fresh(
        plan=plan,
        start=run_start(plan),
        origin=D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS),
        locations=D1FreshRunLocations(journal, checkpoint),
        clock=DeterministicTrainerClock(),
    )

    assert eval_steps == [0, 1, 2]
    assert checkpoint_steps == expected_checkpoint_steps


def test_resume_routes_evaluation_and_checkpoint_through_shared_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)
    real_eval = segment_engine.record_d1_segment_evaluation
    real_checkpoint = segment_engine.save_d1_segment_checkpoint
    eval_steps: list[int] = []
    checkpoint_steps: list[int] = []

    def record_eval(**kwargs: object):
        eval_steps.append(kwargs["successful_step"])  # type: ignore[arg-type]
        return real_eval(**kwargs)  # type: ignore[arg-type]

    def record_checkpoint(**kwargs: object):
        checkpoint_steps.append(kwargs["successful_step"])  # type: ignore[arg-type]
        return real_checkpoint(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(segment_engine, "record_d1_segment_evaluation", record_eval)
    monkeypatch.setattr(segment_engine, "save_d1_segment_checkpoint", record_checkpoint)
    run_d1_resume(
        plan=case.plan,
        start=resume_start(case.plan, case.checkpoint_sha256),
        origin=origin(),
        locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
        observer=RecordingObserver(),
        clock=DeterministicTrainerClock(),
    )

    assert eval_steps == [2]
    assert checkpoint_steps == [2]
