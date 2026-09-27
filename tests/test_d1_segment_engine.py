"""Shared D1 segment evaluation and checkpoint ownership contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

import entropic_substrate.ladder.run.trainer.d1_segment_engine as segment_engine
from d1_resume_fixtures import RecordingObserver, captured_step_one_case, origin, resume_start
from d1_trainer_fixtures import (
    ORIGIN_NS,
    STARTED_AT,
    DeterministicTrainerClock,
    micro_plan,
    output_paths,
    run_start,
)
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.run.trainer.d1_fresh_trainer import (
    D1FreshRunFailure,
    D1FreshRunLocations,
    run_d1_fresh,
)
from entropic_substrate.ladder.run.trainer.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.ladder.run.trainer.d1_resume_trainer import run_d1_resume
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import D1TrainerWallOrigin

pytestmark = pytest.mark.unit


@dataclass
class _Sync:
    calls: list[tuple[int, Path, str, Path]] = field(default_factory=list)

    def sync(
        self,
        *,
        successful_step: int,
        run_root: Path,
        checkpoint: object,
        journal_path: Path,
    ) -> None:
        self.calls.append(
            (
                successful_step,
                run_root,
                checkpoint.sha256,  # type: ignore[attr-defined]
                journal_path,
            )
        )


class _FailingSync:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def sync(self, **kwargs: object) -> None:
        self.steps.append(kwargs["successful_step"])  # type: ignore[arg-type]
        raise RuntimeError("checkpoint sync sentinel")


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
    sync = _Sync()

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
        checkpoint_sync=sync,
    )

    assert eval_steps == [0, 1, 2]
    assert checkpoint_steps == expected_checkpoint_steps
    assert [step for step, *_ in sync.calls] == expected_checkpoint_steps
    assert all(root == checkpoint.parent for _, root, _, _ in sync.calls)
    assert all(path == journal for *_, path in sync.calls)


def test_resume_routes_evaluation_and_checkpoint_through_shared_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)
    real_eval = segment_engine.record_d1_segment_evaluation
    real_checkpoint = segment_engine.save_d1_segment_checkpoint
    eval_steps: list[int] = []
    checkpoint_steps: list[int] = []
    sync = _Sync()

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
        checkpoint_sync=sync,
    )

    assert eval_steps == [2]
    assert checkpoint_steps == [2]
    assert [step for step, *_ in sync.calls] == [2]
    assert sync.calls[0][1] == case.checkpoint_path.parent
    assert sync.calls[0][3] == case.journal_path


def test_checkpoint_sync_failure_aborts_before_another_training_step(tmp_path: Path) -> None:
    plan = micro_plan(D1Arm.NONE, save_interval=1)
    journal, checkpoint = output_paths(tmp_path, D1Arm.NONE)
    sync = _FailingSync()

    with pytest.raises(D1FreshRunFailure) as raised:
        run_d1_fresh(
            plan=plan,
            start=run_start(plan),
            origin=D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS),
            locations=D1FreshRunLocations(journal, checkpoint),
            clock=DeterministicTrainerClock(),
            checkpoint_sync=sync,
        )

    assert sync.steps == [1]
    assert raised.value.handoff.termination.successful_step == 1
    assert raised.value.handoff.termination.model_successful_step == 1


def test_fresh_routes_the_full_training_segment_through_shared_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = micro_plan(D1Arm.SG)
    journal, checkpoint = output_paths(tmp_path, D1Arm.SG)
    real_run = segment_engine.run_d1_training_segment
    starts: list[int] = []

    def record_run(**kwargs: object):
        assert "latest_checkpoint_sha256" not in kwargs
        starts.append(kwargs["successful_step"])  # type: ignore[arg-type]
        return real_run(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(segment_engine, "run_d1_training_segment", record_run)
    run_d1_fresh(
        plan=plan,
        start=run_start(plan),
        origin=D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS),
        locations=D1FreshRunLocations(journal, checkpoint),
        clock=DeterministicTrainerClock(),
    )

    assert starts == [0]


def test_resume_routes_the_full_training_segment_through_shared_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)
    real_run = segment_engine.run_d1_training_segment
    starts: list[int] = []

    def record_run(**kwargs: object):
        assert "latest_checkpoint_sha256" not in kwargs
        starts.append(kwargs["successful_step"])  # type: ignore[arg-type]
        return real_run(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(segment_engine, "run_d1_training_segment", record_run)
    run_d1_resume(
        plan=case.plan,
        start=resume_start(case.plan, case.checkpoint_sha256),
        origin=origin(),
        locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
        observer=RecordingObserver(),
        clock=DeterministicTrainerClock(),
    )

    assert starts == [1]
