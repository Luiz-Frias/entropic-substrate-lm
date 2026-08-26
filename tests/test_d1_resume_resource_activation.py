"""Resumed D1 training restores and drives the registered resource activation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import entropic_substrate.d1_segment_engine as segment_engine
from d1_resume_fixtures import (
    RecordingObserver,
    captured_step_one_case,
    origin,
    resume_start,
)
from d1_trainer_fixtures import DeterministicTrainerClock
from entropic_substrate.d1_resource_activation import (
    D1ProductionResourceActivationFactory,
)
from entropic_substrate.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.d1_resume_trainer import D1ResumeRunFailure, run_d1_resume
from entropic_substrate.d1_tracking import D1MetricExtractor
from entropic_substrate.evidence.d1_journal import read_journal
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.evidence.resource_meter_models import (
    PositionFlopModel,
    ResourcePhase,
)

pytestmark = pytest.mark.unit


class _Timer:
    def __init__(self) -> None:
        self.starts: list[ResourcePhase] = []
        self.stops: list[ResourcePhase] = []

    def start(self, phase: ResourcePhase) -> object:
        self.starts.append(phase)
        return phase

    def stop(self, ticket: object, phase: ResourcePhase) -> float:
        assert ticket is phase
        self.stops.append(phase)
        return 0.5 if phase is ResourcePhase.TRAIN else 0.25


class _Memory:
    def peak_bytes(self) -> tuple[int, int]:
        return 1_000, 2_000


def _factory(timers: list[_Timer]) -> D1ProductionResourceActivationFactory:
    def timer_factory(_start: object) -> _Timer:
        timer = _Timer()
        timers.append(timer)
        return timer

    return D1ProductionResourceActivationFactory(
        registration_digest="b" * 64,
        meter_provider="torch-cuda",
        device_scope="cuda:0",
        phase_timer_factory=timer_factory,  # type: ignore[arg-type]
        peak_memory_factory=lambda _start: _Memory(),
        flop_model=PositionFlopModel("resume-d1-flops", 2, 3, 5, 7),
    )


def test_resume_restores_meter_state_and_emits_exact_snapshot_and_mlflow_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timers: list[_Timer] = []
    factory = _factory(timers)
    case = captured_step_one_case(tmp_path, monkeypatch, resource_activation_factory=factory)
    completion = run_d1_resume(
        plan=case.plan,
        start=replace(
            resume_start(case.plan, case.checkpoint_sha256),
            resource_activation=factory.registration,
        ),
        origin=origin(),
        locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
        observer=RecordingObserver(),
        clock=DeterministicTrainerClock(),
        resource_activation_factory=factory,
    )

    assert len(timers) == 2
    assert (
        timers[1].starts
        == timers[1].stops
        == [
            ResourcePhase.TRAIN,
            ResourcePhase.EVAL,
        ]
    )
    work = completion.final_resource_work
    assert work.gpu_train_device_seconds.value == 1.0  # type: ignore[union-attr]
    assert work.gpu_eval_device_seconds.value == 0.75  # type: ignore[union-attr]
    assert work.cumulative_train_flops.value == 46.0  # type: ignore[union-attr]
    assert work.cumulative_eval_flops.value == 165.0  # type: ignore[union-attr]
    assert work.peak_gpu_reserved_bytes.value == 2_000.0  # type: ignore[union-attr]

    final_eval = [
        event
        for event in read_journal(case.journal_path)
        if event.event_kind is JournalEventKind.EVAL
    ][-1]
    extracted = D1MetricExtractor().extract(final_eval)
    assert extracted is not None
    assert extracted.metrics["d1/resource/measured/gpu_eval_device_seconds"] == 0.75
    assert extracted.metrics["d1/resource/model_estimate/cumulative_train_flops"] == 46.0


def test_failed_resumed_train_is_timed_without_modeling_a_complete_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timers: list[_Timer] = []
    factory = _factory(timers)
    case = captured_step_one_case(tmp_path, monkeypatch, resource_activation_factory=factory)

    def fail_step(**_kwargs: object) -> object:
        raise RuntimeError("deliberate resumed transaction failure")

    monkeypatch.setattr(segment_engine, "apply_d1_training_step", fail_step)
    with pytest.raises(D1ResumeRunFailure) as captured:
        run_d1_resume(
            plan=case.plan,
            start=replace(
                resume_start(case.plan, case.checkpoint_sha256),
                resource_activation=factory.registration,
            ),
            origin=origin(),
            locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
            observer=RecordingObserver(),
            clock=DeterministicTrainerClock(),
            resource_activation_factory=factory,
        )

    assert timers[1].starts == timers[1].stops == [ResourcePhase.TRAIN]
    readings = dict(captured.value.handoff.termination.resource_ledger_state.last_optional_readings)
    assert readings["gpu_train_device_seconds"].value == 1.0  # type: ignore[union-attr]
    assert readings["cumulative_train_flops"].value == 23.0  # type: ignore[union-attr]
