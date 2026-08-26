"""Fresh D1 training drives only an explicitly activated resource strategy."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import entropic_substrate.d1_segment_engine as segment_engine
from d1_trainer_fixtures import (
    ORIGIN_NS,
    STARTED_AT,
    DeterministicTrainerClock,
    micro_plan,
    output_paths,
    run_start,
)
from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.d1_resource_activation import (
    D1ProductionResourceActivationFactory,
    PowerSamplingRegistration,
)
from entropic_substrate.d1_tracking import D1MetricExtractor
from entropic_substrate.d1_trainer_handoff import D1FreshRunFailure
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.evidence.d1_journal import read_journal
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.evidence.resource_meter_models import (
    PositionFlopModel,
    PowerSample,
    ResourcePhase,
)
from entropic_substrate.evidence.resource_provenance import EvidenceGap, PowerIntegrationMethod
from entropic_substrate.experiments.d1_primitives import D1Arm

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


class _Samples:
    def samples(self) -> tuple[PowerSample, ...]:
        return PowerSample(0, 100.0), PowerSample(1_000_000_000, 100.0)


class _Lifecycle:
    def __init__(self, registration: PowerSamplingRegistration) -> None:
        self.registration = registration
        self.sample_source = _Samples()
        self.entered = self.exited = 0

    def __enter__(self) -> _Lifecycle:
        self.entered += 1
        return self

    def __exit__(self, *_: object) -> None:
        self.exited += 1


def _factory(timer: _Timer, lifecycle: _Lifecycle) -> D1ProductionResourceActivationFactory:
    return D1ProductionResourceActivationFactory(
        registration_digest="a" * 64,
        meter_provider="torch-cuda+nvml",
        device_scope="cuda:0",
        phase_timer_factory=lambda _start: timer,
        peak_memory_factory=lambda _start: _Memory(),
        flop_model=PositionFlopModel("fresh-d1-flops", 2, 3, 5, 7),
        power_registration=lifecycle.registration,
        power_lifecycle_factory=lambda _start, _registration: lifecycle,
    )


def _locations(tmp_path: Path) -> tuple[D1FreshRunLocations, Path]:
    journal, checkpoint = output_paths(tmp_path, D1Arm.NONE)
    return D1FreshRunLocations(journal, checkpoint), journal


def test_fresh_run_records_exact_phase_work_snapshot_and_mlflow_projection(
    tmp_path: Path,
) -> None:
    plan = micro_plan(D1Arm.NONE)
    timer = _Timer()
    lifecycle = _Lifecycle(
        PowerSamplingRegistration(
            1.0,
            2.0,
            PowerIntegrationMethod.TRAPEZOIDAL_MONOTONIC_NS,
            "nvml",
            "cuda:0",
        )
    )
    factory = _factory(timer, lifecycle)
    locations, journal = _locations(tmp_path)
    completion = run_d1_fresh(
        plan=plan,
        start=replace(run_start(plan), resource_activation=factory.registration),
        origin=D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS),
        locations=locations,
        clock=DeterministicTrainerClock(),
        resource_activation_factory=factory,
    )

    assert (
        timer.starts
        == timer.stops
        == [
            ResourcePhase.EVAL,
            ResourcePhase.TRAIN,
            ResourcePhase.EVAL,
            ResourcePhase.TRAIN,
            ResourcePhase.EVAL,
        ]
    )
    assert lifecycle.entered == lifecycle.exited == 1
    work = completion.final_resource_work
    assert work.gpu_train_device_seconds.value == 1.0  # type: ignore[union-attr]
    assert work.gpu_eval_device_seconds.value == 0.75  # type: ignore[union-attr]
    assert work.peak_gpu_allocated_bytes.value == 1_000.0  # type: ignore[union-attr]
    assert work.cumulative_train_flops.value == 46.0  # type: ignore[union-attr]
    assert work.cumulative_eval_flops.value == 165.0  # type: ignore[union-attr]
    assert work.energy_joules.value == 100.0  # type: ignore[union-attr]
    assert work.energy_sample_count == 2

    final_eval = [
        event for event in read_journal(journal) if event.event_kind is JournalEventKind.EVAL
    ][-1]
    extracted = D1MetricExtractor().extract(final_eval)
    assert extracted is not None
    assert extracted.metrics["d1/resource/measured/gpu_train_device_seconds"] == 1.0
    assert extracted.metrics["d1/resource/model_estimate/cumulative_eval_flops"] == 165.0
    assert extracted.metrics["d1/resource/measured/energy_sample_count"] == 2.0


def test_fresh_default_keeps_optional_channels_as_gaps(tmp_path: Path) -> None:
    plan = micro_plan(D1Arm.NONE)
    locations, _journal = _locations(tmp_path)
    completion = run_d1_fresh(
        plan=plan,
        start=run_start(plan),
        origin=D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS),
        locations=locations,
        clock=DeterministicTrainerClock(),
    )
    assert type(completion.final_resource_work.gpu_train_device_seconds) is EvidenceGap
    assert type(completion.final_resource_work.cumulative_train_flops) is EvidenceGap


def test_failed_train_is_timed_once_and_reaches_terminal_ledger_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = micro_plan(D1Arm.NONE)
    timer = _Timer()
    lifecycle = _Lifecycle(
        PowerSamplingRegistration(
            1.0,
            2.0,
            PowerIntegrationMethod.TRAPEZOIDAL_MONOTONIC_NS,
            "nvml",
            "cuda:0",
        )
    )
    factory = _factory(timer, lifecycle)
    locations, _journal = _locations(tmp_path)

    def fail_step(**_kwargs: object) -> object:
        raise RuntimeError("deliberate failed transaction")

    monkeypatch.setattr(segment_engine, "apply_d1_training_step", fail_step)
    with pytest.raises(D1FreshRunFailure) as captured:
        run_d1_fresh(
            plan=plan,
            start=replace(run_start(plan), resource_activation=factory.registration),
            origin=D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS),
            locations=locations,
            clock=DeterministicTrainerClock(),
            resource_activation_factory=factory,
        )

    assert timer.starts == timer.stops == [ResourcePhase.EVAL, ResourcePhase.TRAIN]
    assert lifecycle.entered == lifecycle.exited == 1
    readings = dict(captured.value.handoff.termination.resource_ledger_state.last_optional_readings)
    assert readings["gpu_train_device_seconds"].value == 0.5  # type: ignore[union-attr]
    assert readings["cumulative_train_flops"].value == 0.0  # type: ignore[union-attr]
