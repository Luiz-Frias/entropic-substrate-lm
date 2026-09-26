"""Production D1 meter strategy composes optional probes without inventing evidence."""

from __future__ import annotations

import pytest

from d1_telemetry_fixtures import STARTED_AT, start_provenance
from entropic_substrate.kernel.evidence_class.resource_provenance import EvidenceGap
from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass
from entropic_substrate.ladder.capture.resource.resource_ledger import ResourceLedger
from entropic_substrate.ladder.capture.resource.resource_meter_models import (
    PositionFlopModel,
    PowerSample,
    ResourcePhase,
)
from entropic_substrate.ladder.capture.resource.resource_meter_strategy import (
    D1ProductionResourceMeter,
)
from entropic_substrate.ladder.capture.resource.resource_work import CostReading

pytestmark = pytest.mark.unit


class _Timer:
    def __init__(self, *seconds: float) -> None:
        self.seconds = list(seconds)
        self.started: list[ResourcePhase] = []

    def start(self, phase: ResourcePhase) -> object:
        self.started.append(phase)
        return (phase, len(self.started))

    def stop(self, ticket: object, phase: ResourcePhase) -> float:
        assert ticket == (phase, len(self.started))
        return self.seconds.pop(0)


class _Memory:
    def __init__(self, allocated: int, reserved: int) -> None:
        self.peaks = (allocated, reserved)

    def peak_bytes(self) -> tuple[int, int]:
        return self.peaks


class _Power:
    def __init__(self, *samples: PowerSample) -> None:
        self.values = tuple(samples)

    def samples(self) -> tuple[PowerSample, ...]:
        return self.values


def _flops(*, train_online: int = 11) -> PositionFlopModel:
    return PositionFlopModel(
        "d1-operation-count-v1",
        train_online,
        7,
        5,
        3,
    )


def _power() -> _Power:
    return _Power(
        PowerSample(0, 100.0),
        PowerSample(1_000_000_000, 200.0),
    )


def test_strategy_separates_train_eval_and_labels_every_derived_channel() -> None:
    timer = _Timer(1.25, 0.5)
    meter = D1ProductionResourceMeter(
        start_provenance(),
        phase_timer=timer,
        peak_memory=_Memory(8_000, 10_000),
        flop_model=_flops(),
        power_samples=_power(),
        max_power_gap_seconds=2.0,
    )
    with meter.measure(ResourcePhase.TRAIN):
        pass
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with meter.measure(ResourcePhase.EVAL):
            raise RuntimeError("evaluation failed")
    meter.record_positions(ResourcePhase.TRAIN, online_positions=10, target_positions=4)
    meter.record_positions(ResourcePhase.EVAL, online_positions=5, target_positions=2)

    snapshot = meter.snapshot(trainer_wall_seconds=1800.0)

    assert snapshot.gpu_train_device_seconds.value == 1.25  # type: ignore[union-attr]
    assert snapshot.gpu_eval_device_seconds.value == 0.5  # type: ignore[union-attr]
    assert snapshot.peak_gpu_allocated_bytes.value == 8_000.0  # type: ignore[union-attr]
    assert snapshot.cumulative_train_flops.value == 138.0  # type: ignore[union-attr]
    assert snapshot.cumulative_train_flops.evidence_class is EvidenceClass.MODEL_ESTIMATE  # type: ignore[union-attr]
    assert snapshot.flop_model_identity == _flops().identity_digest
    assert snapshot.energy_joules.value == 150.0  # type: ignore[union-attr]
    assert snapshot.energy_sample_count == 2
    assert type(snapshot.rate_estimated_cost) is CostReading
    assert snapshot.rate_estimated_cost.amount.canonical_amount == "0.795"


def test_unavailable_optional_probes_are_field_bound_gaps_but_rate_cost_is_available() -> None:
    meter = D1ProductionResourceMeter(start_provenance())
    snapshot = meter.snapshot(trainer_wall_seconds=10.0)

    for field in (
        "gpu_train_device_seconds",
        "gpu_eval_device_seconds",
        "peak_gpu_allocated_bytes",
        "peak_gpu_reserved_bytes",
        "cumulative_train_flops",
        "cumulative_eval_flops",
        "flop_model_identity",
        "energy_joules",
        "energy_sample_count",
    ):
        value = getattr(snapshot, field)
        assert type(value) is EvidenceGap
        assert value.field == field
    assert type(snapshot.rate_estimated_cost) is CostReading


def test_strategy_fails_loud_on_overlapping_phases_or_regressing_wall_and_samples() -> None:
    meter = D1ProductionResourceMeter(
        start_provenance(), phase_timer=_Timer(1.0), power_samples=_power()
    )
    with meter.measure(ResourcePhase.TRAIN):
        with pytest.raises(RuntimeError, match="already active"):
            with meter.measure(ResourcePhase.EVAL):
                pass
    meter.snapshot(trainer_wall_seconds=2.0)
    with pytest.raises(ValueError, match="trainer_wall_seconds"):
        meter.snapshot(trainer_wall_seconds=1.0)
    power = _power()
    meter = D1ProductionResourceMeter(start_provenance(), power_samples=power)
    meter.snapshot(trainer_wall_seconds=1.0)
    power.values = power.values[:1]
    with pytest.raises(ValueError, match="append-only"):
        meter.snapshot(trainer_wall_seconds=2.0)


def test_resume_adds_cumulative_work_cost_energy_and_uses_maximum_vram() -> None:
    first_meter = D1ProductionResourceMeter(
        start_provenance(),
        phase_timer=_Timer(1.0, 0.25),
        peak_memory=_Memory(8_000, 10_000),
        flop_model=_flops(),
        power_samples=_power(),
    )
    for phase in (ResourcePhase.TRAIN, ResourcePhase.EVAL):
        with first_meter.measure(phase):
            pass
        first_meter.record_positions(phase, online_positions=10, target_positions=4)
    ledger = ResourceLedger(start_provenance(), monotonic_origin_ns=0, meter_strategy=first_meter)
    ledger.record_eval(
        online_positions=1,
        target_positions=1,
        training_step=1,
        observed_at=STARTED_AT,
        monotonic_ns=1_800_000_000_000,
    )

    resumed = D1ProductionResourceMeter.restore(
        start_provenance(1, "f" * 64),
        ledger.current_state(),
        phase_timer=_Timer(0.5, 0.1),
        peak_memory=_Memory(7_000, 12_000),
        flop_model=_flops(),
        power_samples=_power(),
    )
    for phase in (ResourcePhase.TRAIN, ResourcePhase.EVAL):
        with resumed.measure(phase):
            pass
        resumed.record_positions(phase, online_positions=5, target_positions=2)
    snapshot = resumed.snapshot(trainer_wall_seconds=2700.0)

    assert snapshot.gpu_train_device_seconds.value == 1.5  # type: ignore[union-attr]
    assert snapshot.gpu_eval_device_seconds.value == 0.35  # type: ignore[union-attr]
    assert snapshot.peak_gpu_allocated_bytes.value == 8_000.0  # type: ignore[union-attr]
    assert snapshot.peak_gpu_reserved_bytes.value == 12_000.0  # type: ignore[union-attr]
    assert snapshot.cumulative_train_flops.value == 207.0  # type: ignore[union-attr]
    assert snapshot.cumulative_eval_flops.value == 93.0  # type: ignore[union-attr]
    assert snapshot.energy_joules.value == 300.0  # type: ignore[union-attr]
    assert snapshot.energy_sample_count == 4
    assert snapshot.rate_estimated_cost.amount.canonical_amount == "1.1925"  # type: ignore[union-attr]


def test_resume_refuses_a_changed_flop_model_before_emitting_a_snapshot() -> None:
    first_meter = D1ProductionResourceMeter(start_provenance(), flop_model=_flops())
    first_meter.record_positions(ResourcePhase.TRAIN, online_positions=1, target_positions=1)
    ledger = ResourceLedger(start_provenance(), monotonic_origin_ns=0, meter_strategy=first_meter)
    ledger.record_eval(
        online_positions=1,
        target_positions=1,
        training_step=1,
        observed_at=STARTED_AT,
        monotonic_ns=1,
    )

    with pytest.raises(ValueError, match="flop model identity"):
        D1ProductionResourceMeter.restore(
            start_provenance(1, "f" * 64),
            ledger.current_state(),
            flop_model=_flops(train_online=12),
        )
