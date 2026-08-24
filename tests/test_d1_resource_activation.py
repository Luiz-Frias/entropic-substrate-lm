"""D1 resource activation selects production evidence only through an injected factory."""

from __future__ import annotations

from dataclasses import replace

import pytest

from d1_telemetry_fixtures import STARTED_AT, start_provenance
from entropic_substrate.d1_resource_activation import (
    D1ProductionResourceActivationFactory,
    PowerSamplingRegistration,
    UnavailableResourceActivationFactory,
)
from entropic_substrate.evidence.resource_ledger import ResourceLedger
from entropic_substrate.evidence.resource_meter_models import (
    PositionFlopModel,
    PowerSample,
    ResourcePhase,
)
from entropic_substrate.evidence.resource_provenance import EvidenceGap

pytestmark = pytest.mark.unit


class _Samples:
    def samples(self) -> tuple[PowerSample, ...]:
        return (PowerSample(0, 100.0), PowerSample(1_000_000_000, 100.0))


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


def _flops() -> PositionFlopModel:
    return PositionFlopModel("registered-d1-flops", 2, 3, 5, 7)


def _registration(*, cadence: float = 1.0) -> PowerSamplingRegistration:
    return PowerSamplingRegistration(cadence_seconds=cadence, max_gap_seconds=2.0)


def test_absent_factory_builds_active_noop_meter_with_only_typed_gaps() -> None:
    activation = UnavailableResourceActivationFactory().fresh(start_provenance())
    with activation:
        with activation.meter.measure(ResourcePhase.TRAIN):
            pass
        activation.meter.record_positions(
            ResourcePhase.TRAIN, online_positions=1, target_positions=2
        )
    snapshot = activation.meter.snapshot(trainer_wall_seconds=1.0)

    assert all(
        type(getattr(snapshot, field)) is EvidenceGap
        for field in (
            "gpu_train_device_seconds",
            "peak_gpu_allocated_bytes",
            "cumulative_train_flops",
            "flop_model_identity",
            "energy_joules",
            "energy_sample_count",
            "rate_estimated_cost",
        )
    )


def test_registered_factory_uses_explicit_flop_and_power_methodologies() -> None:
    registration = _registration()
    lifecycle = _Lifecycle(registration)
    factory = D1ProductionResourceActivationFactory(
        registration_digest="a" * 64,
        flop_model=_flops(),
        power_registration=registration,
        power_lifecycle_factory=lambda _start, _registration: lifecycle,
    )
    activation = factory.fresh(start_provenance())
    with activation:
        activation.meter.record_positions(
            ResourcePhase.TRAIN, online_positions=4, target_positions=2
        )
        snapshot = activation.meter.snapshot(trainer_wall_seconds=1.0)

    assert lifecycle.entered == lifecycle.exited == 1
    assert snapshot.cumulative_train_flops.value == 14.0  # type: ignore[union-attr]
    assert snapshot.flop_model_identity == _flops().identity_digest
    assert snapshot.energy_joules.value == 100.0  # type: ignore[union-attr]
    assert snapshot.energy_sample_count == 2


def test_factory_resume_seeds_production_meter_from_v2_ledger_state() -> None:
    registration = _registration()
    factory = D1ProductionResourceActivationFactory(
        registration_digest="a" * 64,
        flop_model=_flops(),
        power_registration=registration,
        power_lifecycle_factory=lambda _start, _registration: _Lifecycle(registration),
    )
    first = factory.fresh(start_provenance())
    first.meter.record_positions(ResourcePhase.TRAIN, online_positions=4, target_positions=2)
    ledger = ResourceLedger(start_provenance(), monotonic_origin_ns=0, meter_strategy=first.meter)
    ledger.record_eval(
        online_positions=1,
        target_positions=1,
        training_step=1,
        observed_at=STARTED_AT,
        monotonic_ns=1_000_000_000,
    )
    start = start_provenance(1, "f" * 64)
    resumed = factory.resume(start, ledger.current_state())
    resumed.meter.record_positions(ResourcePhase.TRAIN, online_positions=2, target_positions=1)
    snapshot = resumed.meter.snapshot(trainer_wall_seconds=2.0)

    assert snapshot.cumulative_train_flops.value == 21.0  # type: ignore[union-attr]
    assert snapshot.energy_joules.value == 200.0  # type: ignore[union-attr]
    assert snapshot.energy_sample_count == 4


def test_power_registration_and_lifecycle_must_be_explicit_and_identical() -> None:
    with pytest.raises(ValueError, match="together"):
        D1ProductionResourceActivationFactory(
            registration_digest="a" * 64,
            power_registration=_registration(),
        )
    with pytest.raises(ValueError, match="max_gap_seconds"):
        PowerSamplingRegistration(cadence_seconds=2.0, max_gap_seconds=1.0)

    registration = _registration()
    changed = replace(registration, cadence_seconds=0.5)
    factory = D1ProductionResourceActivationFactory(
        registration_digest="a" * 64,
        power_registration=registration,
        power_lifecycle_factory=lambda _start, _registration: _Lifecycle(changed),
    )
    with pytest.raises(ValueError, match="registration"):
        factory.fresh(start_provenance())
