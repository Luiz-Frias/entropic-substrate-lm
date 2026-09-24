"""Composition boundary for activating optional D1 production resource evidence."""

from __future__ import annotations

import re
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol, Self, cast, runtime_checkable

from entropic_substrate.__legacy.evidence.resource_ledger import (
    ResourceLedgerState,
    ResourceMeterSnapshot,
    ResourceMeterStrategy,
    UnavailableMeterStrategy,
)
from entropic_substrate.__legacy.evidence.resource_meter_models import (
    PositionFlopModel,
    ResourcePhase,
)
from entropic_substrate.__legacy.evidence.resource_meter_strategy import (
    D1ProductionResourceMeter,
    DevicePhaseTimerPort,
    PeakMemoryPort,
    PowerSampleSourcePort,
)
from entropic_substrate.kernel.evidence_class.resource_provenance import (
    PowerSamplingRegistration,
    ResourceActivationMode,
    ResourceActivationRegistration,
    RunStartProvenance,
    unavailable_resource_activation_registration,
)

__all__ = [
    "D1ActiveResourceMeter",
    "D1ProductionResourceActivationFactory",
    "D1ResourceActivation",
    "D1ResourceActivationFactory",
    "PowerSamplingLifecyclePort",
    "PowerSamplingRegistration",
    "UnavailableResourceActivationFactory",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@runtime_checkable
class D1ActiveResourceMeter(ResourceMeterStrategy, Protocol):
    def measure(self, phase: ResourcePhase) -> AbstractContextManager[None]: ...
    def record_positions(
        self, phase: ResourcePhase, *, online_positions: int, target_positions: int
    ) -> None: ...


@runtime_checkable
class PowerSamplingLifecyclePort(Protocol):
    @property
    def registration(self) -> PowerSamplingRegistration | None: ...
    @property
    def sample_source(self) -> PowerSampleSourcePort | None: ...
    def __enter__(self) -> object: ...
    def __exit__(self, *_: object) -> None: ...


class _InactivePowerSamplingLifecycle:
    @property
    def registration(self) -> None:
        return None

    @property
    def sample_source(self) -> None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _UnavailableActiveResourceMeter:
    def __init__(self) -> None:
        self._snapshot_strategy = UnavailableMeterStrategy()

    @contextmanager
    def measure(self, phase: ResourcePhase) -> Generator[None]:
        if type(phase) is not ResourcePhase:
            raise TypeError("resource phase must be ResourcePhase")
        yield

    def record_positions(
        self, phase: ResourcePhase, *, online_positions: int, target_positions: int
    ) -> None:
        if type(phase) is not ResourcePhase:
            raise TypeError("resource phase must be ResourcePhase")
        if any(
            type(value) is not int or value < 0 for value in (online_positions, target_positions)
        ):
            raise ValueError("resource positions must be nonnegative integers")

    def snapshot(self, *, trainer_wall_seconds: float) -> ResourceMeterSnapshot:
        return self._snapshot_strategy.snapshot(trainer_wall_seconds=trainer_wall_seconds)


class D1ResourceActivation:
    def __init__(
        self,
        meter: D1ActiveResourceMeter,
        power_lifecycle: PowerSamplingLifecyclePort,
    ) -> None:
        if not isinstance(cast(object, meter), D1ActiveResourceMeter):
            raise TypeError("D1 activation meter must implement D1ActiveResourceMeter")
        if not isinstance(cast(object, power_lifecycle), PowerSamplingLifecyclePort):
            raise TypeError("D1 activation lifecycle must implement PowerSamplingLifecyclePort")
        self.meter = meter
        self.power_lifecycle = power_lifecycle
        self._entered = False

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("D1 resource activation is already entered")
        self.power_lifecycle.__enter__()
        self._entered = True
        return self

    def __exit__(self, *args: object) -> None:
        if not self._entered:
            raise RuntimeError("D1 resource activation was not entered")
        try:
            self.power_lifecycle.__exit__(*args)
        finally:
            self._entered = False


@runtime_checkable
class D1ResourceActivationFactory(Protocol):
    @property
    def registration(self) -> ResourceActivationRegistration: ...
    def fresh(self, start: RunStartProvenance) -> D1ResourceActivation: ...
    def resume(
        self, start: RunStartProvenance, state: ResourceLedgerState
    ) -> D1ResourceActivation: ...


class UnavailableResourceActivationFactory:
    @property
    def registration(self) -> ResourceActivationRegistration:
        return unavailable_resource_activation_registration()

    def _activation(self) -> D1ResourceActivation:
        return D1ResourceActivation(
            _UnavailableActiveResourceMeter(), _InactivePowerSamplingLifecycle()
        )

    def fresh(self, start: RunStartProvenance) -> D1ResourceActivation:
        if type(start) is not RunStartProvenance:
            raise TypeError("D1 fresh resource activation requires RunStartProvenance")
        return self._activation()

    def resume(self, start: RunStartProvenance, state: ResourceLedgerState) -> D1ResourceActivation:
        if type(start) is not RunStartProvenance or type(state) is not ResourceLedgerState:
            raise TypeError("D1 resume resource activation requires typed start and ledger state")
        return self._activation()


type PhaseTimerFactory = Callable[[RunStartProvenance], DevicePhaseTimerPort | None]
type PeakMemoryFactory = Callable[[RunStartProvenance], PeakMemoryPort | None]
type PowerLifecycleFactory = Callable[
    [RunStartProvenance, PowerSamplingRegistration], PowerSamplingLifecyclePort
]


class D1ProductionResourceActivationFactory:
    """Build one segment meter from operator-supplied, registered measurement choices."""

    def __init__(
        self,
        *,
        registration_digest: str,
        meter_provider: str,
        device_scope: str,
        phase_timer_factory: PhaseTimerFactory | None = None,
        peak_memory_factory: PeakMemoryFactory | None = None,
        flop_model: PositionFlopModel | None = None,
        power_registration: PowerSamplingRegistration | None = None,
        power_lifecycle_factory: PowerLifecycleFactory | None = None,
    ) -> None:
        if type(registration_digest) is not str or _SHA256.fullmatch(registration_digest) is None:
            raise ValueError("resource meter registration_digest must be a lowercase SHA-256")
        if phase_timer_factory is not None and not callable(phase_timer_factory):
            raise TypeError("phase_timer_factory must be callable")
        if peak_memory_factory is not None and not callable(peak_memory_factory):
            raise TypeError("peak_memory_factory must be callable")
        if flop_model is not None and type(flop_model) is not PositionFlopModel:
            raise TypeError("flop_model must be PositionFlopModel")
        if (power_registration is None) is not (power_lifecycle_factory is None):
            raise ValueError("power registration and lifecycle factory must be supplied together")
        self.registration = ResourceActivationRegistration(
            factory_registration_digest=registration_digest,
            mode=ResourceActivationMode.PRODUCTION,
            meter_provider=meter_provider,
            device_scope=device_scope,
            flop_model_identity=(None if flop_model is None else flop_model.identity_digest),
            power_sampling=power_registration,
        )
        self.registration_digest = self.registration.factory_registration_digest
        self._timer_factory = phase_timer_factory
        self._memory_factory = peak_memory_factory
        self._flop_model = flop_model
        self._power_registration = power_registration
        self._power_factory = power_lifecycle_factory

    def _lifecycle(self, start: RunStartProvenance) -> PowerSamplingLifecyclePort:
        if self._power_registration is None or self._power_factory is None:
            return _InactivePowerSamplingLifecycle()
        lifecycle = self._power_factory(start, self._power_registration)
        if not isinstance(cast(object, lifecycle), PowerSamplingLifecyclePort):
            raise TypeError("power lifecycle factory returned an invalid lifecycle")
        if lifecycle.registration != self._power_registration or lifecycle.sample_source is None:
            raise ValueError("power lifecycle registration does not match its sampling policy")
        return lifecycle

    def _ports(
        self, start: RunStartProvenance
    ) -> tuple[DevicePhaseTimerPort | None, PeakMemoryPort | None]:
        timer = self._timer_factory(start) if self._timer_factory is not None else None
        memory = self._memory_factory(start) if self._memory_factory is not None else None
        return timer, memory

    def fresh(self, start: RunStartProvenance) -> D1ResourceActivation:
        lifecycle = self._lifecycle(start)
        timer, memory = self._ports(start)
        kwargs: dict[str, object] = {
            "phase_timer": timer,
            "peak_memory": memory,
            "flop_model": self._flop_model,
            "power_samples": lifecycle.sample_source,
        }
        if lifecycle.registration is not None:
            kwargs["max_power_gap_seconds"] = lifecycle.registration.max_gap_seconds
        meter = D1ProductionResourceMeter(start, **kwargs)  # type: ignore[arg-type]
        return D1ResourceActivation(meter, lifecycle)

    def resume(self, start: RunStartProvenance, state: ResourceLedgerState) -> D1ResourceActivation:
        lifecycle = self._lifecycle(start)
        timer, memory = self._ports(start)
        kwargs: dict[str, object] = {
            "phase_timer": timer,
            "peak_memory": memory,
            "flop_model": self._flop_model,
            "power_samples": lifecycle.sample_source,
        }
        if lifecycle.registration is not None:
            kwargs["max_power_gap_seconds"] = lifecycle.registration.max_gap_seconds
        meter = D1ProductionResourceMeter.restore(start, state, **kwargs)
        return D1ResourceActivation(meter, lifecycle)
