"""Stateful production D1 resource strategy over injectable hardware meter ports."""

from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import contextmanager
from decimal import Decimal
from typing import Protocol, cast, runtime_checkable

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    EvidenceGap,
    Money,
    RunStartProvenance,
)
from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass
from entropic_substrate.ladder.capture.resource.resource_ledger import (
    ResourceLedgerState,
    ResourceMeterSnapshot,
)
from entropic_substrate.ladder.capture.resource.resource_meter_models import (
    PositionFlopModel,
    PowerSample,
    ResourcePhase,
    continuous_rate_cost,
    integrate_power_samples,
)
from entropic_substrate.ladder.capture.resource.resource_work import (
    CostReading,
    ReadingMethod,
    ScalarReading,
)

__all__ = [
    "DEFAULT_MAX_POWER_GAP_SECONDS",
    "D1ProductionResourceMeter",
    "DevicePhaseTimerPort",
    "PeakMemoryPort",
    "PowerSampleSourcePort",
]

DEFAULT_MAX_POWER_GAP_SECONDS = 5.0


@runtime_checkable
class DevicePhaseTimerPort(Protocol):
    def start(self, phase: ResourcePhase) -> object: ...
    def stop(self, ticket: object, phase: ResourcePhase) -> float: ...


@runtime_checkable
class PeakMemoryPort(Protocol):
    def peak_bytes(self) -> tuple[int, int]: ...


@runtime_checkable
class PowerSampleSourcePort(Protocol):
    def samples(self) -> tuple[PowerSample, ...]: ...


def _gap(field: str, reason: str) -> EvidenceGap:
    return EvidenceGap(field, reason)


def _nonnegative(value: float, *, label: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative float")


def _prior_scalar(
    readings: dict[str, ScalarReading | CostReading],
    field: str,
    evidence: EvidenceClass,
    method: ReadingMethod,
) -> float | None:
    value = readings.get(field)
    if value is None:
        return None
    if (
        type(value) is not ScalarReading
        or value.evidence_class is not evidence
        or value.method is not method
    ):
        raise ValueError(f"resume {field} has incompatible evidence semantics")
    return value.value


def _canonical_money(amount: Decimal, currency: str) -> Money:
    spelling = "0" if amount.is_zero() else format(amount.normalize(), "f")
    return Money(Decimal(spelling), currency)


class D1ProductionResourceMeter:
    """Accumulate optional resource evidence while preserving gaps and phase identity."""

    def __init__(
        self,
        start: RunStartProvenance,
        *,
        phase_timer: DevicePhaseTimerPort | None = None,
        peak_memory: PeakMemoryPort | None = None,
        flop_model: PositionFlopModel | None = None,
        power_samples: PowerSampleSourcePort | None = None,
        max_power_gap_seconds: float = DEFAULT_MAX_POWER_GAP_SECONDS,
    ) -> None:
        if type(start) is not RunStartProvenance:
            raise TypeError("production resource meter requires RunStartProvenance")
        for label, value, port in (
            ("phase_timer", phase_timer, DevicePhaseTimerPort),
            ("peak_memory", peak_memory, PeakMemoryPort),
            ("power_samples", power_samples, PowerSampleSourcePort),
        ):
            if value is not None and not isinstance(cast(object, value), port):
                raise TypeError(f"{label} does not implement its resource meter port")
        if flop_model is not None and type(flop_model) is not PositionFlopModel:
            raise TypeError("flop_model must be PositionFlopModel")
        if (
            type(max_power_gap_seconds) is not float
            or not math.isfinite(max_power_gap_seconds)
            or max_power_gap_seconds <= 0.0
        ):
            raise ValueError("max_power_gap_seconds must be a finite positive float")
        self.start = start
        self._timer = phase_timer
        self._memory = peak_memory
        self._flop_model = flop_model
        self._power = power_samples
        self._max_power_gap = max_power_gap_seconds
        self._active: ResourcePhase | None = None
        self._base_wall = self._last_wall = 0.0
        self._gpu_train = self._gpu_eval = 0.0
        self._train_flops = self._eval_flops = 0.0
        self._peak_allocated = self._peak_reserved = 0.0
        self._base_energy = 0.0
        self._base_energy_count = 0
        self._prior_cost = Money(Decimal("0"), start.quoted_rate.amount.currency)
        self._prior_samples: tuple[PowerSample, ...] = ()
        self._gpu_complete = self._memory_complete = self._flops_complete = True
        self._energy_complete = self._cost_complete = True

    @classmethod
    def restore(
        cls,
        start: RunStartProvenance,
        state: ResourceLedgerState,
        *,
        phase_timer: DevicePhaseTimerPort | None = None,
        peak_memory: PeakMemoryPort | None = None,
        flop_model: PositionFlopModel | None = None,
        power_samples: PowerSampleSourcePort | None = None,
        max_power_gap_seconds: float = DEFAULT_MAX_POWER_GAP_SECONDS,
    ) -> D1ProductionResourceMeter:
        if type(state) is not ResourceLedgerState:
            raise TypeError("resource meter restore requires ResourceLedgerState")
        if start.logical_run_id != state.logical_run_id:
            raise ValueError("resource meter restore logical_run_id changed")
        if start.execution_segment_index != state.execution_segment_index + 1:
            raise ValueError("resource meter restore requires the exact next segment")
        meter = cls(
            start,
            phase_timer=phase_timer,
            peak_memory=peak_memory,
            flop_model=flop_model,
            power_samples=power_samples,
            max_power_gap_seconds=max_power_gap_seconds,
        )
        readings = dict(state.last_optional_readings)
        meter._base_wall = meter._last_wall = state.trainer_wall_seconds
        prior_work = sum(
            (
                state.online_train_token_positions,
                state.target_train_token_positions,
                state.baseline_calibration_positions,
                state.eval_positions,
            )
        )
        train_gpu = _prior_scalar(
            readings,
            "gpu_train_device_seconds",
            EvidenceClass.MEASURED,
            ReadingMethod.GPU_DEVICE_COUNTER,
        )
        eval_gpu = _prior_scalar(
            readings,
            "gpu_eval_device_seconds",
            EvidenceClass.MEASURED,
            ReadingMethod.GPU_DEVICE_COUNTER,
        )
        meter._gpu_complete = prior_work == 0 or (train_gpu is not None and eval_gpu is not None)
        meter._gpu_train, meter._gpu_eval = train_gpu or 0.0, eval_gpu or 0.0
        allocated = _prior_scalar(
            readings,
            "peak_gpu_allocated_bytes",
            EvidenceClass.MEASURED,
            ReadingMethod.GPU_DEVICE_COUNTER,
        )
        reserved = _prior_scalar(
            readings,
            "peak_gpu_reserved_bytes",
            EvidenceClass.MEASURED,
            ReadingMethod.GPU_DEVICE_COUNTER,
        )
        meter._memory_complete = prior_work == 0 or (allocated is not None and reserved is not None)
        meter._peak_allocated, meter._peak_reserved = allocated or 0.0, reserved or 0.0
        train_flops = _prior_scalar(
            readings,
            "cumulative_train_flops",
            EvidenceClass.MODEL_ESTIMATE,
            ReadingMethod.FLOP_MODEL,
        )
        eval_flops = _prior_scalar(
            readings,
            "cumulative_eval_flops",
            EvidenceClass.MODEL_ESTIMATE,
            ReadingMethod.FLOP_MODEL,
        )
        meter._flops_complete = prior_work == 0 or (
            train_flops is not None and eval_flops is not None
        )
        meter._train_flops, meter._eval_flops = train_flops or 0.0, eval_flops or 0.0
        if meter._flop_model is not None and state.last_flop_model_identity not in (
            None,
            meter._flop_model.identity_digest,
        ):
            raise ValueError("resume flop model identity changed")
        meter._base_energy_count = state.last_energy_sample_count or 0
        energy = _prior_scalar(
            readings,
            "energy_joules",
            EvidenceClass.DERIVED_FROM_MEASURED,
            ReadingMethod.ENERGY_INTEGRATED,
        )
        meter._energy_complete = meter._base_energy_count == 0 or energy is not None
        meter._base_energy = energy or 0.0
        prior_cost = readings.get("rate_estimated_cost")
        if prior_cost is None:
            meter._cost_complete = state.trainer_wall_seconds == 0.0
        elif type(prior_cost) is not CostReading:
            raise ValueError("resume rate_estimated_cost has incompatible evidence semantics")
        else:
            if prior_cost.amount.currency != start.quoted_rate.amount.currency:
                raise ValueError("resume rate-estimated cost currency changed")
            meter._prior_cost = prior_cost.amount
        return meter

    @contextmanager
    def measure(self, phase: ResourcePhase) -> Generator[None]:
        if type(phase) is not ResourcePhase:
            raise TypeError("resource phase must be ResourcePhase")
        if self._active is not None:
            raise RuntimeError(f"resource phase {self._active.value} is already active")
        self._active = phase
        ticket = self._timer.start(phase) if self._timer is not None else None
        try:
            yield
        finally:
            try:
                if self._timer is not None:
                    elapsed = self._timer.stop(ticket, phase)
                    _nonnegative(elapsed, label=f"gpu_{phase.value}_device_seconds")
                    if phase is ResourcePhase.TRAIN:
                        self._gpu_train += elapsed
                    else:
                        self._gpu_eval += elapsed
            finally:
                self._active = None

    def record_positions(
        self, phase: ResourcePhase, *, online_positions: int, target_positions: int
    ) -> None:
        if type(phase) is not ResourcePhase:
            raise TypeError("resource phase must be ResourcePhase")
        if self._flop_model is None:
            if any(
                type(value) is not int or value < 0
                for value in (online_positions, target_positions)
            ):
                raise ValueError("resource positions must be nonnegative integers")
            return
        estimate = self._flop_model.estimate(
            phase, online_positions=online_positions, target_positions=target_positions
        )
        if phase is ResourcePhase.TRAIN:
            self._train_flops += estimate
        else:
            self._eval_flops += estimate

    def _gpu_reading(self, phase: ResourcePhase) -> ScalarReading | EvidenceGap:
        field = f"gpu_{phase.value}_device_seconds"
        if self._timer is None:
            return _gap(field, "CUDA phase timer unavailable")
        if not self._gpu_complete:
            return _gap(field, "prior execution segment lacks compatible GPU timing")
        value = self._gpu_train if phase is ResourcePhase.TRAIN else self._gpu_eval
        return ScalarReading(value, EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER)

    def _memory_readings(self) -> tuple[ScalarReading | EvidenceGap, ScalarReading | EvidenceGap]:
        fields = ("peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")
        if self._memory is None:
            return tuple(_gap(field, "CUDA memory meter unavailable") for field in fields)  # type: ignore[return-value]
        allocated, reserved = self._memory.peak_bytes()
        if any(type(value) is not int or value < 0 for value in (allocated, reserved)):
            raise ValueError("CUDA peak memory values must be nonnegative integer bytes")
        if reserved < allocated:
            raise ValueError("CUDA peak reserved bytes cannot be below allocated bytes")
        self._peak_allocated = max(self._peak_allocated, float(allocated))
        self._peak_reserved = max(self._peak_reserved, float(reserved))
        if not self._memory_complete:
            return tuple(
                _gap(field, "prior segment lacks compatible VRAM peaks") for field in fields
            )  # type: ignore[return-value]
        return (
            ScalarReading(
                self._peak_allocated, EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER
            ),
            ScalarReading(
                self._peak_reserved, EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER
            ),
        )

    def _flop_readings(self) -> tuple[ScalarReading | EvidenceGap, ScalarReading | EvidenceGap]:
        fields = ("cumulative_train_flops", "cumulative_eval_flops")
        if self._flop_model is None:
            return tuple(_gap(field, "FLOP model unavailable") for field in fields)  # type: ignore[return-value]
        if not self._flops_complete:
            return tuple(
                _gap(field, "prior segment lacks compatible FLOP estimates") for field in fields
            )  # type: ignore[return-value]
        return (
            ScalarReading(
                self._train_flops, EvidenceClass.MODEL_ESTIMATE, ReadingMethod.FLOP_MODEL
            ),
            ScalarReading(self._eval_flops, EvidenceClass.MODEL_ESTIMATE, ReadingMethod.FLOP_MODEL),
        )

    def _energy_readings(self) -> tuple[ScalarReading | EvidenceGap, int | EvidenceGap]:
        if self._power is None:
            reason = "timestamped GPU power samples unavailable"
            return _gap("energy_joules", reason), _gap("energy_sample_count", reason)
        samples = self._power.samples()
        if samples[: len(self._prior_samples)] != self._prior_samples:
            raise ValueError("power sample source must be append-only")
        self._prior_samples = samples
        result = integrate_power_samples(samples, max_gap_seconds=self._max_power_gap)
        count = self._base_energy_count + result.sample_count
        if type(result.joules) is EvidenceGap:
            return result.joules, count
        if not self._energy_complete:
            return _gap("energy_joules", "prior segment has incomplete energy coverage"), count
        joules = cast(ScalarReading, result.joules)
        return (
            ScalarReading(
                self._base_energy + joules.value,
                EvidenceClass.DERIVED_FROM_MEASURED,
                ReadingMethod.ENERGY_INTEGRATED,
            ),
            count,
        )

    def snapshot(self, *, trainer_wall_seconds: float) -> ResourceMeterSnapshot:
        _nonnegative(trainer_wall_seconds, label="trainer_wall_seconds")
        if trainer_wall_seconds < self._last_wall:
            raise ValueError("trainer_wall_seconds regressed")
        allocated, reserved = self._memory_readings()
        train_flops, eval_flops = self._flop_readings()
        energy, energy_count = self._energy_readings()
        if self._cost_complete:
            segment_cost = continuous_rate_cost(
                self.start.quoted_rate,
                trainer_wall_seconds=trainer_wall_seconds - self._base_wall,
                accelerator_count=self.start.accelerator_count,
            )
            amount = _canonical_money(
                self._prior_cost.amount + segment_cost.amount,
                self._prior_cost.currency,
            )
            cost: CostReading | EvidenceGap = CostReading(
                amount, EvidenceClass.RATE_ESTIMATE, ReadingMethod.RATE_TIMES_SCOPE
            )
        else:
            cost = _gap("rate_estimated_cost", "prior segment lacks rate-estimated cost")
        snapshot = ResourceMeterSnapshot(
            gpu_train_device_seconds=self._gpu_reading(ResourcePhase.TRAIN),
            gpu_eval_device_seconds=self._gpu_reading(ResourcePhase.EVAL),
            peak_gpu_allocated_bytes=allocated,
            peak_gpu_reserved_bytes=reserved,
            cumulative_train_flops=train_flops,
            cumulative_eval_flops=eval_flops,
            flop_model_identity=(
                self._flop_model.identity_digest
                if self._flop_model is not None
                else _gap("flop_model_identity", "FLOP model unavailable")
            ),
            energy_joules=energy,
            energy_sample_count=energy_count,
            rate_estimated_cost=cost,
        )
        self._last_wall = trainer_wall_seconds
        return snapshot
