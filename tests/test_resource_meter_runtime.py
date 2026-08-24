"""Runtime meter adapters bind CUDA/NVML units without requiring live hardware."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import pytest

from entropic_substrate.evidence.resource_meter_models import ResourcePhase
from entropic_substrate.evidence.resource_meter_runtime import (
    NvmlPowerReader,
    TimestampedPowerSampler,
    TorchCudaPeakMemory,
    TorchCudaPhaseTimer,
    build_torch_cuda_adapters,
)

pytestmark = pytest.mark.unit


class _Event:
    def __init__(self, runtime: _Cuda) -> None:
        self.runtime = runtime
        self.recorded = 0

    def record(self) -> None:
        self.recorded += 1

    def elapsed_time(self, end: object) -> float:
        assert isinstance(end, _Event)
        return self.runtime.elapsed_ms.pop(0)


class _Cuda:
    def __init__(
        self,
        *,
        available: bool = True,
        elapsed_ms: tuple[float, ...] = (250.0,),
        allocated: int = 8_000,
        reserved: int = 10_000,
    ) -> None:
        self.available = available
        self.elapsed_ms = list(elapsed_ms)
        self.allocated = allocated
        self.reserved = reserved
        self.events: list[_Event] = []
        self.synchronizations = 0
        self.device_entries: list[int] = []

    def is_available(self) -> bool:
        return self.available

    def Event(self, *, enable_timing: bool) -> _Event:
        assert enable_timing is True
        event = _Event(self)
        self.events.append(event)
        return event

    def synchronize(self) -> None:
        self.synchronizations += 1

    @contextmanager
    def device(self, index: int) -> Generator[None]:
        self.device_entries.append(index)
        yield

    def max_memory_allocated(self, device: int | None = None) -> int:
        assert device in (None, 0)
        return self.allocated

    def max_memory_reserved(self, device: int | None = None) -> int:
        assert device in (None, 0)
        return self.reserved


def test_cuda_events_convert_milliseconds_and_enforce_ticket_phase_and_one_shot() -> None:
    cuda = _Cuda(elapsed_ms=(250.0,))
    timer = TorchCudaPhaseTimer(cuda)
    ticket = timer.start(ResourcePhase.TRAIN)

    assert timer.stop(ticket, ResourcePhase.TRAIN) == 0.25
    assert [event.recorded for event in cuda.events] == [1, 1]
    assert cuda.synchronizations == 1
    with pytest.raises(RuntimeError, match="already stopped"):
        timer.stop(ticket, ResourcePhase.TRAIN)

    other = timer.start(ResourcePhase.EVAL)
    with pytest.raises(ValueError, match="phase"):
        timer.stop(other, ResourcePhase.TRAIN)


def test_cuda_event_and_memory_defects_fail_loud() -> None:
    timer = TorchCudaPhaseTimer(_Cuda(elapsed_ms=(-1.0,)))
    with pytest.raises(ValueError, match="elapsed"):
        timer.stop(timer.start(ResourcePhase.TRAIN), ResourcePhase.TRAIN)
    with pytest.raises(ValueError, match="reserved"):
        TorchCudaPeakMemory(_Cuda(allocated=10_000, reserved=8_000)).peak_bytes()


def test_cuda_factory_returns_explicit_absence_or_two_live_adapters() -> None:
    assert build_torch_cuda_adapters(cuda=_Cuda(available=False)) == (None, None)
    timer, memory = build_torch_cuda_adapters(cuda=_Cuda(), device=0)
    assert type(timer) is TorchCudaPhaseTimer
    assert type(memory) is TorchCudaPeakMemory
    assert memory.peak_bytes() == (8_000, 10_000)


class _Clock:
    def __init__(self, *values: int) -> None:
        self.values = list(values)

    def monotonic_ns(self) -> int:
        return self.values.pop(0)


class _Power:
    def __init__(self, *values: float) -> None:
        self.values = list(values)

    def power_watts(self) -> float:
        return self.values.pop(0)


def test_timestamped_power_sampler_is_append_only_and_uses_monotonic_time() -> None:
    sampler = TimestampedPowerSampler(clock=_Clock(10, 20), power_reader=_Power(100.0, 200.0))
    sampler.record()
    sampler.record()

    assert tuple((item.monotonic_ns, item.watts) for item in sampler.samples()) == (
        (10, 100.0),
        (20, 200.0),
    )
    assert sampler.samples() is not sampler.samples()


def test_power_sampler_refuses_duplicate_time_and_nonfinite_reading() -> None:
    duplicate = TimestampedPowerSampler(clock=_Clock(10, 10), power_reader=_Power(1.0, 2.0))
    duplicate.record()
    with pytest.raises(ValueError, match="strictly increase"):
        duplicate.record()
    invalid = TimestampedPowerSampler(clock=_Clock(10), power_reader=_Power(float("nan")))
    with pytest.raises(ValueError, match="watts"):
        invalid.record()


class _Nvml:
    def __init__(self, milliwatts: int = 250_000) -> None:
        self.milliwatts = milliwatts
        self.initialized = self.shutdown = 0
        self.handle = object()

    def nvmlInit(self) -> None:
        self.initialized += 1

    def nvmlShutdown(self) -> None:
        self.shutdown += 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
        assert index == 0
        return self.handle

    def nvmlDeviceGetPowerUsage(self, handle: object) -> int:
        assert handle is self.handle
        return self.milliwatts


def test_nvml_reader_converts_milliwatts_and_owns_shutdown() -> None:
    runtime = _Nvml()
    with NvmlPowerReader(runtime, device_index=0) as reader:
        assert reader.power_watts() == 250.0
    assert runtime.initialized == 1
    assert runtime.shutdown == 1
    with pytest.raises(RuntimeError, match="closed"):
        reader.power_watts()


def test_nvml_reader_rejects_invalid_device_and_power_units() -> None:
    with pytest.raises(ValueError, match="device_index"):
        NvmlPowerReader(_Nvml(), device_index=-1)
    reader = NvmlPowerReader(_Nvml(milliwatts=-1))
    with pytest.raises(ValueError, match="milliwatts"):
        reader.power_watts()
    reader.close()
