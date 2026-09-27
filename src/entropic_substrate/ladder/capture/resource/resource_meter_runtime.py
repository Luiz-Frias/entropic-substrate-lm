"""CUDA, monotonic-clock, and NVML adapters for D1 production resource ports."""

from __future__ import annotations

import math
import threading
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Protocol, Self, cast, runtime_checkable

from entropic_substrate.ladder.capture.resource.resource_meter_models import (
    PowerSample,
    ResourcePhase,
)
from entropic_substrate.ladder.capture.resource.resource_meter_strategy import (
    DevicePhaseTimerPort,
    PeakMemoryPort,
)

__all__ = [
    "NvmlPowerReader",
    "SystemMonotonicClock",
    "TimestampedPowerSampler",
    "TorchCudaPeakMemory",
    "TorchCudaPhaseTimer",
    "build_torch_cuda_adapters",
]


@runtime_checkable
class CudaEventPort(Protocol):
    def record(self) -> None: ...
    def elapsed_time(self, end_event: object) -> float: ...


@runtime_checkable
class CudaRuntimePort(Protocol):
    def is_available(self) -> bool: ...
    def Event(self, *, enable_timing: bool) -> CudaEventPort: ...
    def synchronize(self) -> None: ...
    def device(self, device: int) -> AbstractContextManager[object]: ...
    def max_memory_allocated(self, device: int | None = None) -> int: ...
    def max_memory_reserved(self, device: int | None = None) -> int: ...


@runtime_checkable
class MonotonicClockPort(Protocol):
    def monotonic_ns(self) -> int: ...


@runtime_checkable
class GpuPowerReaderPort(Protocol):
    def power_watts(self) -> float: ...


@runtime_checkable
class NvmlRuntimePort(Protocol):
    def nvmlInit(self) -> None: ...
    def nvmlShutdown(self) -> None: ...
    def nvmlDeviceGetHandleByIndex(self, index: int) -> object: ...
    def nvmlDeviceGetPowerUsage(self, handle: object) -> int: ...


def _cuda_runtime(value: object) -> CudaRuntimePort:
    if not isinstance(value, CudaRuntimePort):
        raise TypeError("CUDA runtime does not implement the resource adapter port")
    return value


def _device_index(value: int | None) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("CUDA device must be a nonnegative integer or None")


@dataclass(slots=True)
class _CudaTicket:
    owner: object
    phase: ResourcePhase
    start_event: CudaEventPort
    stopped: bool = False


class TorchCudaPhaseTimer(DevicePhaseTimerPort):
    """Measure current-stream device time with paired CUDA events."""

    def __init__(self, cuda: object, *, device: int | None = None) -> None:
        self._cuda = _cuda_runtime(cuda)
        _device_index(device)
        if not self._cuda.is_available():
            raise RuntimeError("CUDA event timer requires an available CUDA runtime")
        self._device = device
        self._owner = object()

    def _device_context(self) -> AbstractContextManager[object]:
        if self._device is None:
            return nullcontext()
        return self._cuda.device(self._device)

    def start(self, phase: ResourcePhase) -> object:
        if type(phase) is not ResourcePhase:
            raise TypeError("CUDA timing phase must be ResourcePhase")
        with self._device_context():
            event = self._cuda.Event(enable_timing=True)
            event.record()
        return _CudaTicket(self._owner, phase, event)

    def stop(self, ticket: object, phase: ResourcePhase) -> float:
        if type(ticket) is not _CudaTicket or ticket.owner is not self._owner:
            raise TypeError("CUDA timer ticket belongs to another timer")
        if ticket.phase is not phase:
            raise ValueError("CUDA timer ticket phase mismatch")
        if ticket.stopped:
            raise RuntimeError("CUDA timer ticket was already stopped")
        with self._device_context():
            end = self._cuda.Event(enable_timing=True)
            end.record()
            self._cuda.synchronize()
            elapsed_ms = ticket.start_event.elapsed_time(end)
        ticket.stopped = True
        if type(elapsed_ms) is not float or not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
            raise ValueError("CUDA event elapsed milliseconds must be finite and nonnegative")
        return elapsed_ms / 1000.0


class TorchCudaPeakMemory(PeakMemoryPort):
    """Read PyTorch allocator high-water marks as measured byte counters."""

    def __init__(self, cuda: object, *, device: int | None = None) -> None:
        self._cuda = _cuda_runtime(cuda)
        _device_index(device)
        if not self._cuda.is_available():
            raise RuntimeError("CUDA memory meter requires an available CUDA runtime")
        self._device = device

    def peak_bytes(self) -> tuple[int, int]:
        allocated = self._cuda.max_memory_allocated(self._device)
        reserved = self._cuda.max_memory_reserved(self._device)
        if any(type(value) is not int or value < 0 for value in (allocated, reserved)):
            raise ValueError("CUDA peak memory counters must be nonnegative integer bytes")
        if reserved < allocated:
            raise ValueError("CUDA peak reserved bytes cannot be below allocated bytes")
        return allocated, reserved


def build_torch_cuda_adapters(
    *, cuda: object | None = None, device: int | None = None
) -> tuple[TorchCudaPhaseTimer | None, TorchCudaPeakMemory | None]:
    """Return explicit absence on CPU, otherwise construct both CUDA meter adapters."""
    if cuda is None:
        import torch

        cuda = torch.cuda
    runtime = _cuda_runtime(cuda)
    _device_index(device)
    if not runtime.is_available():
        return None, None
    return (
        TorchCudaPhaseTimer(runtime, device=device),
        TorchCudaPeakMemory(runtime, device=device),
    )


class SystemMonotonicClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()


class TimestampedPowerSampler:
    """Thread-safe append-only source of measured timestamped GPU power samples."""

    def __init__(
        self,
        *,
        power_reader: GpuPowerReaderPort,
        clock: MonotonicClockPort | None = None,
    ) -> None:
        active_clock = clock or SystemMonotonicClock()
        if not isinstance(cast(object, active_clock), MonotonicClockPort):
            raise TypeError("power sampler clock does not implement MonotonicClockPort")
        if not isinstance(cast(object, power_reader), GpuPowerReaderPort):
            raise TypeError("power reader does not implement GpuPowerReaderPort")
        self._clock = active_clock
        self._reader = power_reader
        self._samples: list[PowerSample] = []
        self._lock = threading.Lock()

    def record(self) -> PowerSample:
        sample = PowerSample(self._clock.monotonic_ns(), self._reader.power_watts())
        with self._lock:
            if self._samples and sample.monotonic_ns <= self._samples[-1].monotonic_ns:
                raise ValueError("power sample timestamps must strictly increase")
            self._samples.append(sample)
        return sample

    def samples(self) -> tuple[PowerSample, ...]:
        with self._lock:
            return tuple(self._samples)


class NvmlPowerReader(GpuPowerReaderPort):
    """Own one NVML session and expose its milliwatt counter as watts."""

    def __init__(self, runtime: object, *, device_index: int = 0) -> None:
        if not isinstance(runtime, NvmlRuntimePort):
            raise TypeError("NVML runtime does not implement the power reader port")
        if type(device_index) is not int or device_index < 0:
            raise ValueError("device_index must be a nonnegative integer")
        self._runtime = runtime
        self._closed = True
        runtime.nvmlInit()
        try:
            self._handle = runtime.nvmlDeviceGetHandleByIndex(device_index)
        except BaseException:
            runtime.nvmlShutdown()
            raise
        self._closed = False

    @classmethod
    def from_pynvml(cls, *, device_index: int = 0) -> Self:
        import pynvml

        return cls(pynvml, device_index=device_index)

    def power_watts(self) -> float:
        if self._closed:
            raise RuntimeError("NVML power reader is closed")
        milliwatts = self._runtime.nvmlDeviceGetPowerUsage(self._handle)
        if type(milliwatts) is not int or milliwatts < 0:
            raise ValueError("NVML power milliwatts must be a nonnegative integer")
        return milliwatts / 1000.0

    def close(self) -> None:
        if not self._closed:
            self._runtime.nvmlShutdown()
            self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("NVML power reader is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
