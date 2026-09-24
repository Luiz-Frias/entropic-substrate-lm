"""Measured system runtime facts for the pod-side D1 trainer bootstrap."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import NoReturn, cast

from entropic_substrate.__legacy.d1_operator_values import D1RuntimeObservation

__all__ = ["D1RuntimeProbeError", "SystemD1RuntimeProbe"]


class D1RuntimeProbeError(RuntimeError):
    """Sanitized refusal when a required measured runtime fact is unavailable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"D1 system runtime probe refused [{code}]")


def _fail(code: str, error: Exception | None = None) -> NoReturn:
    del error
    raise D1RuntimeProbeError(code) from None


def _load_torch() -> object:
    import torch

    return torch


def _load_nvml() -> object:
    import pynvml

    return pynvml


def _text(value: object, *, code: str) -> str:
    if type(value) is bytes:
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as error:
            _fail(code, error)
    if not isinstance(value, str):
        _fail(code)
    normalized = str(value)
    if not normalized or normalized.strip() != normalized:
        _fail(code)
    return normalized


class SystemD1RuntimeProbe:
    """Measure the registered pre-construction wall origin and runtime facts.

    ``trainer_started_at`` is D1's after-preflight/before-construction work
    origin, not the later foreground command's OS process birth timestamp.
    """

    def __init__(
        self,
        *,
        torch_runtime: object | None = None,
        nvml_loader: Callable[[], object] | None = None,
        utc_now: Callable[[], datetime] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        self._torch = torch_runtime
        self._nvml_loader = nvml_loader or _load_nvml
        self._utc_now = utc_now or (lambda: datetime.now(UTC))
        self._monotonic_ns = monotonic_ns or time.monotonic_ns

    def _clock_boundary(self) -> tuple[datetime, int]:
        started_at = self._utc_now()
        monotonic = self._monotonic_ns()
        if (
            type(started_at) is not datetime
            or started_at.tzinfo is None
            or started_at.utcoffset() != timedelta(0)
        ):
            _fail("clock_unavailable")
        if type(monotonic) is not int or monotonic < 0:
            _fail("clock_unavailable")
        return started_at, monotonic

    def _torch_facts(self) -> tuple[str, bool, object]:
        try:
            runtime = _load_torch() if self._torch is None else self._torch
            pytorch = _text(getattr(runtime, "__version__", None), code="pytorch_unavailable")
            cuda = getattr(runtime, "cuda", None)
            is_available = getattr(cuda, "is_available", None)
            if not callable(is_available):
                _fail("cuda_availability_unavailable")
            available = is_available()
            if type(available) is not bool:
                _fail("cuda_availability_unavailable")
            return pytorch, available, getattr(getattr(runtime, "version", None), "cuda", None)
        except D1RuntimeProbeError:
            raise
        except Exception as error:
            _fail("pytorch_unavailable", error)

    def _driver_version(self) -> str:
        initialized = False
        shutdown_call: Callable[[], object] | None = None
        try:
            runtime = self._nvml_loader()
            initialize = getattr(runtime, "nvmlInit", None)
            read_driver = getattr(runtime, "nvmlSystemGetDriverVersion", None)
            shutdown = getattr(runtime, "nvmlShutdown", None)
            if not callable(initialize) or not callable(read_driver) or not callable(shutdown):
                _fail("nvml_unavailable")
            initialize_call = cast(Callable[[], object], initialize)
            read_driver_call = cast(Callable[[], object], read_driver)
            shutdown_call = cast(Callable[[], object], shutdown)
            initialize_call()
            initialized = True
            return _text(read_driver_call(), code="nvml_unavailable")
        except D1RuntimeProbeError:
            raise
        except Exception as error:
            _fail("nvml_unavailable", error)
        finally:
            if initialized and shutdown_call is not None:
                try:
                    shutdown_call()
                except Exception as error:
                    _fail("nvml_unavailable", error)

    def __call__(self, pod_id: str) -> D1RuntimeObservation:
        started_at, monotonic = self._clock_boundary()
        pytorch, cuda_available, raw_cuda = self._torch_facts()
        if cuda_available:
            cuda_version = _text(raw_cuda, code="cuda_version_unavailable")
            driver_version = self._driver_version()
        else:
            cuda_version = driver_version = "not-available"
        # D1OptimizationSpec rejects compile=True, so disabled is enforced by
        # the construction contract rather than guessed from ambient process state.
        return D1RuntimeObservation(
            pod_id=pod_id,
            trainer_started_at=started_at,
            cuda_version=cuda_version,
            driver_version=driver_version,
            pytorch_version=pytorch,
            compiler_enabled=False,
            compiler_version="disabled",
            trainer_monotonic_origin_ns=monotonic,
        )
