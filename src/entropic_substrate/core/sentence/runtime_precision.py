"""Read-only precision admission for the registered FP32 geometric computation.

PyTorch documents a TF32 environment override independent of the normal setting:
https://docs.pytorch.org/docs/main/cuda_environment_variables.html
This guard never repairs ambient settings or changes another model's runtime.
"""

import os
from typing import Protocol, cast

import torch

from entropic_substrate.core.sentence.profile import HProfile


class _PrecisionBackend(Protocol):
    @property
    def fp32_precision(self) -> str: ...


class ExecutionPrecisionError(ValueError):
    code = "execution_precision_contract"


def require_execution_precision() -> dict[str, object]:
    """Refuse unexpected defaults, TF32, overrides, and inconsistent backend APIs."""
    try:
        observed: dict[str, object] = {
            "default_dtype": str(torch.get_default_dtype()),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "generic_fp32_precision": cast(_PrecisionBackend, torch.backends).fp32_precision,
            "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "effective_tf32": torch.backends.cuda.matmul.allow_tf32,
            "torch_tf32_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        }
    except (RuntimeError, AttributeError) as error:
        raise ExecutionPrecisionError(
            "execution precision controls are unavailable or inconsistent"
        ) from error
    if (
        observed["default_dtype"] != "torch.float32"
        or observed["matmul_precision"] != "highest"
        or observed["effective_tf32"] is not False
        or observed["generic_fp32_precision"] not in ("none", "ieee")
        or observed["cuda_matmul_fp32_precision"] not in ("none", "ieee")
        or observed["torch_tf32_override"] not in (None, "0")
        or observed["nvidia_tf32_override"] not in (None, "0")
    ):
        raise ExecutionPrecisionError(f"execution precision contract violated: {observed}")
    return observed


def require_observed_execution_precision(profile: HProfile) -> dict[str, object]:
    """The guarded record plus the declared precision and the autocast torch reports for it.

    The refusal is the unchanged guard and runs first, so a refused regime never enters an
    autocast context. The probe then opens the exact context every execution site opens
    (``torch.autocast(device_type, dtype=torch.bfloat16, enabled=profile.precision == "bf16")``)
    and reads torch's own state inside it. The context manager restores the ambient state on
    exit; a torch error (unsupported device, CUDA without bf16) propagates with torch's text and
    is never recorded as a disabled autocast.

    MEASURED 2026-10-22 on torch 2.13.0 (CPU host): bf16 requested reads (True, torch.bfloat16);
    enabled=False reads (False, torch.bfloat16). ``autocast.__enter__`` stores the requested
    dtype unconditionally, so the dtype echoes the request even when autocast is off and
    ``enabled`` is the only independent observation. The pair is recorded as read.
    """
    record = require_execution_precision()
    device_type = "cuda" if profile.device == "cuda" else "cpu"
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=profile.precision == "bf16"):
        enabled = bool(torch.is_autocast_enabled(device_type))
        dtype = str(torch.get_autocast_dtype(device_type))
    return {
        **record,
        "declared_precision": profile.precision,
        "autocast": {"enabled": enabled, "dtype": dtype, "device_type": device_type},
    }
