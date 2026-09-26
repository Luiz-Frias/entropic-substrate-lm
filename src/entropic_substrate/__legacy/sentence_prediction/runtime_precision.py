"""Read-only precision admission for the registered FP32 geometric computation.

PyTorch documents a TF32 environment override independent of the normal setting:
https://docs.pytorch.org/docs/main/cuda_environment_variables.html
This guard never repairs ambient settings or changes another model's runtime.
"""

import os
from typing import Protocol, cast

import torch


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
