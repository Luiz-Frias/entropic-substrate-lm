"""Matrix-free local Fisher pullback for a unit-covariance Gaussian law.

The caller supplies a pointwise mean map, including any normalization, from
one latent vector to one predictive mean. No graph or metric is cached.
"""

from collections.abc import Callable

import torch

type MeanMap = Callable[[torch.Tensor], torch.Tensor]


class FisherNumericalError(ValueError):
    """A numerical defect with a machine-readable code and operation context."""

    def __init__(self, code: str, operation: str) -> None:
        self.code = code
        self.operation = operation
        super().__init__(f"{code}: {operation}")


def finite_tensor(values: torch.Tensor, operation: str) -> None:
    """Reject nonfinite measured arithmetic without substituting a value."""
    if not bool(torch.isfinite(values).all()):
        raise FisherNumericalError(f"nonfinite_{operation}", operation)


def numerical_pair(states: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and promote numerical inputs, retaining float64 for oracles."""
    if states.ndim < 1 or any(size < 1 for size in states.shape):
        raise FisherNumericalError("invalid_shape", "states require nonempty latent vectors")
    if states.shape != vector.shape or states.device != vector.device:
        raise FisherNumericalError("shape_mismatch", "states and vector must align")
    if not states.is_floating_point() or not vector.is_floating_point():
        raise FisherNumericalError("invalid_dtype", "states and vector must be floating")
    dtype = torch.float64 if torch.float64 in (states.dtype, vector.dtype) else torch.float32
    states, vector = states.to(dtype), vector.to(dtype)
    finite_tensor(states, "states")
    finite_tensor(vector, "vector")
    return states, vector


def fisher_action(mean_map: MeanMap, states: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Return ``J(mean)^T J(mean) vector`` independently at each latent state.

    Leading dimensions are independent positions. ``mean_map`` accepts a
    single ``[D]`` vector and returns ``[K]``; this boundary prevents accidental
    differentiation through neighboring context positions. Closure parameters
    remain differentiable. Production uses FP32 outside ambient autocast;
    float64 inputs retain precision for derivative/oracle tests.
    """
    states, vector = numerical_pair(states, vector)

    def local_action(
        state: torch.Tensor, tangent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reverse_result = torch.func.vjp(mean_map, state)
        mean, pullback = reverse_result[0], reverse_result[1]
        if mean.ndim != 1 or mean.numel() == 0:
            raise FisherNumericalError("invalid_mean_shape", "mean must be a nonempty vector")
        pushed = torch.func.jvp(mean_map, (state,), (tangent,))[1]
        return mean, pullback(pushed)[0]

    with torch.autocast(device_type=states.device.type, enabled=False):
        means, actions = torch.vmap(local_action)(
            states.reshape(-1, states.shape[-1]), vector.reshape(-1, vector.shape[-1])
        )
        finite_tensor(means, "mean")
        finite_tensor(actions, "metric_action")
        return actions.reshape_as(states)
