"""Stable latent views, keeping carrier values separate from their coordinates."""

from typing import Literal, NamedTuple, cast

import torch

from entropic_substrate.__legacy.predictive_state.metric import FisherNumericalError, finite_tensor

type InterfaceMode = Literal["sphere", "magnitude"]


class UndefinedDirection(ValueError):
    """A measured zero-norm gap; scientific clauses decide whether it is collapse."""

    code = "undefined_zero_norm"

    def __init__(self, positions: int) -> None:
        self.positions = positions
        super().__init__(f"{self.code}: {positions} latent positions have no direction")


class Coordinates(NamedTuple):
    direction: torch.Tensor
    log_radius: torch.Tensor
    zero_mask: torch.Tensor


def _layout(values: torch.Tensor) -> None:
    if not isinstance(cast(object, values), torch.Tensor):
        raise TypeError("latent values must be a tensor")
    if values.ndim < 1 or values.shape[-1] < 2 or any(size < 1 for size in values.shape):
        raise FisherNumericalError("invalid_shape", "latent coordinates require nonempty vectors")
    if not values.is_floating_point():
        raise FisherNumericalError("invalid_dtype", "latent coordinates require floating values")


def coordinate_map(values: torch.Tensor) -> Coordinates:
    """Functional map for JVP/VJP: FP32 in production, FP64 for tiny oracles.

    Scale before squaring, so finite tiny/huge vectors do not vanish/overflow
    during the norm calculation. There is no positive-value floor. At exactly
    zero, direction is a zero view (not a unit point), log-radius is -infinity,
    and the mask is explicit. Validation of data-dependent finiteness and zero
    counts belongs outside vmap, in checked_coordinates or its owning caller.
    This function never overwrites the input carrier.
    """
    _layout(values)
    with torch.autocast(device_type=values.device.type, enabled=False):
        dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        promoted = values.to(dtype)
        scale = promoted.abs().amax(dim=-1, keepdim=True)
        zero = scale == 0
        safe_scale = torch.where(zero, torch.ones_like(scale), scale)
        scaled = promoted / safe_scale
        squared = scaled.square().sum(dim=-1, keepdim=True)
        safe_squared = torch.where(zero, torch.ones_like(squared), squared)
        direction = scaled / safe_squared.sqrt()
        log_radius = safe_scale.log() + 0.5 * safe_squared.log()
        log_radius = torch.where(zero, torch.full_like(log_radius, -torch.inf), log_radius)
        return Coordinates(direction, log_radius.squeeze(-1), zero.squeeze(-1))


def checked_coordinates(values: torch.Tensor, *, allow_zero: bool = False) -> Coordinates:
    """Validate a batch outside functional transforms and retain explicit gaps."""
    _layout(values)
    if type(allow_zero) is not bool:
        raise TypeError("allow_zero must be an explicit boolean")
    finite_tensor(values, "latent_values")
    coordinates = coordinate_map(values)
    if not allow_zero and bool(coordinates.zero_mask.any()):
        raise UndefinedDirection(int(coordinates.zero_mask.sum()))
    finite_tensor(coordinates.direction, "latent_direction")
    return coordinates


def apply_interface(values: torch.Tensor, mode: InterfaceMode) -> torch.Tensor:
    """Select an independent sentence interface, never an implicit default.

    Magnitude mode preserves the original FP32/FP64 tensor and permits measured
    zero states. Low-precision values are promoted, not renormalized. Sphere
    mode requires a defined direction; zero does not become a fabricated point.
    """
    if mode not in ("sphere", "magnitude"):
        raise ValueError("unknown sentence interface")
    _layout(values)
    finite_tensor(values, "latent_values")
    if mode == "sphere":
        return checked_coordinates(values).direction
    with torch.autocast(device_type=values.device.type, enabled=False):
        dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        return values.to(dtype)
