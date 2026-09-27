"""Future-target value types and causal horizon-one construction for D1."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from entropic_substrate.core.latent.state import LatentTrajectory, TargetEncoder
from entropic_substrate.core.observation.contracts import (
    CodecIdentity,
    EncodedObservations,
    ObservationAlignment,
)

__all__ = [
    "FutureTarget",
    "StabilizedTargets",
    "one_step_displacement",
    "one_step_target_from_context",
]


def _validate_target_values(values: torch.Tensor, *, owner: str) -> None:
    if type(values) is not torch.Tensor:
        msg = f"{owner} values must be a torch.Tensor"
        raise TypeError(msg)
    if values.ndim != 3:
        msg = f"{owner} values must be rank-3, got shape {tuple(values.shape)}"
        raise ValueError(msg)
    if not values.is_floating_point():
        msg = f"{owner} values must have floating dtype, got {values.dtype}"
        raise TypeError(msg)
    if any(size < 1 for size in values.shape):
        msg = f"{owner} values require positive dimensions, got shape {tuple(values.shape)}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class FutureTarget:
    """A codec-independent latent displacement target for the future law."""

    values: torch.Tensor
    codec_identity: CodecIdentity

    def __post_init__(self) -> None:
        _validate_target_values(self.values, owner="future target")
        if type(self.codec_identity) is not CodecIdentity:
            msg = "future target requires a CodecIdentity"
            raise TypeError(msg)

    @property
    def width(self) -> int:
        """Target width, matched to the causal-state width rather than a vocabulary."""
        return int(self.values.shape[-1])


@dataclass(frozen=True, slots=True)
class StabilizedTargets:
    """Future targets after the selected target-update strategy boundary."""

    values: torch.Tensor
    codec_identity: CodecIdentity
    detached: bool

    def __post_init__(self) -> None:
        _validate_target_values(self.values, owner="stabilized target")
        if type(self.codec_identity) is not CodecIdentity:
            msg = "stabilized target requires a CodecIdentity"
            raise TypeError(msg)
        if type(self.detached) is not bool:
            msg = "stabilized target detached flag must be bool"
            raise TypeError(msg)
        if self.detached and self.values.requires_grad:
            msg = "a detached stabilized target must not require gradients"
            raise ValueError(msg)

    @property
    def width(self) -> int:
        """Target width after stabilization."""
        return int(self.values.shape[-1])


def one_step_displacement(target_states: LatentTrajectory) -> FutureTarget:
    """Return the exact one-step target ``Y_t = xi[t+1] - xi[t]``."""
    if type(target_states) is not LatentTrajectory:
        msg = "one-step target construction requires a LatentTrajectory"
        raise TypeError(msg)
    if target_states.time < 2:
        msg = "one-step target construction requires at least two target-context positions"
        raise ValueError(msg)
    with torch.autocast(device_type=target_states.values.device.type, enabled=False):
        states32 = target_states.values.float()
        displacement = states32[:, 1:] - states32[:, :-1]
    return FutureTarget(values=displacement, codec_identity=target_states.codec_identity)


def one_step_target_from_context(
    target_encoder: TargetEncoder,
    target_context: EncodedObservations,
    alignment: ObservationAlignment,
) -> FutureTarget:
    """Encode the legal D1 ``T+1`` context and return its one-step target."""
    if type(target_encoder) is not TargetEncoder:
        msg = "D1 target construction requires a TargetEncoder"
        raise TypeError(msg)
    if type(target_context) is not EncodedObservations:
        msg = "D1 target construction requires EncodedObservations"
        raise TypeError(msg)
    if type(alignment) is not ObservationAlignment:
        msg = "D1 target construction requires an ObservationAlignment"
        raise TypeError(msg)
    if alignment.horizon != 1:
        msg = f"D1 one-step target construction requires horizon 1, got {alignment.horizon}"
        raise ValueError(msg)
    expected_context = alignment.target_encoder_context.length
    if target_context.values.shape[1] != expected_context:
        msg = (
            f"target context length {target_context.values.shape[1]} does not match "
            f"alignment target context length {expected_context}"
        )
        raise ValueError(msg)
    future = one_step_displacement(target_encoder(target_context))
    expected_time = alignment.online_observations.length
    if future.values.shape[1] != expected_time:
        msg = (
            f"one-step target time {future.values.shape[1]} does not match online "
            f"sequence length {expected_time}"
        )
        raise RuntimeError(msg)
    return future
