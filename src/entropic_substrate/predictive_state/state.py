"""Causal latent-state contracts and encoder for D1.

The online state is deliberately a fixed-width latent trajectory.  This
module knows about codec *identity* for provenance, but never about an
alphabet-sized probability vector or language readout.  Future targets and
their predictor are separate feature owners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from entropic_substrate.config import ModelConfig
from entropic_substrate.models.blocks import build_block
from entropic_substrate.observation import CodecIdentity, EncodedObservations

__all__ = [
    "AffineFreeRMSNorm",
    "CausalStateEncoder",
    "CausalStateEncoderPort",
    "EncoderPair",
    "LatentState",
    "LatentTrajectory",
    "TargetEncoder",
    "build_encoder_pair",
]


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        msg = f"{name} must be a positive integer, got {value!r}"
        raise ValueError(msg)
    return value


def _validate_latent_values(*, values: torch.Tensor, rank: int, owner: str) -> None:
    if type(values) is not torch.Tensor:
        msg = f"{owner} values must be a torch.Tensor"
        raise TypeError(msg)
    if values.ndim != rank:
        msg = f"{owner} values must be rank-{rank}, got shape {tuple(values.shape)}"
        raise ValueError(msg)
    if not values.is_floating_point():
        msg = f"{owner} values must have floating dtype, got {values.dtype}"
        raise TypeError(msg)
    if any(size < 1 for size in values.shape):
        msg = f"{owner} values require only positive dimensions, got shape {tuple(values.shape)}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LatentState:
    """One batch of latent states at a single source position."""

    values: torch.Tensor
    codec_identity: CodecIdentity

    def __post_init__(self) -> None:
        _validate_latent_values(values=self.values, rank=2, owner="latent state")
        if type(self.codec_identity) is not CodecIdentity:
            msg = "latent state requires a CodecIdentity"
            raise TypeError(msg)

    @property
    def width(self) -> int:
        """Fixed semantic width, independent of codec cardinality."""
        return int(self.values.shape[-1])


@dataclass(frozen=True, slots=True)
class LatentTrajectory:
    """Causal ``(batch, time, width)`` latent states with codec provenance."""

    values: torch.Tensor
    codec_identity: CodecIdentity

    def __post_init__(self) -> None:
        _validate_latent_values(values=self.values, rank=3, owner="latent trajectory")
        if type(self.codec_identity) is not CodecIdentity:
            msg = "latent trajectory requires a CodecIdentity"
            raise TypeError(msg)

    @property
    def width(self) -> int:
        """Latent width; never inferred from the observation cardinality."""
        return int(self.values.shape[-1])

    @property
    def time(self) -> int:
        """Number of causally ordered source positions."""
        return int(self.values.shape[1])


class AffineFreeRMSNorm(nn.Module):
    """Final RMS normalizer without affine parameters or mean centering.

    The reduction is always calculated in FP32 inside an explicit autocast
    island.  The returned tensor preserves the input dtype so the encoder can
    participate in a mixed-precision training path without moving its output
    contract to ambient FP32.
    """

    def __init__(self, *, width: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.width = _positive_int("width", width)
        if not isinstance(eps, float) or not torch.isfinite(torch.tensor(eps)) or eps <= 0.0:
            msg = f"eps must be a finite positive float, got {eps!r}"
            raise ValueError(msg)
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize the final latent axis using a full-precision RMS."""
        _validate_latent_values(values=values, rank=3, owner="RMS-normalizer input")
        if values.shape[-1] != self.width:
            msg = f"RMS-normalizer width {values.shape[-1]} does not match configured width {self.width}"
            raise ValueError(msg)
        with torch.autocast(device_type=values.device.type, enabled=False):
            values32 = values.float()
            inv_rms = values32.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            normalized = values32 * inv_rms
        return normalized.to(values.dtype)


@runtime_checkable
class CausalStateEncoderPort(Protocol):
    """Codec-bound, future-blind source of a fixed-width latent trajectory."""

    latent_width: int

    def forward(self, observations: EncodedObservations) -> LatentTrajectory: ...


class CausalStateEncoder(nn.Module):
    """Compose existing causal blocks into a future-blind latent encoder.

    ``EncodedObservations`` supplies already-adapted boundary vectors.  The
    module intentionally has no observation table, language classifier, or
    future-law decoder, so none of those modules can make the state depend on
    their weights or outputs.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if type(cfg) is not ModelConfig:
            msg = f"causal state encoder requires ModelConfig, got {type(cfg).__name__}"
            raise TypeError(msg)
        self.latent_width = _positive_int("cfg.d_model", cfg.d_model)
        self.blocks = nn.ModuleList([build_block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = AffineFreeRMSNorm(width=self.latent_width)

    def forward(self, observations: EncodedObservations) -> LatentTrajectory:
        """Emit causal states with the same width as the adapted observations."""
        if type(observations) is not EncodedObservations:
            msg = "causal state encoder requires EncodedObservations"
            raise TypeError(msg)
        if observations.latent_width != self.latent_width:
            msg = (
                f"encoded observation latent width {observations.latent_width} does not match "
                f"causal state encoder width {self.latent_width}"
            )
            raise ValueError(msg)
        states = observations.values
        for block in self.blocks:
            states = block(states)
        return LatentTrajectory(
            values=self.final_norm(states), codec_identity=observations.codec_identity
        )


class TargetEncoder(CausalStateEncoder):
    """A separately-owned causal encoder used only for target construction."""


@dataclass(frozen=True, slots=True)
class EncoderPair:
    """Online and target encoders with matched construction, never storage."""

    online: CausalStateEncoder
    target: TargetEncoder

    def __post_init__(self) -> None:
        online = dict(self.online.named_parameters())
        target = dict(self.target.named_parameters())
        if online.keys() != target.keys():
            msg = "online and target encoders must expose the same parameter-name map"
            raise ValueError(msg)
        for name, online_parameter in online.items():
            target_parameter = target[name]
            if online_parameter.shape != target_parameter.shape:
                msg = f"online/target parameter shape mismatch for {name!r}"
                raise ValueError(msg)
            if online_parameter.dtype != target_parameter.dtype:
                msg = f"online/target parameter dtype mismatch for {name!r}"
                raise ValueError(msg)
            if online_parameter.data_ptr() == target_parameter.data_ptr():
                msg = f"online and target encoders share parameter storage for {name!r}"
                raise ValueError(msg)


def build_encoder_pair(cfg: ModelConfig) -> EncoderPair:
    """Build independent encoders with bit-identical initial parameter values."""
    online = CausalStateEncoder(cfg)
    target = TargetEncoder(cfg)
    # ``load_state_dict`` copies into target-owned tensors; it does not alias
    # module storage, which is checked by ``EncoderPair`` above.
    target.load_state_dict(online.state_dict(), strict=True)
    return EncoderPair(online=online, target=target)
