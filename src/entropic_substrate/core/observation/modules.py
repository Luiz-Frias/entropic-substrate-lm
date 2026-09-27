"""Learned observation adapter and language readout boundary modules."""

from __future__ import annotations

import torch
from torch import nn

from entropic_substrate.core.observation.contracts import CodecIdentity, EncodedObservations

__all__ = ["LanguageReadout", "LearnedObservationAdapter"]


def _positive_width(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        msg = f"{name} must be a positive integer, got {value!r}"
        raise ValueError(msg)
    return value


class LearnedObservationAdapter(nn.Module):
    """Map boundary IDs to latent vectors without defining the latent width.

    ``vocab_size`` is a property of the input codec adapter.  ``latent_width``
    belongs to the causal-state domain and may be held fixed across codecs.
    """

    def __init__(self, *, vocab_size: int, latent_width: int) -> None:
        super().__init__()
        self.vocab_size = _positive_width("vocab_size", vocab_size)
        self.latent_width = _positive_width("latent_width", latent_width)
        self.embedding = nn.Embedding(self.vocab_size, self.latent_width)

    def forward(
        self, observation_ids: torch.Tensor, codec_identity: CodecIdentity
    ) -> EncodedObservations:
        if type(observation_ids) is not torch.Tensor:
            msg = "observation_ids must be a torch.Tensor"
            raise TypeError(msg)
        if observation_ids.ndim != 2:
            msg = (
                "observation_ids must be rank-2 (batch, time), "
                f"got shape {tuple(observation_ids.shape)}"
            )
            raise ValueError(msg)
        if observation_ids.dtype != torch.int64:
            msg = f"observation_ids must be int64, got {observation_ids.dtype}"
            raise TypeError(msg)
        if any(size <= 0 for size in observation_ids.shape):
            msg = f"observation_ids require positive batch and time, got {tuple(observation_ids.shape)}"
            raise ValueError(msg)
        if type(codec_identity) is not CodecIdentity:
            msg = "observation_ids require a CodecIdentity"
            raise TypeError(msg)
        if codec_identity.cardinality != self.vocab_size:
            msg = (
                f"codec identity cardinality {codec_identity.cardinality} does not match "
                f"adapter vocab_size {self.vocab_size}"
            )
            raise ValueError(msg)
        if int(observation_ids.min()) < 0 or int(observation_ids.max()) >= self.vocab_size:
            msg = f"observation_ids must be in [0, {self.vocab_size - 1}]"
            raise ValueError(msg)
        return EncodedObservations(
            values=self.embedding(observation_ids),
            codec_identity=codec_identity,
        )


class LanguageReadout(nn.Module):
    """Untied latent-to-observation decoder for the measured language task."""

    def __init__(self, *, latent_width: int, vocab_size: int) -> None:
        super().__init__()
        self.latent_width = _positive_width("latent_width", latent_width)
        self.vocab_size = _positive_width("vocab_size", vocab_size)
        self.projection = nn.Linear(self.latent_width, self.vocab_size, bias=False)

    def forward(self, latent_states: torch.Tensor) -> torch.Tensor:
        if type(latent_states) is not torch.Tensor:
            msg = "latent_states must be a torch.Tensor"
            raise TypeError(msg)
        if latent_states.ndim != 3:
            msg = (
                "latent_states must be rank-3 (batch, time, latent_width), "
                f"got shape {tuple(latent_states.shape)}"
            )
            raise ValueError(msg)
        if not latent_states.is_floating_point():
            msg = f"latent_states must have floating dtype, got {latent_states.dtype}"
            raise TypeError(msg)
        if any(size <= 0 for size in latent_states.shape[:2]):
            msg = (
                "latent_states require positive batch and time axes, "
                f"got shape {tuple(latent_states.shape)}"
            )
            raise ValueError(msg)
        if latent_states.shape[-1] != self.latent_width:
            msg = (
                f"latent_states width {latent_states.shape[-1]} does not match "
                f"readout latent_width {self.latent_width}"
            )
            raise ValueError(msg)
        return self.projection(latent_states)
