"""Pure value contracts for the D1 observation boundary.

The codec owns alphabet cardinality at the external boundary.  Once
observations have been adapted, their ``(batch, time, latent_width)`` values
carry only the codec identity required to prove provenance; no alphabet-sized
state is implied by this type.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

__all__ = [
    "CodecIdentity",
    "EncodedObservations",
    "ObservationAlignment",
    "ObservationCodecPort",
    "SourceSpan",
    "validate_observation_ids",
]


@dataclass(frozen=True, slots=True)
class CodecIdentity:
    """Stable, serialisable identity of an observation codec construction."""

    name: str
    construction_digest: str
    cardinality: int

    def __post_init__(self) -> None:
        if not self.name:
            msg = "codec identity name must be non-empty"
            raise ValueError(msg)
        if len(self.construction_digest) != 64 or any(
            char not in string.hexdigits for char in self.construction_digest
        ):
            msg = "codec construction_digest must be a 64-character hexadecimal SHA-256 digest"
            raise ValueError(msg)
        if type(self.cardinality) is not int or self.cardinality < 1:
            msg = f"codec cardinality must be a positive integer, got {self.cardinality!r}"
            raise ValueError(msg)

    @property
    def digest(self) -> str:
        """Compatibility-friendly short name for the construction digest."""
        return self.construction_digest


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """One non-empty half-open interval in the source observation stream."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.stop) is not int:
            msg = "source span bounds must be integers, not booleans"
            raise TypeError(msg)
        if self.start < 0:
            msg = f"source span start must be nonnegative, got {self.start}"
            raise ValueError(msg)
        if self.stop <= self.start:
            msg = f"source span must be non-empty, got [{self.start}, {self.stop})"
            raise ValueError(msg)

    @property
    def length(self) -> int:
        """Number of source observations in the interval."""
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class ObservationAlignment:
    """Causal/target source alignment for one horizon-aware example.

    ``online_observations`` feeds the causal online encoder.  The overlapping
    ``language_targets`` is the ordinary next-observation supervision.  The
    target encoder receives the longer ``target_encoder_context`` only to
    construct a future target; it is never an online-encoder input.
    """

    online_observations: SourceSpan
    language_targets: SourceSpan
    target_encoder_context: SourceSpan
    horizon: int

    def __post_init__(self) -> None:
        if type(self.horizon) is not int:
            msg = "horizon must be an integer"
            raise TypeError(msg)
        if self.horizon < 1:
            msg = f"horizon must be at least 1, got {self.horizon}"
            raise ValueError(msg)
        online = self.online_observations
        language = self.language_targets
        context = self.target_encoder_context
        if language != SourceSpan(online.start + 1, online.stop + 1):
            msg = "language target span must be the online span shifted forward by one"
            raise ValueError(msg)
        if context != SourceSpan(online.start, online.stop + self.horizon):
            msg = "target encoder context must extend the online span by exactly horizon"
            raise ValueError(msg)

    @classmethod
    def predictive(cls, *, start: int, sequence_length: int, horizon: int) -> ObservationAlignment:
        """Construct the one legal alignment for a sampled source offset."""
        if type(sequence_length) is not int:
            msg = "sequence_length must be an integer"
            raise TypeError(msg)
        if sequence_length < 1:
            msg = f"sequence_length must be positive, got {sequence_length}"
            raise ValueError(msg)
        online = SourceSpan(start, start + sequence_length)
        return cls(
            online_observations=online,
            language_targets=SourceSpan(start + 1, start + sequence_length + 1),
            target_encoder_context=SourceSpan(start, start + sequence_length + horizon),
            horizon=horizon,
        )

    @property
    def future_observations(self) -> SourceSpan:
        """The strictly future suffix available only to target construction."""
        return SourceSpan(self.online_observations.stop, self.target_encoder_context.stop)


@dataclass(frozen=True, slots=True)
class EncodedObservations:
    """Floating latent inputs plus the codec identity that produced them."""

    values: torch.Tensor
    codec_identity: CodecIdentity

    def __post_init__(self) -> None:
        if type(self.values) is not torch.Tensor:
            msg = "encoded observation values must be a torch.Tensor"
            raise TypeError(msg)
        if type(self.codec_identity) is not CodecIdentity:
            msg = "encoded observations require a CodecIdentity"
            raise TypeError(msg)
        if self.values.ndim != 3:
            msg = (
                "encoded observation values must be rank-3 "
                "(batch, time, latent_width), "
                f"got shape {tuple(self.values.shape)}"
            )
            raise ValueError(msg)
        if not self.values.is_floating_point():
            msg = f"encoded observation values must have floating dtype, got {self.values.dtype}"
            raise TypeError(msg)
        if any(size <= 0 for size in self.values.shape):
            msg = (
                "encoded observation values require positive batch, time, and latent_width, "
                f"got shape {tuple(self.values.shape)}"
            )
            raise ValueError(msg)

    @property
    def latent_width(self) -> int:
        """Latent width, deliberately independent of the codec cardinality."""
        return int(self.values.shape[-1])


@runtime_checkable
class ObservationCodecPort(Protocol):
    """External observation adapter contract; alphabet size does not cross it."""

    @property
    def cardinality(self) -> int: ...

    @property
    def codec_identity(self) -> CodecIdentity: ...

    def encode(self, source: bytes) -> torch.Tensor: ...

    def decode(self, observation_ids: torch.Tensor) -> bytes: ...


def validate_observation_ids(
    observation_ids: torch.Tensor, *, owner: str, cardinality: int
) -> torch.Tensor:
    """Check one decode input and return it detached on the CPU.

    The decode-side checks every byte codec shares, in their fixed order: a
    real tensor, rank 1, int64, then every value inside ``[0, cardinality)``.
    ``owner`` names the codec in the refusal text.  Empty input is valid.
    Each codec's one-line encode guard stays inline in that codec.
    """
    if type(observation_ids) is not torch.Tensor:
        msg = f"{owner} observation_ids must be a torch.Tensor"
        raise TypeError(msg)
    if observation_ids.ndim != 1:
        msg = f"{owner} observation_ids must be rank-1, got {observation_ids.ndim}"
        raise ValueError(msg)
    if observation_ids.dtype != torch.int64:
        msg = f"{owner} observation_ids must be int64, got {observation_ids.dtype}"
        raise TypeError(msg)
    ids = observation_ids.detach().to(device="cpu")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= cardinality):
        msg = f"{owner} observation_ids must be in range [0, {cardinality - 1}]"
        raise ValueError(msg)
    return ids
