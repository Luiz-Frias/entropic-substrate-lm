"""Deterministic bijective byte relabeling for D5 codec controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar

import torch

from entropic_substrate.core.observation.contracts import (
    CodecIdentity,
    validate_observation_ids,
)


@dataclass(frozen=True, slots=True)
class PermutedByteObservationCodec:
    """Map canonical source byte ``b`` to ``mapping[b]`` at the boundary only.

    Explicit mapping bytes determine identity, independently of RNG libraries.
    Source offsets and source bits are unchanged. The semantic core receives
    the same latent width as the identity-codec construction.
    """

    mapping: tuple[int, ...]
    construction_label: str = "permuted-byte-v1"

    cardinality: ClassVar[int] = 256

    def __post_init__(self) -> None:
        if (
            type(self.mapping) is not tuple
            or len(self.mapping) != self.cardinality
            or any(type(value) is not int for value in self.mapping)
            or set(self.mapping) != set(range(self.cardinality))
        ):
            msg = "byte mapping must be a bijection of the integers 0 through 255"
            raise ValueError(msg)
        if type(self.construction_label) is not str or not self.construction_label.strip():
            msg = "permuted byte construction_label must be a nonblank string"
            raise ValueError(msg)

    @classmethod
    def from_seed(cls, seed: int) -> PermutedByteObservationCodec:
        """Generate a reproducible mapping by SHA-256 ordering, independent of torch RNG."""
        if type(seed) is not int or seed < 0:
            msg = "permutation seed must be a nonnegative integer"
            raise ValueError(msg)
        mapping = tuple(
            sorted(
                range(256),
                key=lambda byte: (
                    hashlib.sha256(f"byte-permutation-v1:{seed}:{byte}".encode()).digest(),
                    byte,
                ),
            )
        )
        return cls(mapping=mapping)

    @classmethod
    def from_mapping(
        cls,
        *,
        mapping: tuple[int, ...],
        expected_digest: str,
        construction_label: str = "permuted-byte-v1",
    ) -> PermutedByteObservationCodec:
        """Restore persisted construction and reject an identity/mapping mismatch."""
        codec = cls(mapping=mapping, construction_label=construction_label)
        if codec.construction_digest != expected_digest:
            msg = "permutation construction digest does not match its explicit mapping"
            raise ValueError(msg)
        return codec

    @property
    def construction_digest(self) -> str:
        """SHA-256 over the versioned mapping and construction label."""
        payload = json.dumps(
            {
                "cardinality": self.cardinality,
                "codec": "permuted-byte",
                "construction_label": self.construction_label,
                "mapping": self.mapping,
                "schema_version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def codec_identity(self) -> CodecIdentity:
        """Boundary identity carried through states and captured readouts."""
        return CodecIdentity("permuted-byte", self.construction_digest, self.cardinality)

    @property
    def identity(self) -> CodecIdentity:
        """Boundary identity alias matching the identity byte codec."""
        return self.codec_identity

    @property
    def inverse_mapping(self) -> tuple[int, ...]:
        """Map encoded observation labels back to canonical source bytes."""
        inverse = [0] * self.cardinality
        for canonical, encoded in enumerate(self.mapping):
            inverse[encoded] = canonical
        return tuple(inverse)

    def encode(self, source: bytes) -> torch.Tensor:
        """Encode one source byte per observation without modifying source length."""
        if type(source) is not bytes:
            msg = "permuted byte codec expects bytes"
            raise TypeError(msg)
        return torch.tensor([self.mapping[byte] for byte in source], dtype=torch.int64)

    def decode(self, observation_ids: torch.Tensor) -> bytes:
        """Invert an integer observation sequence exactly, including empty input."""
        values = validate_observation_ids(
            observation_ids, owner="permuted byte", cardinality=self.cardinality
        )
        inverse = self.inverse_mapping
        return bytes(inverse[index] for index in values.tolist())

    def canonical_logits(
        self, logits: torch.Tensor, *, codec_identity: CodecIdentity
    ) -> torch.Tensor:
        """Reorder encoded-label logits into canonical byte order without detaching."""
        if codec_identity != self.codec_identity:
            msg = "readout codec identity does not match the permutation"
            raise ValueError(msg)
        if logits.ndim < 1 or logits.shape[-1] != self.cardinality:
            msg = "permuted readout requires 256 logits on its final axis"
            raise ValueError(msg)
        if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
            msg = "permuted readout logits must be finite floating values"
            raise ValueError(msg)
        indices = torch.tensor(self.mapping, dtype=torch.int64, device=logits.device)
        return logits.index_select(-1, indices)

    def permute_boundary_rows(self, canonical_rows: torch.Tensor) -> torch.Tensor:
        """Return relabeled observation/readout rows for an exactly coupled control.

        Apply to each input table, output weight, and optional output bias.
        The caller owns parameter installation and weight tying; this function
        preserves autograd and never mutates an already-trained model.
        """
        if canonical_rows.ndim < 1 or canonical_rows.shape[0] != self.cardinality:
            msg = "boundary tensor requires exactly 256 rows"
            raise ValueError(msg)
        indices = torch.tensor(
            self.inverse_mapping, dtype=torch.int64, device=canonical_rows.device
        )
        return canonical_rows.index_select(0, indices)
