"""Identity byte codec, the current reversible external observation adapter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar

import torch

from entropic_substrate.observation.contracts import CodecIdentity

__all__ = ["ByteObservationCodec"]


@dataclass(frozen=True, slots=True)
class ByteObservationCodec:
    """Map each source byte to its identical ``0..255`` observation ID.

    This class is an observation boundary adapter, not a semantic state
    representation.  Its cardinality is intentionally visible here and absent
    from :class:`~entropic_substrate.observation.EncodedObservations`.
    """

    construction_label: str = "identity-byte-v1"

    cardinality: ClassVar[int] = 256
    _NAME: ClassVar[str] = "identity-byte"

    def __post_init__(self) -> None:
        if not self.construction_label:
            msg = "byte codec construction_label must be non-empty"
            raise ValueError(msg)

    @property
    def construction_digest(self) -> str:
        """Stable SHA-256 over the codec's explicit construction recipe."""
        payload = json.dumps(
            {
                "cardinality": self.cardinality,
                "construction_label": self.construction_label,
                "codec": self._NAME,
                "schema_version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def codec_identity(self) -> CodecIdentity:
        """Provenance identity attached to latent encoded observations."""
        return CodecIdentity(
            name=self._NAME,
            construction_digest=self.construction_digest,
            cardinality=self.cardinality,
        )

    @property
    def identity(self) -> CodecIdentity:
        """Alias used by consumers that speak in boundary identities."""
        return self.codec_identity

    def encode(self, source: bytes) -> torch.Tensor:
        """Encode a byte string as a rank-1 int64 tensor without relabeling."""
        if type(source) is not bytes:
            msg = f"byte codec expects bytes, got {type(source).__name__}"
            raise TypeError(msg)
        return torch.tensor(list(source), dtype=torch.int64)

    def decode(self, observation_ids: torch.Tensor) -> bytes:
        """Invert :meth:`encode`, refusing non-byte-valued observation IDs."""
        if type(observation_ids) is not torch.Tensor:
            msg = "byte codec observation_ids must be a torch.Tensor"
            raise TypeError(msg)
        if observation_ids.ndim != 1:
            msg = f"byte codec observation_ids must be rank-1, got {observation_ids.ndim}"
            raise ValueError(msg)
        if observation_ids.dtype != torch.int64:
            msg = f"byte codec observation_ids must be int64, got {observation_ids.dtype}"
            raise TypeError(msg)
        ids = observation_ids.detach().to(device="cpu")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.cardinality):
            msg = "byte codec observation_ids must be in [0, 255]"
            raise ValueError(msg)
        return bytes(ids.tolist())

    def encode_bytes(self, source: bytes) -> torch.Tensor:
        """Explicit-name alias for :meth:`encode`."""
        return self.encode(source)

    def decode_ids(self, observation_ids: torch.Tensor) -> bytes:
        """Explicit-name alias for :meth:`decode`."""
        return self.decode(observation_ids)
