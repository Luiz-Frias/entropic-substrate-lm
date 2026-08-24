"""D1 observation boundary: codecs, source provenance, and readout modules."""

from __future__ import annotations

from entropic_substrate.observation.byte_codec import ByteObservationCodec
from entropic_substrate.observation.contracts import (
    CodecIdentity,
    EncodedObservations,
    ObservationAlignment,
    ObservationCodecPort,
    SourceSpan,
)
from entropic_substrate.observation.modules import LanguageReadout, LearnedObservationAdapter

__all__ = [
    "ByteObservationCodec",
    "CodecIdentity",
    "EncodedObservations",
    "LanguageReadout",
    "LearnedObservationAdapter",
    "ObservationAlignment",
    "ObservationCodecPort",
    "SourceSpan",
]
