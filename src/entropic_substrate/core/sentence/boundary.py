"""Reversible sentence/document envelope, distinct from the E/D byte codec."""

from dataclasses import dataclass, field
from typing import ClassVar

from entropic_substrate.core.sentence.records import content_identity


@dataclass(frozen=True, slots=True)
class SentenceByteCodec:
    """Bytes remain 256 boundary symbols, never a semantic latent width.

    Sentence-end separates spans; document-end terminates the last span.
    BOS and PAD are input-only symbols, not outcomes in the 258-way readout.
    The optional permutation acts on bytes alone, leaving framing unchanged.
    """

    permutation: tuple[int, ...] = tuple(range(256))
    inverse: tuple[int, ...] = field(init=False, repr=False)
    identity: str = field(init=False)

    sentence_end: ClassVar[int] = 256
    document_end: ClassVar[int] = 257
    bos: ClassVar[int] = 258
    pad: ClassVar[int] = 259
    output_size: ClassVar[int] = 258
    input_size: ClassVar[int] = 260

    def __post_init__(self) -> None:
        if (
            type(self.permutation) is not tuple
            or len(self.permutation) != 256
            or any(type(value) is not int for value in self.permutation)
            or set(self.permutation) != set(range(256))
        ):
            raise ValueError("byte mapping must be an immutable 256-element bijection")
        inverse = [0] * 256
        for canonical, encoded in enumerate(self.permutation):
            inverse[encoded] = canonical
        object.__setattr__(self, "inverse", tuple(inverse))
        object.__setattr__(
            self,
            "identity",
            content_identity(
                ("sentence-byte-codec-v1", "se=256,de=257,bos=258,pad=259", str(self.permutation))
            ),
        )

    def canonical_byte(self, observation: int) -> int:
        """Invert one byte, refusing control symbols in observable-byte probes."""
        if type(observation) is not int or not 0 <= observation < 256:
            raise ValueError("control or invalid observation is not a canonical byte")
        return self.inverse[observation]

    def encode_document(self, sentences: tuple[bytes, ...]) -> tuple[int, ...]:
        """Encode nonempty sentence payloads with a terminating document symbol."""
        if type(sentences) is not tuple or not sentences:
            raise ValueError("document requires an immutable nonempty sentence tuple")
        result: list[int] = []
        for index, sentence in enumerate(sentences):
            if type(sentence) is not bytes or not sentence:
                raise ValueError("sentence payload must be nonempty bytes")
            result.extend(self.permutation[value] for value in sentence)
            result.append(self.document_end if index == len(sentences) - 1 else self.sentence_end)
        return tuple(result)

    def decode_document(self, observations: tuple[int, ...]) -> tuple[bytes, ...]:
        """Invert the complete envelope, rejecting partial or ambiguous framing."""
        if type(observations) is not tuple or not observations:
            raise ValueError("encoded document must be an immutable nonempty tuple")
        if observations[-1] != self.document_end:
            raise ValueError("encoded document is missing document-end")
        sentences: list[bytes] = []
        pending = bytearray()
        for index, observation in enumerate(observations):
            if type(observation) is not int or not 0 <= observation < self.output_size:
                raise ValueError("invalid document observation or input-only control")
            if observation < 256:
                pending.append(self.canonical_byte(observation))
                continue
            if observation == self.document_end and index != len(observations) - 1:
                raise ValueError("document-end must terminate the envelope")
            if not pending:
                raise ValueError("empty sentence at boundary")
            sentences.append(bytes(pending))
            pending.clear()
        return tuple(sentences)
