"""Version-pinned offline sentence segmentation with exact source-byte coverage."""

from dataclasses import dataclass
from importlib import metadata
from typing import ClassVar, Literal

from pysbd import Segmenter

from entropic_substrate.sentence_prediction.records import (
    SentenceSpan,
    SourceDocument,
    content_identity,
)


@dataclass(frozen=True, slots=True)
class SegmentationPolicy:
    """Preprocessing construction, not a claim that sentences are concepts."""

    max_sentence_bytes: int = 4096
    engine_version: ClassVar[str] = "0.3.4"

    def __post_init__(self) -> None:
        if type(self.max_sentence_bytes) is not int or self.max_sentence_bytes < 1:
            raise ValueError("sentence byte limit must be a positive integer")

    @property
    def identity(self) -> str:
        """Bind segmentation, whitespace allocation, and whole-document rejection."""
        return content_identity(
            (
                "sentence-segmentation-v1",
                "pysbd",
                self.engine_version,
                "en,clean=False,char_span=True,whitespace-prefix,oversize-document-rejection",
                str(self.max_sentence_bytes),
            )
        )


@dataclass(frozen=True, slots=True)
class SentenceSegmentation:
    """An ordered, nonoverlapping partition of one complete source document."""

    document: SourceDocument
    policy: SegmentationPolicy
    spans: tuple[SentenceSpan, ...]

    def __post_init__(self) -> None:
        if type(self.document) is not SourceDocument or type(self.policy) is not SegmentationPolicy:
            raise TypeError("segmentation requires a source document and policy")
        if type(self.spans) is not tuple or not self.spans:
            raise ValueError("segmentation requires nonempty immutable spans")
        cursor = 0
        for index, span in enumerate(self.spans):
            if (
                type(span) is not SentenceSpan
                or span.document != self.document
                or span.index != index
                or span.start != cursor
                or len(span.source) > self.policy.max_sentence_bytes
            ):
                raise ValueError("sentence coverage, order, source binding, or size mismatch")
            cursor = span.end
        if cursor != len(self.document.source):
            raise ValueError("sentence coverage omits a source suffix")


@dataclass(frozen=True, slots=True)
class OversizeDocument:
    """Expected preprocessing rejection; no remaining fragments are admitted."""

    span: SentenceSpan
    policy: SegmentationPolicy
    reason: ClassVar[Literal["oversize_sentence"]] = "oversize_sentence"

    def __post_init__(self) -> None:
        if type(self.span) is not SentenceSpan or type(self.policy) is not SegmentationPolicy:
            raise TypeError("oversize rejection requires a bound span and policy")
        if len(self.span.source) <= self.policy.max_sentence_bytes:
            raise ValueError("rejected sentence is not oversize")


def segment_document(
    document: SourceDocument, policy: SegmentationPolicy
) -> SentenceSegmentation | OversizeDocument:
    """Partition original text; account for whitespace but never invent lost text.

    pySBD returns character offsets and may omit leading whitespace. Prefix
    whitespace belongs to the following span; a final whitespace-only suffix
    belongs to the last span. Any omitted non-whitespace or overlapping output
    is a defect. This offline boundary-augmented task is not a streaming lexer.
    """
    if type(document) is not SourceDocument or type(policy) is not SegmentationPolicy:
        raise TypeError("segmenter requires a source document and explicit policy")
    installed = metadata.version("pysbd")
    if installed != policy.engine_version:
        raise RuntimeError(f"segmenter version mismatch: {installed} != {policy.engine_version}")
    engine = Segmenter(language="en", clean=False, char_span=True)
    outputs = engine.segment(document.text)
    if not outputs:
        raise RuntimeError("segmenter returned no spans for a nonblank document")
    byte_offsets = [0]
    for character in document.text:
        byte_offsets.append(byte_offsets[-1] + len(character.encode("utf-8")))
    spans: list[SentenceSpan] = []
    cursor = 0
    for index, output in enumerate(outputs):
        # The library annotates a string return even in char_span mode. Validate
        # its external object shape instead of trusting or casting that union.
        start = getattr(output, "start", None)
        end = getattr(output, "end", None)
        text = getattr(output, "sent", None)
        if (
            type(start) is not int
            or type(end) is not int
            or type(text) is not str
            or not cursor <= start < end <= len(document.text)
            or text != document.text[start:end]
        ):
            raise RuntimeError("segmenter produced invalid, altered, or overlapping spans")
        if document.text[cursor:start].strip():
            raise RuntimeError("segmenter omitted non-whitespace source text")
        if index == len(outputs) - 1:
            if document.text[end:].strip():
                raise RuntimeError("segmenter omitted non-whitespace source suffix")
            end = len(document.text)
        spans.append(SentenceSpan(document, byte_offsets[cursor], byte_offsets[end], index))
        cursor = end
    for span in spans:
        if len(span.source) > policy.max_sentence_bytes:
            return OversizeDocument(span, policy)
    return SentenceSegmentation(document, policy, tuple(spans))
