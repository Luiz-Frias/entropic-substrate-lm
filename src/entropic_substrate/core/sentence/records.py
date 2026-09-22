"""Immutable source records and exact UTF-8 byte spans for sentence experiments."""

import hashlib
import json
from dataclasses import dataclass, field


def content_identity(parts: tuple[str, ...]) -> str:
    """Length-unambiguous identity for a declared construction or source record."""
    return hashlib.sha256(json.dumps(parts, ensure_ascii=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A complete document before splitting, segmentation, or byte recoding.

    Content identity is separate from source provenance so aliases of identical
    text cannot escape deduplication by changing their provider document ID.
    """

    source_id: str
    text: str
    revision: str
    source: bytes = field(init=False, repr=False)
    content_digest: str = field(init=False)
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("source_id", self.source_id),
            ("text", self.text),
            ("revision", self.revision),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"document {label} must be a nonblank string")
        try:
            source = self.text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("document is not valid UTF-8 text") from error
        digest = hashlib.sha256(source).hexdigest()
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "content_digest", digest)
        object.__setattr__(
            self,
            "identity",
            content_identity(("sentence-source-v1", self.revision, self.source_id, digest)),
        )


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """A nonempty source-byte slice, not a reconstructed or cleaned sentence.

    Retaining the immutable document makes source binding verifiable without a
    duplicated payload that could disagree with independently supplied offsets.
    Whole-document coverage is the segmentation adapter's separate contract.
    """

    document: SourceDocument
    start: int
    end: int
    index: int

    def __post_init__(self) -> None:
        if type(self.document) is not SourceDocument:
            raise TypeError("sentence requires a SourceDocument")
        if any(type(value) is not int for value in (self.start, self.end, self.index)):
            raise TypeError("sentence coordinates must be integers")
        if not 0 <= self.start < self.end <= len(self.document.source) or self.index < 0:
            raise ValueError("invalid sentence source coordinates")
        try:
            self.source.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("sentence offsets split a UTF-8 code point") from error

    @property
    def source(self) -> bytes:
        """Return precisely the original bytes, including surrounding whitespace."""
        return self.document.source[self.start : self.end]
