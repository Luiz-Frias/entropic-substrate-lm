"""enwik8 as a document stream: one verbatim ``<page>`` element per document.

The raw loader (``enwik8.py``) knows bytes; this module knows documents. It cuts
the first ``bound`` bytes into header, pages, inter-page gaps and, at most, one
declared trimmed tail, and builds one ``SourceDocument`` per page. Nothing is
decoded loosely, unescaped, stripped or normalised: a document is the exact byte
slice from ``<page>`` to ``</page>``. Split buckets, deduplication and
segmentation stay in ``core.sentence.corpus``; this module never assigns them and
never catches, counts or skips a segmenter defect.

Page rule, bound-independent by construction: a page starts at a literal
``<page>`` and ends at the next ``</page>``. Both occur only as element tags in
enwik8 (text is XML-escaped), so the bytes of a complete page, and therefore its
content digest and split bucket, never depend on where the bound falls. A partial
``<pa`` after the last complete page is gap. If the bound cuts a page open, that
page is emitted truncated, never dropped, trimmed only of an incomplete trailing
UTF-8 sequence (at most 3 bytes) with the trimmed count exposed; any other
invalid UTF-8 fails loud.
"""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from entropic_substrate.core.sentence.records import SourceDocument
from entropic_substrate.ladder.capture.custody.enwik8 import ENWIK8_RAW_BYTES

__all__ = [
    "ENWIK8_REVISION",
    "ENWIK8_SHA256",
    "MIN_PAGE_BYTES",
    "PAGE_CONSTRUCTION_VERSION",
    "Enwik8Page",
    "Enwik8Partition",
    "Enwik8Segment",
    "candidate_limit_for_bound",
    "iter_enwik8_documents",
    "iter_enwik8_pages",
    "partition_enwik8",
    "scan_enwik8",
    "verify_enwik8_raw",
]

ENWIK8_SHA256 = (
    "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"  # pragma: allowlist secret
)
# Any change to the slice, tail or source-id rule must change this token, and
# with it the revision, every document identity and every capture built on them.
PAGE_CONSTRUCTION_VERSION = "enwik8-page-v1"
# The single revision string of every capture: the raw file's digest plus the
# page construction. Never a CLI argument.
ENWIK8_REVISION = f"enwik8-sha256-{ENWIK8_SHA256}+{PAGE_CONSTRUCTION_VERSION}"

_PAGE_OPEN = b"<page>"
_PAGE_CLOSE = b"</page>"
# The shortest possible complete page; bounds a bound's page yield from above.
MIN_PAGE_BYTES = len(_PAGE_OPEN) + len(_PAGE_CLOSE)
_PAGE_HEAD = re.compile(r"\A<page>[ \t\r\n]*(?:<title>([^<]*)</title>[ \t\r\n]*)?<id>([0-9]+)</id>")

type SegmentKind = Literal["header", "page", "gap", "trimmed"]


@dataclass(frozen=True, slots=True)
class Enwik8Segment:
    """One run of raw bytes, accounted to exactly one part of the partition."""

    kind: SegmentKind
    start: int
    end: int
    complete: bool = True


@dataclass(frozen=True, slots=True)
class Enwik8Partition:
    """Every byte in ``[0, bound)`` belongs to exactly one segment, in order."""

    bound: int
    segments: tuple[Enwik8Segment, ...]

    @property
    def trimmed_tail_bytes(self) -> int:
        return sum(s.end - s.start for s in self.segments if s.kind == "trimmed")


@dataclass(frozen=True, slots=True)
class Enwik8Page:
    """A page document with its provenance beside it, never inside it."""

    document: SourceDocument
    title: str | None
    start: int
    end: int
    complete: bool


def _check_bound(raw: bytes, bound: int) -> None:
    if type(raw) is not bytes:
        raise TypeError("enwik8 raw input must be bytes")
    if type(bound) is not int or not 0 <= bound <= len(raw):
        raise ValueError(f"enwik8 bound must lie within the input (0..{len(raw)}), got {bound!r}")


def _trimmed_tail_bytes(data: bytes, offset: int) -> int:
    """Bytes of a valid-but-incomplete UTF-8 sequence ending ``data``; else fail loud."""
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as error:
        if error.reason == "unexpected end of data" and error.end == len(data):
            return len(data) - error.start
        raise ValueError(
            f"enwik8 bytes at {offset + error.start} are not valid UTF-8: {error.reason}"
        ) from error
    return 0


def _filler(raw: bytes, kind: SegmentKind, start: int, end: int) -> Enwik8Segment:
    """A header or gap run: kept out of every document, but still strict UTF-8."""
    if _trimmed_tail_bytes(raw[start:end], start):
        raise ValueError(f"enwik8 {kind} bytes at {start} end in a split UTF-8 character")
    return Enwik8Segment(kind, start, end)


def scan_enwik8(raw: bytes, bound: int) -> Iterator[Enwik8Segment]:
    """Partition ``raw[:bound]`` into header, page, gap and trimmed segments."""
    _check_bound(raw, bound)
    cursor = 0
    seen_page = False
    while True:
        start = raw.find(_PAGE_OPEN, cursor, bound)
        if start < 0:
            if cursor < bound:
                yield _filler(raw, "gap" if seen_page else "header", cursor, bound)
            return
        if start > cursor:
            yield _filler(raw, "gap" if seen_page else "header", cursor, start)
        close = raw.find(_PAGE_CLOSE, start, bound)
        if close >= 0:
            end = close + len(_PAGE_CLOSE)
            yield Enwik8Segment("page", start, end)
            cursor = end
            seen_page = True
            continue
        trimmed = _trimmed_tail_bytes(raw[start:bound], start)
        yield Enwik8Segment("page", start, bound - trimmed, complete=False)
        if trimmed:
            yield Enwik8Segment("trimmed", bound - trimmed, bound)
        return


def partition_enwik8(raw: bytes, bound: int) -> Enwik8Partition:
    return Enwik8Partition(bound, tuple(scan_enwik8(raw, bound)))


def _strict_text(raw: bytes, segment: Enwik8Segment) -> str:
    try:
        return raw[segment.start : segment.end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"enwik8 page at byte {segment.start} is not valid UTF-8: {error.reason}"
        ) from error


def iter_enwik8_pages(raw: bytes, bound: int) -> Iterator[Enwik8Page]:
    """One ``Enwik8Page`` per ``<page>`` in ``raw[:bound]``, in file order.

    The source id comes from the page's own leading ``<id>``: independent of the
    content digest, the title and the bound. A page without one, or a repeated id,
    fails loud; an id is never fabricated.
    """
    seen: set[str] = set()
    for segment in scan_enwik8(raw, bound):
        if segment.kind != "page":
            continue
        text = _strict_text(raw, segment)
        head = _PAGE_HEAD.match(text)
        if head is None:
            raise ValueError(f"enwik8 page at byte {segment.start} carries no leading page id")
        source_id = f"enwik8-page-{head.group(2)}"
        if source_id in seen:
            raise ValueError(f"enwik8 source id {source_id} repeats at byte {segment.start}")
        seen.add(source_id)
        yield Enwik8Page(
            SourceDocument(source_id, text, ENWIK8_REVISION),
            head.group(1),
            segment.start,
            segment.end,
            segment.complete,
        )


def iter_enwik8_documents(raw: bytes, bound: int) -> Iterator[SourceDocument]:
    """The document stream a ``PreparationSpec`` consumes."""
    for page in iter_enwik8_pages(raw, bound):
        yield page.document


def candidate_limit_for_bound(bound: int) -> int:
    """A candidate limit that strictly exceeds the pages ``bound`` bytes can yield.

    ``prepare_documents`` reports ``candidate_limit`` whenever the count equals the
    limit, even when the stream was exhausted there, so equality must never occur.
    """
    if type(bound) is not int or bound < 1:
        raise ValueError("bound must be a positive integer")
    return bound // MIN_PAGE_BYTES + 2


def verify_enwik8_raw(data: memoryview) -> None:
    """Refuse anything but the exact pinned enwik8, before any document is emitted."""
    if len(data) != ENWIK8_RAW_BYTES:
        raise ValueError(f"enwik8 length {len(data)} differs from pinned {ENWIK8_RAW_BYTES}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != ENWIK8_SHA256:
        raise ValueError(f"enwik8 sha256 {digest} differs from pinned {ENWIK8_SHA256}")
