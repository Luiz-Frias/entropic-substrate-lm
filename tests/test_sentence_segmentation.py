"""Production pySBD spans must cover source bytes without hidden text edits."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from entropic_substrate.sentence_prediction.records import SentenceSpan, SourceDocument
from entropic_substrate.sentence_prediction.segmentation import (
    OversizeDocument,
    SegmentationPolicy,
    SentenceSegmentation,
    segment_document,
)

pytestmark = pytest.mark.unit


def source(text: str) -> SourceDocument:
    return SourceDocument("fixture", text, "segmentation-test-v1")


@pytest.mark.parametrize(
    "text",
    [
        "  Hello.  Next.\n",
        "Dr. Jones arrived.\nNext sentence.",
        "é. Next.",
        "Hello without punctuation",
        "One.\n\nTwo.  ",
    ],
)
def test_real_segmenter_preserves_complete_source(text: str) -> None:
    result = segment_document(source(text), SegmentationPolicy())
    assert isinstance(result, SentenceSegmentation)
    assert b"".join(span.source for span in result.spans) == text.encode("utf-8")
    assert [span.index for span in result.spans] == list(range(len(result.spans)))
    assert result.spans[0].start == 0
    assert result.spans[-1].end == len(text.encode("utf-8"))


def test_offsets_count_utf8_bytes_not_python_characters() -> None:
    result = segment_document(source("é. Next."), SegmentationPolicy())
    assert isinstance(result, SentenceSegmentation)
    assert result.spans[0].source == "é. ".encode()
    assert result.spans[0].end == result.spans[1].start == 4


def test_oversize_sentence_rejects_whole_document_without_truncation() -> None:
    doc = source("Short. " + "z" * 20 + ".")
    result = segment_document(doc, SegmentationPolicy(max_sentence_bytes=10))
    assert isinstance(result, OversizeDocument)
    assert result.span.document is doc
    assert len(result.span.source) == 21
    assert result.span.index == 1
    assert result.reason == "oversize_sentence"
    with pytest.raises(ValueError, match="not oversize"):
        OversizeDocument(result.span, SegmentationPolicy(max_sentence_bytes=100))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit_is_not_a_runtime_default(limit: int) -> None:
    with pytest.raises(ValueError):
        SegmentationPolicy(max_sentence_bytes=limit)


def test_policy_identity_binds_limit_and_engine() -> None:
    policy = SegmentationPolicy()
    assert policy.identity != replace(policy, max_sentence_bytes=80).identity
    assert policy.engine_version == "0.3.4"


def test_wrong_installed_engine_version_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(
        "entropic_substrate.sentence_prediction.segmentation.metadata.version", lambda _: "0.0"
    )
    with pytest.raises(RuntimeError, match="version"):
        segment_document(source("Hello."), SegmentationPolicy())


@pytest.mark.parametrize(
    "spans",
    [
        [],
        [object()],
        [SimpleNamespace(start=1, end=6, sent="ello.")],
        [SimpleNamespace(start=0, end=6, sent="Edited")],
        [SimpleNamespace(start=0, end=3, sent="Hel")],
        [
            SimpleNamespace(start=0, end=6, sent="Hello."),
            SimpleNamespace(start=0, end=6, sent="Hello."),
        ],
    ],
)
def test_engine_text_loss_or_overlap_is_a_defect_not_repaired(monkeypatch, spans) -> None:
    monkeypatch.setattr("pysbd.Segmenter.segment", lambda self, text: spans)
    with pytest.raises(RuntimeError, match="segmenter"):
        segment_document(source("Hello."), SegmentationPolicy())


def test_manual_segmentation_cannot_omit_source_prefix() -> None:
    doc = source(" First.")
    with pytest.raises(ValueError, match="coverage"):
        SentenceSegmentation(doc, SegmentationPolicy(), (SentenceSpan(doc, 1, 7, 0),))
