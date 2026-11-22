"""Bounded document admission; aliases never become independent split samples."""

import sys
from collections.abc import Generator, Iterator
from dataclasses import replace
from itertools import count
from types import SimpleNamespace

import pytest

from entropic_substrate.core.sentence.corpus import (
    PreparationSpec,
    SplitPolicy,
    prepare_documents,
)
from entropic_substrate.core.sentence.records import SourceDocument
from entropic_substrate.core.sentence.segmentation import SegmentationPolicy

pytestmark = pytest.mark.unit


def document(number: int, text: str | None = None) -> SourceDocument:
    return SourceDocument(f"doc-{number}", text or f"Sentence {number}.", "fixture-v1")


def spec(*, segmentation: SegmentationPolicy | None = None) -> PreparationSpec:
    return PreparationSpec(
        candidate_limit=10,
        admitted_byte_limit=1000,
        revision="fixture-v1",
        segmentation=segmentation if segmentation is not None else SegmentationPolicy(),
    )


def test_duplicate_content_cannot_escape_by_changing_source_id() -> None:
    original = document(1)
    alias = replace(original, source_id="alias")
    policy = SplitPolicy()
    assert policy.assign(original) == policy.assign(alias)
    assert {policy.assign(replace(original, source_id=f"alias-{i}")) for i in range(100)} == {
        policy.assign(original)
    }
    result = prepare_documents([original, alias, document(2)], spec())
    assert len(result.records) == 2
    assert result.examined_documents == 3
    assert result.rejections[0].reason == "duplicate_content"
    assert result.rejections[0].source_id == "alias"
    assert result.admitted_bytes == sum(
        len(row.segmentation.document.source) for row in result.records
    )


def test_source_identity_reuse_with_changed_content_is_a_defect() -> None:
    doc = document(1)
    with pytest.raises(ValueError, match="reused"):
        prepare_documents([doc, replace(doc, text="Different.")], spec())


def test_preparation_refuses_mixed_source_revision() -> None:
    with pytest.raises(ValueError, match="revision"):
        prepare_documents([replace(document(1), revision="wrong")], spec())


def test_candidate_limit_bounds_even_an_infinite_duplicate_stream() -> None:
    consumed: list[int] = []

    def stream() -> Iterator[SourceDocument]:
        for index in count():
            consumed.append(index)
            yield document(index, "Identical.")

    result = prepare_documents(stream(), replace(spec(), candidate_limit=3))
    assert consumed == [0, 1, 2]
    assert len(result.records) == 1
    assert len(result.rejections) == 2


def test_budget_stops_at_complete_documents_without_looking_for_shorter_ones() -> None:
    rows = [document(1, "Fits."), document(2, "Too long."), document(3, "X.")]
    result = prepare_documents(rows, replace(spec(), admitted_byte_limit=6))
    assert result.examined_documents == 2
    assert result.admitted_bytes == 5
    assert len(result.records) == 1
    assert result.rejections[-1].reason == "source_budget"
    assert result.stopped_by == "source_budget"


def test_oversize_document_is_recorded_without_admitting_its_short_fragments() -> None:
    result = prepare_documents(
        [document(1, "Short. " + "Z" * 30 + ".")], spec(segmentation=SegmentationPolicy(10))
    )
    assert result.status == "empty"
    assert result.records == ()
    assert result.rejections[0].reason == "oversize_sentence"
    assert result.rejections[0].sentence_index == 1


def test_split_assignment_is_content_stable_and_covers_three_partitions() -> None:
    policy = SplitPolicy()
    assignments = {policy.assign(document(index)) for index in range(300)}
    assert assignments == {"train", "validation", "test"}
    assert policy.identity != replace(policy, seed=43).identity
    assert policy.identity != replace(policy, validation_buckets=501).identity


def test_identity_binds_source_order_and_preparation_policy() -> None:
    left, right = document(1), document(2)
    first = prepare_documents([left, right], spec())
    assert first.identity == prepare_documents([left, right], spec()).identity
    assert first.identity != prepare_documents([right, left], spec()).identity
    assert (
        first.identity
        != prepare_documents([left, right], replace(spec(), candidate_limit=11)).identity
    )


@pytest.mark.parametrize(
    "change", [{"candidate_limit": 0}, {"admitted_byte_limit": True}, {"revision": ""}]
)
def test_preparation_spec_refuses_missing_or_invalid_limits(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(spec(), **change)


@pytest.mark.parametrize(
    "change",
    [{"validation_buckets": 0}, {"test_buckets": True}, {"test_buckets": 10000}, {"seed": -1}],
)
def test_split_policy_refuses_invalid_partition(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(SplitPolicy(), **change)


def test_fineweb_document_adapter_is_bounded_pinned_and_unflattened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from entropic_substrate.ladder.capture.custody import fineweb

    seen: dict[str, object] = {}
    consumed: list[int] = []

    def fake_load(repo: str, **kwargs: object) -> Iterator[dict[str, str]]:
        seen.update(repo=repo, **kwargs)
        for index in range(3):
            consumed.append(index)
            yield {"id": f"provider-{index}", "text": "  Text.\n"}

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load))
    rows = list(fineweb.stream_fineweb_documents(max_documents=1))
    assert consumed == [0]
    assert rows[0].text == "  Text.\n"
    assert rows[0].source == b"  Text.\n"
    assert rows[0].source_id.endswith("/provider-0")
    assert rows[0].revision == seen["revision"] == fineweb.FINEWEB_REVISION
    assert seen["streaming"] is True


@pytest.mark.parametrize(
    "row",
    [
        {"text": "No ID."},
        {"id": "id", "text": 5},
        {"id": "id", "text": ""},
        {"id": "id", "text": "  "},
        {"id": " ", "text": "Text."},
    ],
)
def test_fineweb_adapter_refuses_invented_ids_or_text_coercion(
    monkeypatch: pytest.MonkeyPatch, row: dict[str, object]
) -> None:
    from entropic_substrate.ladder.capture.custody import fineweb

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: (r for r in [row]))
    )
    with pytest.raises(ValueError):
        list(fineweb.stream_fineweb_documents(max_documents=1))


@pytest.mark.parametrize("stop", ["consumer_close", "limit", "invalid_row"])
def test_fineweb_owns_and_closes_the_underlying_dataset_iterator(
    monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    from entropic_substrate.ladder.capture.custody import fineweb

    closed = []

    def rows() -> Generator[dict[str, object], None, None]:
        try:
            yield {"id": "one", "text": None if stop == "invalid_row" else "First."}
            pytest.fail("bounded loader read beyond its requested prefix")
        finally:
            closed.append(True)

    iterator = rows()  # Keep a strong reference: garbage collection cannot pass this test.
    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: iterator)
    )
    documents = fineweb.stream_fineweb_documents(max_documents=1)
    try:
        if stop == "consumer_close":
            next(documents)
            documents.close()
        elif stop == "invalid_row":
            with pytest.raises(ValueError):
                list(documents)
        else:
            assert len(list(documents)) == 1
        assert closed == [True]
    finally:
        iterator.close()


def test_fineweb_refuses_a_backend_without_explicit_generator_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from entropic_substrate.ladder.capture.custody import fineweb

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: []))
    with pytest.raises(TypeError, match="closeable dataset generator"):
        list(fineweb.stream_fineweb_documents(max_documents=1))
