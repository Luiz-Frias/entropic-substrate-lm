"""enwik8 documents: one verbatim ``<page>`` slice each, a bound-independent split.

Every test here is hermetic: synthetic byte buffers only, no ``data/enwik8``, no
DVC, no network, no skips on data presence. Nothing pins a value that depends on a
segmentation, preparation, reader or capture identity; those are computed at test
time so a later segmentation fix cannot turn this file red.
"""

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from entropic_substrate.core.sentence import corpus as corpus_module
from entropic_substrate.core.sentence.corpus import PreparationSpec, SplitPolicy, prepare_documents
from entropic_substrate.core.sentence.records import SourceDocument
from entropic_substrate.core.sentence.segmentation import SegmentationPolicy
from entropic_substrate.kernel.lifecycle import ContentHash
from entropic_substrate.ladder.capture.capture import read_prepared_corpus
from entropic_substrate.ladder.capture.custody import enwik8_documents as adapter
from entropic_substrate.ladder.capture.custody.enwik8 import ENWIK8_RAW_BYTES
from entropic_substrate.ladder.capture.custody.enwik8_documents import (
    ENWIK8_REVISION,
    ENWIK8_SHA256,
    PAGE_CONSTRUCTION_VERSION,
    Enwik8Page,
    candidate_limit_for_bound,
    iter_enwik8_documents,
    iter_enwik8_pages,
    partition_enwik8,
    verify_enwik8_raw,
)

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_enwik8_source.py"

HEADER = (
    b'<mediawiki version="0.3">\n  <siteinfo>\n    <sitename>Wikipedia</sitename>\n'
    b"  </siteinfo>\n  "
)
GAP = b"\n  "
TRAILER = b"\n</mediawiki>"
PINNED_SHA256 = (
    "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"  # pragma: allowlist secret
)
DEFECT = "segmenter produced invalid, altered, or overlapping spans"


def page(pid: int, title: str, body: str) -> bytes:
    return (
        f"<page>\n    <title>{title}</title>\n    <id>{pid}</id>\n    <revision>\n"
        f"      <id>{pid + 1000}</id>\n      <text>{body}</text>\n    </revision>\n  </page>"
    ).encode()


def raw_of(*pages: bytes, trailer: bytes = b"") -> bytes:
    return HEADER + GAP.join(pages) + trailer


PAGES = (
    page(1, "Alpha", "Alpha one. Alpha two."),
    page(2, "Beta", "Beta uses a café and 5€ and \U0001f600 here. Done."),
    page(3, "Gamma", "Gamma one. Gamma two."),
)
RAW = raw_of(*PAGES, trailer=TRAILER)


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_enwik8_source_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- page splitting
def test_pages_split_on_page_boundaries_one_verbatim_slice_each() -> None:
    pages = list(iter_enwik8_pages(RAW, len(RAW)))
    assert [p.document.source for p in pages] == list(PAGES)
    assert all(p.document.source.startswith(b"<page>") for p in pages)
    assert all(p.document.source.endswith(b"</page>") for p in pages)
    assert all(p.complete for p in pages)
    assert [RAW[p.start : p.end] for p in pages] == list(PAGES)


def test_the_stream_adapter_yields_the_same_documents_as_the_pages() -> None:
    assert list(iter_enwik8_documents(RAW, len(RAW))) == [
        p.document for p in iter_enwik8_pages(RAW, len(RAW))
    ]


def test_a_raw_without_any_page_yields_nothing_and_is_all_header() -> None:
    assert list(iter_enwik8_pages(HEADER, len(HEADER))) == []
    assert [(s.kind, s.start, s.end) for s in partition_enwik8(HEADER, len(HEADER)).segments] == [
        ("header", 0, len(HEADER))
    ]
    assert list(iter_enwik8_pages(RAW, 0)) == []
    assert partition_enwik8(RAW, 0).segments == ()


def test_entities_markup_and_whitespace_are_kept_verbatim() -> None:
    body = "a &amp; b &lt;page&gt;\n\n  [[Link|text]]  \t"
    raw = raw_of(page(7, "T &amp; U", body))
    (only,) = iter_enwik8_pages(raw, len(raw))
    assert only.title == "T &amp; U"
    assert body in only.document.text


def test_the_bound_must_lie_inside_the_input() -> None:
    with pytest.raises(ValueError, match="bound"):
        list(iter_enwik8_pages(RAW, len(RAW) + 1))
    with pytest.raises(ValueError, match="bound"):
        list(iter_enwik8_pages(RAW, -1))


# ------------------------------------------------------------- byte coverage
def test_every_byte_of_every_bound_is_accounted_to_exactly_one_part() -> None:
    for bound in range(len(RAW) + 1):
        segments = partition_enwik8(RAW, bound).segments
        cursor = 0
        for segment in segments:
            assert segment.start == cursor, bound
            assert segment.end > segment.start, bound
            assert segment.kind in {"header", "page", "gap", "trimmed"}
            cursor = segment.end
        assert cursor == bound
        assert b"".join(RAW[s.start : s.end] for s in segments) == RAW[:bound]
        kinds = [s.kind for s in segments]
        assert kinds.count("header") <= 1
        assert kinds.count("trimmed") <= 1
        if "header" in kinds:
            assert kinds[0] == "header"


def test_header_gaps_and_trailer_are_not_pages() -> None:
    kinds = [(s.kind, RAW[s.start : s.end]) for s in partition_enwik8(RAW, len(RAW)).segments]
    assert kinds[0] == ("header", HEADER[: -len(GAP)] + GAP)
    assert [k for k, _ in kinds] == ["header", "page", "gap", "page", "gap", "page", "gap"]
    assert kinds[2][1] == GAP
    assert kinds[-1][1] == TRAILER


# ------------------------------------------------------------ prefix stability
def test_complete_pages_do_not_depend_on_where_the_bound_falls() -> None:
    full = list(iter_enwik8_pages(RAW, len(RAW)))
    checked = 0
    for bound in range(len(RAW) + 1):
        try:
            cut = list(iter_enwik8_pages(RAW, bound))
        except ValueError:
            # Only a tail page whose own id has not closed may refuse.
            tail = [s for s in partition_enwik8(RAW, bound).segments if s.kind == "page"][-1]
            assert not tail.complete
            assert b"</id>" not in RAW[tail.start : bound]
            continue
        complete = [p for p in cut if p.complete]
        expected = [p for p in full if p.end <= bound]
        assert complete == expected, bound
        assert [p.document.identity for p in complete] == [p.document.identity for p in expected]
        assert [p.document.source_id for p in cut] == [
            p.document.source_id for p in full[: len(cut)]
        ]
        assert len([p for p in cut if not p.complete]) <= 1
        checked += 1
    assert checked > len(RAW) // 2


def test_a_partial_open_tag_after_the_last_page_is_gap_never_page() -> None:
    first_end = RAW.index(b"</page>") + len(b"</page>")
    bound = first_end + len(GAP) + len(b"<pa")
    pages = list(iter_enwik8_pages(RAW, bound))
    assert [p.complete for p in pages] == [True]
    assert pages[0].document.source == PAGES[0]
    segments = partition_enwik8(RAW, bound).segments
    assert [s.kind for s in segments] == ["header", "page", "gap"]
    assert RAW[segments[-1].start : segments[-1].end] == GAP + b"<pa"


# --------------------------------------------------------- the truncated tail
def test_the_final_page_is_truncated_at_the_bound_never_dropped() -> None:
    first_end = RAW.index(b"</page>") + len(b"</page>")
    second_start = first_end + len(GAP)
    bound = second_start + 60  # inside page 2, after its id closed
    pages = list(iter_enwik8_pages(RAW, bound))
    assert [p.document.source_id for p in pages] == ["enwik8-page-1", "enwik8-page-2"]
    tail = pages[-1]
    assert not tail.complete
    assert (tail.start, tail.end) == (second_start, bound)
    assert tail.document.source == RAW[second_start:bound]
    assert tail.title == "Beta"
    assert partition_enwik8(RAW, bound).trimmed_tail_bytes == 0


@pytest.mark.parametrize("char", ["é", "€", "\U0001f600"], ids=["2byte", "3byte", "4byte"])
def test_a_split_utf8_character_at_the_bound_is_trimmed_and_counted(char: str) -> None:
    raw = raw_of(page(1, "Alpha", "one"), page(2, "Beta", f"x{char}y"))
    start = raw.index(char.encode())
    width = len(char.encode())
    for kept in range(1, width):
        bound = start + kept
        partition = partition_enwik8(raw, bound)
        assert partition.trimmed_tail_bytes == kept
        assert partition.segments[-1].kind == "trimmed"
        (*_, tail) = iter_enwik8_pages(raw, bound)
        assert not tail.complete
        assert tail.end == start
        assert tail.document.source == raw[tail.start : start]
        assert tail.document.text.endswith("x")
    on_boundary = partition_enwik8(raw, start + width)
    assert on_boundary.trimmed_tail_bytes == 0
    (*_, whole) = iter_enwik8_pages(raw, start + width)
    assert whole.document.text.endswith(char)


@pytest.mark.parametrize(
    "tail",
    [
        b"\x80",
        b"\xff",
        b"\xc0\x80",
        b"\xe0\x80",
        b"\xed\xa0",
        b"\xf4\x90\x80",
        b"\xe2(",
        b"\xe2\x82(",
    ],
    ids=[
        "lone-continuation",
        "invalid-lead",
        "overlong-2",
        "overlong-3-prefix",
        "surrogate-prefix",
        "above-max-prefix",
        "bad-continuation-2",
        "bad-continuation-3",
    ],
)
def test_invalid_utf8_at_the_bound_fails_loud_instead_of_being_trimmed(tail: bytes) -> None:
    raw = raw_of(page(1, "Alpha", "one"), page(2, "Beta", "x")) + tail
    start = raw.rindex(b"<page>")
    cut = raw[: raw.index(b"</text>", start)] + tail
    with pytest.raises(ValueError, match="UTF-8"):
        list(iter_enwik8_pages(cut, len(cut)))
    with pytest.raises(ValueError, match="UTF-8"):
        partition_enwik8(cut, len(cut))


def test_invalid_utf8_inside_a_complete_page_fails_loud() -> None:
    bad = page(1, "Alpha", "caf") + b""
    bad = bad.replace(b"</text>", b"\xe9</text>")
    raw = raw_of(bad)
    with pytest.raises(ValueError, match="UTF-8"):
        list(iter_enwik8_pages(raw, len(raw)))


@pytest.mark.parametrize("where", ["header", "gap", "trailer"])
def test_invalid_utf8_in_header_gap_or_trailer_fails_loud(where: str) -> None:
    bad = b"\xe9"
    header = HEADER + (bad if where == "header" else b"")
    gap = GAP + (bad if where == "gap" else b"")
    trailer = TRAILER + (bad if where == "trailer" else b"")
    raw = header + page(1, "A", "one") + gap + page(2, "B", "two") + trailer
    with pytest.raises(ValueError, match="UTF-8"):
        partition_enwik8(raw, len(raw))
    with pytest.raises(ValueError, match="UTF-8"):
        list(iter_enwik8_pages(raw, len(raw)))


def test_a_tail_page_without_a_closed_id_fails_loud_never_fabricates_one() -> None:
    start = RAW.index(b"<page>")
    for bound in (start + len(b"<page>"), start + 20, RAW.index(b"<id>", start) + 6):
        with pytest.raises(ValueError, match="page id"):
            list(iter_enwik8_pages(RAW, bound))


def test_a_truncated_page_without_a_title_reports_it_absent() -> None:
    raw = HEADER + b"<page>\n    <id>41</id>\n    <revision>"
    (only,) = iter_enwik8_pages(raw, len(raw))
    assert only.title is None
    assert only.document.source_id == "enwik8-page-41"
    assert not only.complete


# ----------------------------------------------------------------- source ids
def test_source_ids_are_the_page_ids_and_unique_even_when_titles_repeat() -> None:
    raw = raw_of(page(10, "Same", "one"), page(11, "Same", "two"), page(12, "Same", "one"))
    ids = [p.document.source_id for p in iter_enwik8_pages(raw, len(raw))]
    assert ids == ["enwik8-page-10", "enwik8-page-11", "enwik8-page-12"]
    assert len(set(ids)) == 3


def test_a_repeated_page_id_fails_loud_even_for_byte_identical_pages() -> None:
    twin = page(5, "Same", "one")
    raw = raw_of(twin, twin)
    with pytest.raises(ValueError, match="source id"):
        list(iter_enwik8_pages(raw, len(raw)))
    other = raw_of(page(5, "A", "one"), page(5, "B", "two"))
    with pytest.raises(ValueError, match="source id"):
        list(iter_enwik8_pages(other, len(other)))


def test_source_ids_do_not_depend_on_the_bound_or_the_content() -> None:
    full = {p.start: p.document.source_id for p in iter_enwik8_pages(RAW, len(RAW))}
    bound = RAW.index(b"</page>", RAW.index(b"Gamma")) - 5
    cut = {p.start: p.document.source_id for p in iter_enwik8_pages(RAW, bound)}
    assert cut == full


# ------------------------------------------------------------------- revision
def test_there_is_exactly_one_revision_string_bound_to_the_raw_and_the_split_rule() -> None:
    assert ENWIK8_SHA256 == PINNED_SHA256
    assert PAGE_CONSTRUCTION_VERSION == "enwik8-page-v1"
    assert ENWIK8_REVISION == f"enwik8-sha256-{ENWIK8_SHA256}+{PAGE_CONSTRUCTION_VERSION}"
    revisions = {d.revision for d in iter_enwik8_documents(RAW, len(RAW))}
    assert revisions == {ENWIK8_REVISION}
    bound = RAW.index(b"Gamma")
    cut_revisions = {d.revision for d in iter_enwik8_documents(RAW, bound + 40)}
    assert cut_revisions == {ENWIK8_REVISION}


def test_verification_refuses_a_wrong_length_or_a_wrong_digest() -> None:
    with pytest.raises(ValueError, match="length"):
        verify_enwik8_raw(memoryview(bytes(10)))
    with pytest.raises(ValueError, match="sha256"):
        verify_enwik8_raw(memoryview(bytes(ENWIK8_RAW_BYTES)))


# --------------------------------------------------------------------- splits
def test_every_record_split_is_the_split_policy_bucket_of_its_whole_document() -> None:
    spec = PreparationSpec(
        candidate_limit_for_bound(len(RAW)), len(RAW), ENWIK8_REVISION, dataset_name="enwik8"
    )
    corpus = prepare_documents(iter_enwik8_documents(RAW, len(RAW)), spec)
    assert len(corpus.records) == len(PAGES)
    policy = SplitPolicy()
    for row in corpus.records:
        assert row.split == policy.assign(row.segmentation.document)
        assert row.segmentation.document.source.startswith(b"<page>")


# -------------------------------------------------------- the candidate limit
def test_the_candidate_limit_strictly_exceeds_the_densest_possible_yield() -> None:
    minimal = b"<page><id>1</id></page>"
    assert candidate_limit_for_bound(len(minimal)) > 1
    for bound in range(1, 200):
        densest = bound // len(b"<page></page>") + 1  # complete pages plus one tail
        assert candidate_limit_for_bound(bound) > densest


def test_a_capture_whose_count_equals_a_naive_limit_still_ends_source_exhausted() -> None:
    count = len(PAGES)
    naive = PreparationSpec(count, len(RAW), ENWIK8_REVISION, dataset_name="enwik8")
    binding = prepare_documents(iter_enwik8_documents(RAW, len(RAW)), naive)
    assert binding.stopped_by == "candidate_limit"  # the edge the derived limit must dodge
    derived = PreparationSpec(
        candidate_limit_for_bound(len(RAW)), len(RAW), ENWIK8_REVISION, dataset_name="enwik8"
    )
    assert prepare_documents(iter_enwik8_documents(RAW, len(RAW)), derived).stopped_by == (
        "source_exhausted"
    )


# --------------------------------------------------------------- bounded 10 MB
def test_a_ten_megabyte_synthetic_bound_cuts_mid_page_without_a_segmenter() -> None:
    body = "x" * 7000
    pages = [page(i + 1, f"T{i}", body) for i in range(1500)]
    raw = raw_of(*pages)
    bound = 10_000_000
    assert len(raw) > bound
    result = list(iter_enwik8_pages(raw, bound))
    assert all(p.complete for p in result[:-1])
    assert not result[-1].complete
    assert result[-1].end == bound
    assert len({p.document.source_id for p in result}) == len(result)
    covered = sum(s.end - s.start for s in partition_enwik8(raw, bound).segments)
    assert covered == bound


# ----------------------------------------------------------- capture round trip
def test_write_then_read_round_trips_through_read_prepared_corpus(tmp_path: Path) -> None:
    script = _script()
    report = script.prepare_capture(RAW, output=tmp_path, max_artifact_bytes=1_000_000)
    assert report["stopped_by"] == "source_exhausted"
    assert report["examined_documents"] == len(PAGES)
    assert report["admitted_documents"] == len(PAGES)
    assert report["revision"] == ENWIK8_REVISION
    assert report["dataset_name"] == "enwik8"
    assert report["capture_json_bytes"] == (tmp_path / "prepared-corpus.json").stat().st_size

    digest = ContentHash("sha256", str(report["capture_sha256"]))
    replay = read_prepared_corpus(tmp_path, digest, max_bytes=1_000_000)
    assert [d.source for d in replay.candidates] == list(PAGES)
    assert replay.spec.dataset_name == "enwik8"
    assert replay.spec.revision == ENWIK8_REVISION
    assert replay.spec.candidate_limit > replay.examined_documents


def test_a_bounded_prefix_capture_is_a_prefix_of_the_whole(tmp_path: Path) -> None:
    script = _script()
    bound = RAW.index(b"</page>", RAW.index(b"Beta")) + len(b"</page>")
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    report = script.prepare_capture(RAW[:bound], output=prefix, max_artifact_bytes=1_000_000)
    assert report["examined_documents"] == 2
    assert report["stopped_by"] == "source_exhausted"
    whole = list(iter_enwik8_documents(RAW, len(RAW)))
    cut = list(iter_enwik8_documents(RAW, bound))
    assert [d.identity for d in cut] == [d.identity for d in whole[:2]]


def test_prepare_refuses_a_nonempty_output_directory(tmp_path: Path) -> None:
    script = _script()
    (tmp_path / "squatter").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        script.prepare_capture(RAW, output=tmp_path, max_artifact_bytes=1_000_000)
    assert [p.name for p in tmp_path.iterdir()] == ["squatter"]


def test_prepare_refuses_a_sentence_limit_that_differs_from_the_codec_default(
    tmp_path: Path,
) -> None:
    from entropic_substrate.core.sentence.codec import CodecSpec

    script = _script()
    other = SegmentationPolicy(CodecSpec().max_sentence_bytes - 1)
    with pytest.raises(ValueError, match="max_sentence_bytes"):
        script.prepare_capture(
            RAW, output=tmp_path, max_artifact_bytes=1_000_000, segmentation=other
        )
    assert list(tmp_path.iterdir()) == []


def test_prepare_refuses_an_empty_bound(tmp_path: Path) -> None:
    script = _script()
    with pytest.raises(ValueError):
        script.prepare_capture(b"", output=tmp_path, max_artifact_bytes=1_000_000)
    with pytest.raises(ValueError, match="no pages"):
        script.prepare_capture(HEADER, output=tmp_path, max_artifact_bytes=1_000_000)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------- the segmenter defect fails loud
def test_a_segmenter_defect_exits_nonzero_naming_the_page_and_writes_no_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script()
    real = corpus_module.segment_document

    def flawed(document: SourceDocument, policy: SegmentationPolicy) -> Any:
        if document.source_id == "enwik8-page-2":
            raise RuntimeError(DEFECT)
        return real(document, policy)

    monkeypatch.setattr(corpus_module, "segment_document", flawed)
    with pytest.raises(script.SegmenterDefect) as caught:
        script.prepare_capture(RAW, output=tmp_path, max_artifact_bytes=1_000_000)
    message = str(caught.value)
    assert "enwik8-page-2" in message
    assert "Beta" in message
    assert DEFECT in message
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == DEFECT
    assert list(tmp_path.iterdir()) == []


def test_main_exits_nonzero_on_the_defect_and_names_source_id_and_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script()
    monkeypatch.setattr(script, "load_verified_prefix", lambda raw, bound: RAW)

    def flawed(document: SourceDocument, policy: SegmentationPolicy) -> Any:
        raise RuntimeError(DEFECT)

    monkeypatch.setattr(corpus_module, "segment_document", flawed)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(script.SegmenterDefect, match="enwik8-page-1") as caught:
        script.main(["--raw", str(tmp_path / "raw"), "--bound-bytes", "10", "--output", str(out)])
    assert "Alpha" in str(caught.value)
    assert list(out.iterdir()) == []


def test_an_engine_version_mismatch_is_not_relabelled_as_a_segmenter_defect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script()
    monkeypatch.setattr(script.metadata, "version", lambda name: "0.0.0")
    with pytest.raises(RuntimeError, match="version") as caught:
        script.prepare_capture(RAW, output=tmp_path, max_artifact_bytes=1_000_000)
    assert not isinstance(caught.value, script.SegmenterDefect)
    assert list(tmp_path.iterdir()) == []


def test_an_error_before_the_first_page_is_reraised_unchanged_and_never_a_defect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script()
    raised = RuntimeError("page iterator failed before yielding a page")

    def broken(prefix: bytes, bound: int) -> Iterator[Enwik8Page]:
        yield from ()
        raise raised

    monkeypatch.setattr(script, "iter_enwik8_pages", broken)
    with pytest.raises(RuntimeError) as caught:
        script.prepare_capture(RAW, output=tmp_path, max_artifact_bytes=1_000_000)
    assert caught.value is raised
    assert not isinstance(caught.value, script.SegmenterDefect)
    assert caught.value.__cause__ is None
    assert list(tmp_path.iterdir()) == []


def test_a_nonsegmenter_error_during_preparation_is_not_annotated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script()
    twin = page(5, "Same", "one")
    with pytest.raises(ValueError, match="source id"):
        script.prepare_capture(raw_of(twin, twin), output=tmp_path, max_artifact_bytes=1_000_000)


def test_main_refuses_a_raw_that_is_not_the_pinned_enwik8(tmp_path: Path) -> None:
    script = _script()
    raw = tmp_path / "enwik8"
    raw.write_bytes(RAW)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="length"):
        script.main(["--raw", str(raw), "--bound-bytes", "100", "--output", str(out)])
    assert list(out.iterdir()) == []


def _forbid_input_read(script: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    def read(raw: Path, bound: int) -> bytes:
        raise AssertionError("the input was read before the flag combination was refused")

    monkeypatch.setattr(script, "load_verified_prefix", read)


def test_main_refuses_measure_segmenter_with_output_before_reading_the_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _script()
    _forbid_input_read(script, monkeypatch)
    out = tmp_path / "out"
    argv = ["--raw", str(tmp_path / "absent-raw"), "--measure-segmenter", "--output", str(out)]
    with pytest.raises(SystemExit) as caught:
        script.main(argv)
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "--measure-segmenter" in captured.err
    assert "--output" in captured.err
    assert captured.out == ""
    assert not out.exists()


def test_main_refuses_a_missing_output_before_reading_the_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _script()
    _forbid_input_read(script, monkeypatch)
    with pytest.raises(SystemExit) as caught:
        script.main(["--raw", str(tmp_path / "absent-raw")])
    assert caught.value.code == 2
    assert "--output is required" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_main_measure_segmenter_alone_reads_the_input_prints_the_report_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _script()
    monkeypatch.setattr(script, "load_verified_prefix", lambda raw, bound: RAW)
    argv = ["--raw", str(tmp_path / "raw"), "--bound-bytes", "10", "--measure-segmenter"]
    assert script.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["pages_examined"] == len(PAGES)
    assert report["defect_count"] == 0
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------ the measurement-only pass
def test_the_measurement_pass_counts_defects_and_admits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script()
    real = script.segment_document

    def flawed(document: SourceDocument, policy: SegmentationPolicy) -> Any:
        if document.source_id == "enwik8-page-3":
            raise RuntimeError(DEFECT)
        return real(document, policy)

    monkeypatch.setattr(script, "segment_document", flawed)
    report = script.measure_segmenter(RAW)
    assert report["pages_examined"] == 3
    assert report["bytes_examined"] == sum(len(p) for p in PAGES)
    assert report["defect_count"] == 1
    (row,) = report["defects"]
    assert row["source_id"] == "enwik8-page-3"
    assert row["title"] == "Gamma"
    assert row["message"] == DEFECT
    assert report["seconds_per_mb"] > 0
    assert set(report) >= {"seconds", "pysbd_version", "pages_examined", "bytes_examined"}
    json.dumps(report)  # numbers and rows only, never documents
    assert list(tmp_path.iterdir()) == []


def test_the_measurement_pass_fails_loud_once_on_a_wrong_engine_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script()
    monkeypatch.setattr(script.metadata, "version", lambda name: "0.0.0")
    with pytest.raises(RuntimeError, match="version"):
        script.measure_segmenter(RAW)


def test_the_source_module_exposes_no_error_swallowing_helper() -> None:
    public = {name for name in dir(adapter) if not name.startswith("_")}
    assert not {n for n in public if "defect" in n.lower() or "skip" in n.lower()}
    source = Path(adapter.__file__ or "").read_text(encoding="utf-8")
    assert "except RuntimeError" not in source
    assert "print(" not in source
