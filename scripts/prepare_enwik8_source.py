"""Capture the first N bytes of enwik8 as a prepared sentence corpus (one document per page).

Usage::

    uv run python scripts/prepare_enwik8_source.py --bound-bytes 10000000 --output DIR
    uv run python scripts/prepare_enwik8_source.py --bound-bytes 10000000 --measure-segmenter

The raw file must be the pinned enwik8 (length and sha256 are checked over the WHOLE
file before any document is emitted, even for a bounded prefix). The capture is the
shape ``read_prepared_corpus`` admits and is read back through it. One JSON document
is printed on stdout with units spelled out in the key names (MB = 10^6 bytes).

A segmenter defect fails loud: this script never skips, retries, splits or drops a
page to get past one. ``prepare_capture`` lets the defect surface as
``SegmenterDefect`` (non-zero exit) naming the page the capture's own
``prepare_documents`` call was consuming. ``--measure-segmenter`` is a separate,
measurement-only pass: it times ``segment_document`` page by page and counts
defects, and it admits nothing into any capture.
"""

import argparse
import json
import os
import time
from collections.abc import Iterator, Sequence
from importlib import metadata
from pathlib import Path

from entropic_substrate.core.sentence.codec import CodecSpec
from entropic_substrate.core.sentence.corpus import (
    PreparationSpec,
    PreparedCorpus,
    SplitPolicy,
    prepare_documents,
)
from entropic_substrate.core.sentence.records import SourceDocument
from entropic_substrate.core.sentence.segmentation import (
    SegmentationPolicy,
    segment_document,
)
from entropic_substrate.kernel.lifecycle import ContentHash
from entropic_substrate.ladder.capture.capture import (
    read_prepared_corpus,
    write_prepared_corpus,
)
from entropic_substrate.ladder.capture.custody.enwik8 import (
    ENWIK8_RAW_BYTES,
    load_enwik8_bytes,
)
from entropic_substrate.ladder.capture.custody.enwik8_documents import (
    ENWIK8_REVISION,
    ENWIK8_SHA256,
    Enwik8Page,
    candidate_limit_for_bound,
    iter_enwik8_pages,
    verify_enwik8_raw,
)

DATASET_NAME = "enwik8"
# The worker input limit the capture must fit (the L2 brief's kit value).
MAX_OBJECT_BYTES = 2_147_483_648
_MB = 1_000_000


class SegmenterDefect(RuntimeError):
    """The segmenter failed on a page; the original error is the ``__cause__``."""


class _Current:
    """The page the preparation call pulled last: the one it is segmenting now."""

    page: Enwik8Page | None = None


def _tracked(pages: Iterator[Enwik8Page], current: _Current) -> Iterator[SourceDocument]:
    for page in pages:
        current.page = page
        yield page.document


def _check_engine() -> None:
    installed = metadata.version("pysbd")
    if installed != SegmentationPolicy.engine_version:
        raise RuntimeError(
            f"segmenter version mismatch: {installed} != {SegmentationPolicy.engine_version}"
        )


def _defect(page: Enwik8Page, error: RuntimeError) -> SegmenterDefect:
    title = "absent" if page.title is None else repr(page.title)
    return SegmenterDefect(
        f"segmenter defect at source id {page.document.source_id} title {title} "
        f"(bytes [{page.start}, {page.end})): {error}"
    )


def prepare_capture(
    prefix: bytes,
    *,
    output: Path,
    max_artifact_bytes: int,
    segmentation: SegmentationPolicy | None = None,
) -> dict[str, object]:
    """Prepare ``prefix`` (the first ``len(prefix)`` bytes of enwik8) and write the capture."""
    policy = SegmentationPolicy() if segmentation is None else segmentation
    codec_limit = CodecSpec().max_sentence_bytes
    if policy.max_sentence_bytes != codec_limit:
        raise ValueError(
            f"segmentation max_sentence_bytes {policy.max_sentence_bytes} differs "
            f"from the codec's {codec_limit}"
        )
    _check_engine()
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("source preparation requires an empty owned output directory")
    bound = len(prefix)
    spec = PreparationSpec(
        candidate_limit=candidate_limit_for_bound(bound),
        admitted_byte_limit=bound,
        revision=ENWIK8_REVISION,
        segmentation=policy,
        splits=SplitPolicy(),
        dataset_name=DATASET_NAME,
    )
    load = [os.getloadavg()[0]]
    current = _Current()
    started = time.perf_counter()
    try:
        corpus = prepare_documents(_tracked(iter_enwik8_pages(prefix, bound), current), spec)
    except RuntimeError as error:
        if current.page is None:
            raise
        raise _defect(current.page, error) from error
    prepare_seconds = time.perf_counter() - started
    if corpus.stopped_by != "source_exhausted":
        raise ValueError(f"capture stopped early: {corpus.stopped_by}")
    if not corpus.candidates:
        raise ValueError("enwik8 bound holds no pages")
    started = time.perf_counter()
    artifact = write_prepared_corpus(output, corpus, max_bytes=max_artifact_bytes)
    write_seconds = time.perf_counter() - started
    started = time.perf_counter()
    replay = read_prepared_corpus(output, artifact.content_hash, max_bytes=max_artifact_bytes)
    read_seconds = time.perf_counter() - started
    if replay != corpus:
        raise ValueError("capture replay differs from the prepared corpus")
    load.append(os.getloadavg()[0])
    return _report(corpus, bound, artifact.content_hash, output, max_artifact_bytes, load) | {
        "prepare_seconds": prepare_seconds,
        "write_seconds_including_one_replay": write_seconds,
        "read_prepared_corpus_replay_seconds": read_seconds,
    }


def _report(
    corpus: PreparedCorpus,
    bound: int,
    digest: ContentHash,
    output: Path,
    max_artifact_bytes: int,
    load: list[float],
) -> dict[str, object]:
    candidate_bytes = sum(len(d.source) for d in corpus.candidates)
    reasons = ("oversize_sentence", "duplicate_content", "source_budget")
    rejected = {
        reason: {
            "documents": sum(1 for r in corpus.rejections if r.reason == reason),
            "bytes": sum(len(r.document.source) for r in corpus.rejections if r.reason == reason),
        }
        for reason in reasons
    }
    return {
        "dataset_name": DATASET_NAME,
        "revision": ENWIK8_REVISION,
        "raw_sha256": ENWIK8_SHA256,
        "bound_bytes": bound,
        "stopped_by": corpus.stopped_by,
        "examined_documents": corpus.examined_documents,
        "examined_document_bytes": candidate_bytes,
        "admitted_documents": len(corpus.records),
        "admitted_bytes": corpus.admitted_bytes,
        "rejected": rejected,
        "max_sentence_bytes": corpus.spec.segmentation.max_sentence_bytes,
        "codec_max_sentence_bytes": CodecSpec().max_sentence_bytes,
        "capture_sha256": digest.digest,
        "capture_json_bytes": (output / "prepared-corpus.json").stat().st_size,
        "write_max_bytes": max_artifact_bytes,
        "max_object_bytes": MAX_OBJECT_BYTES,
        "pysbd_version": metadata.version("pysbd"),
        "load_average_1m_before_after": load,
    }


def measure_segmenter(prefix: bytes) -> dict[str, object]:
    """Time ``segment_document`` per page and count defects. Measurement only.

    Output is numbers and per-defect rows (source id, title, verbatim message);
    never a ``SourceDocument``, record or corpus. The capture path never reads it.
    """
    _check_engine()
    policy = SegmentationPolicy()
    pages = 0
    examined_bytes = 0
    seconds = 0.0
    defects: list[dict[str, object]] = []
    load = [os.getloadavg()[0]]
    for page in iter_enwik8_pages(prefix, len(prefix)):
        pages += 1
        examined_bytes += len(page.document.source)
        started = time.perf_counter()
        try:
            segment_document(page.document, policy)
        except RuntimeError as error:
            defects.append(
                {
                    "source_id": page.document.source_id,
                    "title": page.title,
                    "start": page.start,
                    "end": page.end,
                    "message": str(error),
                }
            )
        seconds += time.perf_counter() - started
    load.append(os.getloadavg()[0])
    return {
        "pysbd_version": metadata.version("pysbd"),
        "pages_examined": pages,
        "bytes_examined": examined_bytes,
        "seconds": seconds,
        "seconds_per_mb": seconds / (examined_bytes / _MB) if examined_bytes else 0.0,
        "defect_count": len(defects),
        "defects": defects,
        "load_average_1m_before_after": load,
    }


def load_verified_prefix(raw: Path, bound: int) -> bytes:
    """The first ``bound`` bytes, after the WHOLE file is verified as the pinned enwik8."""
    data = load_enwik8_bytes(raw)
    verify_enwik8_raw(memoryview(data))
    if not 1 <= bound <= ENWIK8_RAW_BYTES:
        raise ValueError(f"bound {bound} must lie within 1..{ENWIK8_RAW_BYTES}")
    return data[:bound].tobytes()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--raw", type=Path, default=Path("data/enwik8/enwik8"))
    parser.add_argument("--bound-bytes", type=int, default=ENWIK8_RAW_BYTES)
    parser.add_argument("--output", type=Path, help="empty caller-owned directory for the capture")
    parser.add_argument("--max-artifact-bytes", type=int, default=MAX_OBJECT_BYTES)
    parser.add_argument(
        "--measure-segmenter",
        action="store_true",
        help="measurement-only pass: time pysbd per page, count defects, write nothing",
    )
    args = parser.parse_args(argv)
    # Refuse a bad flag combination before the 100 MB read and SHA-256 pass, and keep
    # the measure-only contract (write nothing) at the CLI boundary: an --output that
    # the measurement branch would ignore could be mistaken for a written capture.
    output: Path | None = args.output
    if args.measure_segmenter and output is not None:
        parser.error("--measure-segmenter writes nothing and cannot be combined with --output")
    if not args.measure_segmenter and output is None:
        parser.error("--output is required unless --measure-segmenter is given")
    prefix = load_verified_prefix(args.raw, args.bound_bytes)
    if output is None:  # the checks above make this exactly the --measure-segmenter pass
        result = measure_segmenter(prefix)
    else:
        result = prepare_capture(prefix, output=output, max_artifact_bytes=args.max_artifact_bytes)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
