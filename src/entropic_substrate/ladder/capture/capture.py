"""Immutable local source capture and replay; not a scientific filing receipt."""

import json
from importlib import import_module, metadata
from pathlib import Path
from platform import python_version
from typing import cast

from entropic_substrate.core.sentence.corpus import (
    PreparationSpec,
    PreparedCorpus,
    SplitPolicy,
    prepare_documents,
)
from entropic_substrate.core.sentence.records import SourceDocument, content_identity
from entropic_substrate.core.sentence.segmentation import SegmentationPolicy
from entropic_substrate.kernel.artifacts import (
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash

_VERSION = "h-source-capture-v2"
_FILENAME = "prepared-corpus.json"
# Faithful restatement (ruling E-2) of the pre-move member set at their
# landed dotted names: the package member resolves to core.sentence (its
# own __init__.py) per the p4 design-gate package-member constraint, not
# the pre-move bare "sentence_prediction" name.
_READER_MODULES = (
    "core.sentence",
    "core.sentence.records",
    "core.sentence.segmentation",
    "core.sentence.corpus",
    "ladder.capture.capture",
    "kernel.artifacts",
    "kernel.lifecycle",
)


def _reader_identity() -> str:
    """Pin the full interpreter version; widening compatibility needs validation."""
    digests = []
    for name in _READER_MODULES:
        module = import_module(f"entropic_substrate.{name}")
        if module.__file__ is None:
            raise RuntimeError(f"source reader cannot bind module {name}")
        source = Path(module.__file__)
        _, _, digest = read_d1_artifact_bytes(source.parent, source.name, label="H source reader")
        digests.append(f"{name}:{digest}")
    return content_identity(
        ("h-source-reader-v3", python_version(), metadata.version("pysbd"), *digests)
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _summary(corpus: PreparedCorpus) -> dict[str, object]:
    return {
        "status": corpus.status,
        "stopped_by": corpus.stopped_by,
        "examined_documents": corpus.examined_documents,
        "admitted_bytes": corpus.admitted_bytes,
        "records": [
            {
                "source_identity": row.segmentation.document.identity,
                "split": row.split,
                "spans": [[span.start, span.end, span.index] for span in row.segmentation.spans],
            }
            for row in corpus.records
        ],
        "rejections": [
            {
                "source_identity": row.document.identity,
                "reason": row.reason,
                "sentence_index": row.sentence_index,
            }
            for row in corpus.rejections
        ],
    }


def write_prepared_corpus(
    root: Path, corpus: PreparedCorpus, *, max_bytes: int
) -> ArtifactIdentity:
    """Replay admission, then publish without overwriting any existing target.

    The caller owns an existing directory and supplies an explicit byte limit.
    All examined candidates, including rejections, are retained for replay.
    """
    if type(corpus) is not PreparedCorpus:
        raise TypeError("source capture requires a PreparedCorpus")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("source capture requires a positive byte limit")
    replayed = prepare_documents(corpus.candidates, corpus.spec)
    if replayed != corpus:
        raise ValueError("source preparation does not match independent replay")
    spec = corpus.spec
    payload = {
        "format_version": _VERSION,
        "reader_id": _reader_identity(),
        "corpus_id": corpus.identity,
        "spec": {
            "candidate_limit": spec.candidate_limit,
            "admitted_byte_limit": spec.admitted_byte_limit,
            "revision": spec.revision,
            "dataset_name": spec.dataset_name,
            "max_sentence_bytes": spec.segmentation.max_sentence_bytes,
            "engine_version": spec.segmentation.engine_version,
            "split_seed": spec.splits.seed,
            "validation_buckets": spec.splits.validation_buckets,
            "test_buckets": spec.splits.test_buckets,
        },
        "candidates": [
            {
                "source_id": doc.source_id,
                "text": doc.text,
                "revision": doc.revision,
                "identity": doc.identity,
            }
            for doc in corpus.candidates
        ],
        "summary": _summary(corpus),
    }
    encoded = _json_bytes(payload)
    if len(encoded) > max_bytes:
        raise ValueError("source capture exceeds its byte limit")
    return publish_d1_atomic_bytes(root, _FILENAME, encoded, label="H prepared source")


def _fields(value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError("source capture fields differ from the declared schema")
    return cast(dict[str, object], value)


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("source capture integer field has wrong type")
    return value


def _text(value: object) -> str:
    if type(value) is not str:
        raise ValueError("source capture string field has wrong type")
    return value


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate source capture key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant in source capture: {value}")


def read_prepared_corpus(root: Path, expected: ContentHash, *, max_bytes: int) -> PreparedCorpus:
    """Verify external digest and reader identity, then recompute every admission.

    V2 retains declared dataset provenance even for empty captures. V1 requires
    its original reader; historical bytes never acquire invented provenance.
    Unknown versions fail explicitly. No old E/D reader is widened.
    """
    if type(expected) is not ContentHash or expected.algorithm != "sha256":
        raise ValueError("source capture requires an expected SHA-256 digest")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("source capture requires a positive byte limit")
    _, encoded, digest = read_d1_artifact_bytes(
        root, _FILENAME, label="H prepared source", max_bytes=max_bytes
    )
    if digest != expected.digest:
        raise ValueError("source capture digest does not match expected artifact")
    payload = _fields(
        json.loads(encoded, object_pairs_hook=_pairs, parse_constant=_reject_constant),
        {"format_version", "reader_id", "corpus_id", "spec", "candidates", "summary"},
    )
    if payload["format_version"] == "h-source-capture-v1":
        raise ValueError("legacy source capture requires its original reader; no migration")
    if payload["format_version"] != _VERSION:
        raise ValueError("unsupported source capture format version")
    if payload["reader_id"] != _reader_identity():
        raise ValueError("source capture reader identity is incompatible")
    values = _fields(
        payload["spec"],
        {
            "candidate_limit",
            "admitted_byte_limit",
            "revision",
            "dataset_name",
            "max_sentence_bytes",
            "engine_version",
            "split_seed",
            "validation_buckets",
            "test_buckets",
        },
    )
    if values["engine_version"] != SegmentationPolicy.engine_version:
        raise ValueError("source capture segmentation version is incompatible")
    spec = PreparationSpec(
        candidate_limit=_integer(values["candidate_limit"]),
        admitted_byte_limit=_integer(values["admitted_byte_limit"]),
        revision=_text(values["revision"]),
        dataset_name=None if values["dataset_name"] is None else _text(values["dataset_name"]),
        segmentation=SegmentationPolicy(_integer(values["max_sentence_bytes"])),
        splits=SplitPolicy(
            _integer(values["split_seed"]),
            _integer(values["validation_buckets"]),
            _integer(values["test_buckets"]),
        ),
    )
    if type(payload["candidates"]) is not list:
        raise ValueError("source capture candidates must be a list")
    candidates: list[SourceDocument] = []
    for raw in payload["candidates"]:
        row = _fields(raw, {"source_id", "text", "revision", "identity"})
        doc = SourceDocument(_text(row["source_id"]), _text(row["text"]), _text(row["revision"]))
        if doc.identity != row["identity"]:
            raise ValueError("source candidate identity does not match its content")
        candidates.append(doc)
    result = prepare_documents(candidates, spec)
    if result.candidates != tuple(candidates):
        raise ValueError("source capture contains unexamined candidate suffix")
    if result.identity != payload["corpus_id"] or _json_bytes(_summary(result)) != _json_bytes(
        payload["summary"]
    ):
        raise ValueError("source capture summary does not match independent replay")
    return result
