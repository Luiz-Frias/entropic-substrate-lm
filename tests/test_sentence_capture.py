"""Local source captures are replayed and hash-bound, never self-attested."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.evidence.lifecycle import ContentHash
from entropic_substrate.sentence_prediction.capture import (
    read_prepared_corpus,
    write_prepared_corpus,
)
from entropic_substrate.sentence_prediction.corpus import (
    PreparationSpec,
    PreparedCorpus,
    prepare_documents,
)
from entropic_substrate.sentence_prediction.records import SourceDocument

pytestmark = pytest.mark.unit


def corpus() -> PreparedCorpus:
    source = SourceDocument("doc", "  One. Next.\n", "fixture-v1")
    return prepare_documents(
        [source, replace(source, source_id="alias")], PreparationSpec(3, 100, "fixture-v1")
    )


def test_capture_roundtrips_candidates_rejections_and_source_offsets(tmp_path: Path) -> None:
    expected = corpus()
    artifact = write_prepared_corpus(tmp_path, expected, max_bytes=10000)
    actual = read_prepared_corpus(tmp_path, artifact.content_hash, max_bytes=10000)
    assert actual == expected
    assert actual.rejections[0].reason == "duplicate_content"
    assert actual.records[0].segmentation.spans[0].source == b"  One. "


def test_capture_never_replaces_an_existing_target(tmp_path: Path) -> None:
    first = write_prepared_corpus(tmp_path, corpus(), max_bytes=10000)
    with pytest.raises(ValueError, match="occupied"):
        write_prepared_corpus(tmp_path, corpus(), max_bytes=10000)
    assert read_prepared_corpus(tmp_path, first.content_hash, max_bytes=10000) == corpus()


def test_capture_refuses_forged_preparation_instead_of_trusting_its_type(tmp_path: Path) -> None:
    prepared = corpus()
    forged = replace(prepared, records=())
    with pytest.raises(ValueError, match="replay"):
        write_prepared_corpus(tmp_path, forged, max_bytes=10000)
    assert list(tmp_path.iterdir()) == []


def test_capture_refuses_wrong_expected_digest(tmp_path: Path) -> None:
    write_prepared_corpus(tmp_path, corpus(), max_bytes=10000)
    with pytest.raises(ValueError, match="digest"):
        read_prepared_corpus(tmp_path, ContentHash("sha256", "0" * 64), max_bytes=10000)


@pytest.mark.parametrize(
    "field,value",
    [
        ("format_version", "unknown"),
        ("reader_id", "different"),
        ("corpus_id", "wrong"),
        ("summary", {"status": "prepared"}),
    ],
)
def test_capture_refuses_rehashed_incompatible_or_forged_payload(
    tmp_path: Path, field: str, value: object
) -> None:
    artifact = write_prepared_corpus(tmp_path, corpus(), max_bytes=10000)
    path = tmp_path / "prepared-corpus.json"
    payload = json.loads(path.read_bytes())
    payload[field] = value
    encoded = json.dumps(payload).encode()
    path.write_bytes(encoded)
    tampered = ContentHash("sha256", hashlib.sha256(encoded).hexdigest())
    assert tampered != artifact.content_hash
    with pytest.raises(ValueError):
        read_prepared_corpus(tmp_path, tampered, max_bytes=10000)


@pytest.mark.parametrize("field", ["max_sentence_bytes", "dataset_name"])
def test_missing_policy_field_is_not_filled_from_a_runtime_default(
    tmp_path: Path, field: str
) -> None:
    write_prepared_corpus(tmp_path, corpus(), max_bytes=10000)
    path = tmp_path / "prepared-corpus.json"
    payload = json.loads(path.read_bytes())
    del payload["spec"][field]
    encoded = json.dumps(payload).encode()
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match="fields"):
        read_prepared_corpus(
            tmp_path, ContentHash("sha256", hashlib.sha256(encoded).hexdigest()), max_bytes=10000
        )


def test_duplicate_json_keys_are_not_last_value_wins(tmp_path: Path) -> None:
    path = tmp_path / "prepared-corpus.json"
    payload = b'{"format_version":"a","format_version":"b"}'
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="duplicate"):
        read_prepared_corpus(
            tmp_path, ContentHash("sha256", hashlib.sha256(payload).hexdigest()), max_bytes=10000
        )


def test_capture_byte_limit_is_checked_before_publication(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit"):
        write_prepared_corpus(tmp_path, corpus(), max_bytes=10)
    assert list(tmp_path.iterdir()) == []


def test_capture_refuses_symlinked_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    artifact = write_prepared_corpus(root, corpus(), max_bytes=10000)
    alternate = tmp_path / "alias"
    alternate.mkdir()
    (alternate / "prepared-corpus.json").symlink_to(root / "prepared-corpus.json")
    with pytest.raises(ValueError):
        read_prepared_corpus(alternate, artifact.content_hash, max_bytes=10000)


def test_reader_refuses_non_sha256_identity_before_file_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        read_prepared_corpus(tmp_path, ContentHash("md5", "0" * 32), max_bytes=10000)


def test_reader_byte_limit_is_independent_of_writer_limit(tmp_path: Path) -> None:
    artifact = write_prepared_corpus(tmp_path, corpus(), max_bytes=10000)
    with pytest.raises(ValueError, match="byte limit"):
        read_prepared_corpus(tmp_path, artifact.content_hash, max_bytes=10)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_reader_refuses_nonfinite_json_constants(tmp_path: Path, constant: str) -> None:
    encoded = f'{{"value":{constant}}}'.encode()
    (tmp_path / "prepared-corpus.json").write_bytes(encoded)
    with pytest.raises(ValueError, match="nonfinite JSON"):
        read_prepared_corpus(
            tmp_path, ContentHash("sha256", hashlib.sha256(encoded).hexdigest()), max_bytes=10000
        )


def test_rehashed_candidate_identity_must_still_match_its_bytes(tmp_path: Path) -> None:
    write_prepared_corpus(tmp_path, corpus(), max_bytes=10000)
    path = tmp_path / "prepared-corpus.json"
    payload = json.loads(path.read_bytes())
    payload["candidates"][0]["identity"] = "forged"
    encoded = json.dumps(payload).encode()
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match="candidate identity"):
        read_prepared_corpus(
            tmp_path, ContentHash("sha256", hashlib.sha256(encoded).hexdigest()), max_bytes=10000
        )


def test_valid_prefix_summary_cannot_hide_an_unexamined_suffix(tmp_path: Path) -> None:
    complete = corpus()
    prefix = prepare_documents(complete.candidates, replace(complete.spec, candidate_limit=1))
    write_prepared_corpus(tmp_path, prefix, max_bytes=10000)
    path = tmp_path / "prepared-corpus.json"
    payload = json.loads(path.read_bytes())
    extra = complete.candidates[1]
    payload["candidates"].append(
        {
            "source_id": extra.source_id,
            "text": extra.text,
            "revision": extra.revision,
            "identity": extra.identity,
        }
    )
    encoded = json.dumps(payload).encode()
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match="unexamined candidate suffix"):
        read_prepared_corpus(
            tmp_path, ContentHash("sha256", hashlib.sha256(encoded).hexdigest()), max_bytes=10000
        )


def test_summary_booleans_cannot_impersonate_integer_offsets(tmp_path: Path) -> None:
    write_prepared_corpus(tmp_path, corpus(), max_bytes=10000)
    path = tmp_path / "prepared-corpus.json"
    payload = json.loads(path.read_bytes())
    assert payload["summary"]["records"][0]["spans"][0][0] == 0
    payload["summary"]["records"][0]["spans"][0][0] = False
    encoded = json.dumps(payload).encode()
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match="independent replay"):
        read_prepared_corpus(
            tmp_path, ContentHash("sha256", hashlib.sha256(encoded).hexdigest()), max_bytes=10000
        )


def test_declared_dataset_roundtrips_and_changes_empty_corpus_identity(tmp_path: Path) -> None:
    first = replace(corpus().spec, dataset_name="sample-10BT")
    second = replace(first, dataset_name="sample-100BT")
    prepared = prepare_documents([], first)
    assert prepared.identity != prepare_documents([], second).identity
    artifact = write_prepared_corpus(tmp_path, prepared, max_bytes=10000)
    assert read_prepared_corpus(tmp_path, artifact.content_hash, max_bytes=10000).spec == first
    payload = json.loads((tmp_path / "prepared-corpus.json").read_bytes())
    assert payload["format_version"] == "h-source-capture-v2"
    assert payload["spec"]["dataset_name"] == "sample-10BT"


@pytest.mark.parametrize("name", ["", "  ", " sample", 42, False])
def test_declared_dataset_name_cannot_be_blank_or_coerced(name: object) -> None:
    with pytest.raises(ValueError, match="dataset"):
        replace(corpus().spec, dataset_name=name)


def test_legacy_capture_requires_original_reader_without_invented_provenance(
    tmp_path: Path,
) -> None:
    write_prepared_corpus(tmp_path, corpus(), max_bytes=10000)
    target = tmp_path / "prepared-corpus.json"
    payload = json.loads(target.read_bytes())
    payload["format_version"] = "h-source-capture-v1"
    payload["spec"].pop("dataset_name", None)
    encoded = json.dumps(payload).encode()
    target.write_bytes(encoded)
    with pytest.raises(ValueError, match="original reader"):
        read_prepared_corpus(
            tmp_path, ContentHash("sha256", hashlib.sha256(encoded).hexdigest()), max_bytes=10000
        )
