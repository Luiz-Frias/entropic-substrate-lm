"""Source preparation commands preserve documents and require explicit policy."""

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock

import pytest

from entropic_substrate.cli import main
from entropic_substrate.data.fineweb import FINEWEB_REVISION
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.sentence_prediction import source_cli
from entropic_substrate.sentence_prediction.capture import read_prepared_corpus
from entropic_substrate.sentence_prediction.corpus import PreparedCorpus
from entropic_substrate.sentence_prediction.journal import write_json
from entropic_substrate.sentence_prediction.records import SourceDocument

pytestmark = pytest.mark.unit


def manifest() -> dict[str, object]:
    return {
        "schema": "h-prepare-source-command-v1",
        "dataset_name": "sample-10BT",
        "revision": FINEWEB_REVISION,
        "output_root": "source",
        "max_artifact_bytes": 1_000_000,
        "candidate_limit": 4,
        "admitted_byte_limit": 10_000,
        "max_sentence_bytes": 64,
        "segmentation_version": "0.3.4",
        "split_seed": 42,
        "validation_buckets": 500,
        "test_buckets": 500,
    }


def test_prepare_and_inspect_source_use_the_same_captured_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    documents = (
        SourceDocument("one", "  Café. Next sentence.\n", FINEWEB_REVISION),
        SourceDocument("alias", "  Café. Next sentence.\n", FINEWEB_REVISION),
        SourceDocument("oversize", "x" * 65 + ".", FINEWEB_REVISION),
    )
    stream = Mock(return_value=(document for document in documents))
    monkeypatch.setattr("entropic_substrate.data.fineweb.stream_fineweb_documents", stream)
    (tmp_path / "source").mkdir()
    write_json(tmp_path, "prepare.json", manifest(), max_bytes=10_000)
    assert main(["h-prepare-source", "--root", str(tmp_path), "--manifest", "prepare.json"]) == 0
    result = json.loads(capsys.readouterr().out)
    stream.assert_called_once_with(max_documents=4, name="sample-10BT")
    assert result["status"] == "source_verified" and result["scientific_authority"] == "none"
    assert result["examined_documents"] == 3
    assert result["admitted_documents"] == 1 and result["rejected_documents"] == 2
    assert result["stopped_by"] == "source_exhausted"
    captured = read_prepared_corpus(
        tmp_path / "source", ContentHash("sha256", result["source"]["sha256"]), max_bytes=1_000_000
    )
    assert captured.candidates == documents
    assert b"".join(s.source for s in captured.records[0].segmentation.spans) == documents[0].source
    inspect = {
        "schema": "h-inspect-source-command-v1",
        "source_root": "source",
        "source": result["source"],
        "max_artifact_bytes": 1_000_000,
    }
    write_json(tmp_path, "inspect.json", inspect, max_bytes=10_000)
    assert main(["h-inspect-source", "--root", str(tmp_path), "--manifest", "inspect.json"]) == 0
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize(
    "change",
    [
        {"revision": "main"},
        {"segmentation_version": "unknown"},
        {"candidate_limit": 0},
        {"candidate_limit": True},
        {"max_artifact_bytes": 0},
        {"split_seed": None},
        {"output_root": "../outside"},
        {"extra": "ignored?"},
        {"validation_buckets": 10000},
        {"dataset_name": " "},
    ],
)
def test_invalid_source_policy_fails_before_network_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    change: dict[str, object],
) -> None:
    stream = Mock(side_effect=AssertionError("invalid request reached provider"))
    monkeypatch.setattr("entropic_substrate.data.fineweb.stream_fineweb_documents", stream)
    (tmp_path / "source").mkdir()
    write_json(tmp_path, "bad.json", manifest() | change, max_bytes=10_000)
    assert main(["h-prepare-source", "--root", str(tmp_path), "--manifest", "bad.json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_request"
    stream.assert_not_called()
    assert not tuple((tmp_path / "source").iterdir())


@pytest.mark.parametrize("field", tuple(manifest()))
def test_missing_policy_field_is_not_filled_from_defaults(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str
) -> None:
    request = manifest()
    del request[field]
    write_json(tmp_path, "missing.json", request, max_bytes=10_000)
    assert main(["h-prepare-source", "--root", str(tmp_path), "--manifest", "missing.json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_request"


def test_empty_source_is_retained_as_empty_not_model_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "entropic_substrate.data.fineweb.stream_fineweb_documents",
        Mock(return_value=(d for d in ())),
    )
    (tmp_path / "source").mkdir()
    write_json(tmp_path, "empty.json", manifest(), max_bytes=10_000)
    assert main(["h-prepare-source", "--root", str(tmp_path), "--manifest", "empty.json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["corpus_status"] == "empty"
    assert result["admitted_documents"] == result["examined_documents"] == 0
    assert result["scientific_authority"] == "none"


def test_empty_dataset_configurations_retain_distinct_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stream = Mock(side_effect=lambda **_: (d for d in ()))
    monkeypatch.setattr("entropic_substrate.data.fineweb.stream_fineweb_documents", stream)
    identities = []
    for name in ("sample-10BT", "sample-100BT"):
        output = tmp_path / name
        output.mkdir()
        command = manifest() | {"dataset_name": name, "output_root": name}
        filename = f"{name}.json"
        write_json(tmp_path, filename, command, max_bytes=10_000)
        assert main(["h-prepare-source", "--root", str(tmp_path), "--manifest", filename]) == 0
        result = json.loads(capsys.readouterr().out)
        identities.append((result["source"]["sha256"], result["corpus_identity"]))
    assert identities[0][0] != identities[1][0]
    assert identities[0][1] != identities[1][1]


def test_existing_source_is_not_overwritten_or_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stream = Mock(side_effect=AssertionError("nonempty destination reached provider"))
    monkeypatch.setattr("entropic_substrate.data.fineweb.stream_fineweb_documents", stream)
    source = tmp_path / "source"
    source.mkdir()
    sentinel = source / "prepared-corpus.json"
    sentinel.write_bytes(b"preserve existing evidence")
    write_json(tmp_path, "existing.json", manifest(), max_bytes=10_000)
    assert main(["h-prepare-source", "--root", str(tmp_path), "--manifest", "existing.json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_request"
    assert sentinel.read_bytes() == b"preserve existing evidence"
    stream.assert_not_called()


@pytest.mark.parametrize("provider_failure", [False, True])
def test_failed_preparation_does_not_publish_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider_failure: bool,
) -> None:
    stream = (
        Mock(side_effect=OSError("provider details must remain private"))
        if provider_failure
        else Mock(return_value=(d for d in ()))
    )
    monkeypatch.setattr("entropic_substrate.data.fineweb.stream_fineweb_documents", stream)
    source = tmp_path / "source"
    source.mkdir()
    write_json(tmp_path, "failed.json", manifest() | {"max_artifact_bytes": 1}, max_bytes=10_000)
    assert main(["h-prepare-source", "--root", str(tmp_path), "--manifest", "failed.json"]) == 1
    output = capsys.readouterr().out
    assert json.loads(output)["status"] == "execution_defect"
    assert "provider details" not in output and not tuple(source.iterdir())


@pytest.mark.parametrize("stop", ["limit", "budget", "invalid_revision", "close_error"])
def test_source_command_closes_owned_stream_before_publication_or_error_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stop: str,
) -> None:
    closed = []

    def rows() -> Generator[SourceDocument, None, None]:
        try:
            for index in range(5):
                yield SourceDocument(
                    str(index),
                    f"Sentence {index}.",
                    "wrong" if stop == "invalid_revision" else FINEWEB_REVISION,
                )
        finally:
            closed.append(True)
            if stop == "close_error":
                raise OSError("owned provider stream failed to close")

    iterator = rows()
    monkeypatch.setattr(
        "entropic_substrate.data.fineweb.stream_fineweb_documents", Mock(return_value=iterator)
    )
    original_write = source_cli.write_prepared_corpus

    def checked_write(root: Path, corpus: PreparedCorpus, *, max_bytes: int) -> ArtifactIdentity:
        assert closed == [True], "source published before closing its owned stream"
        return original_write(root, corpus, max_bytes=max_bytes)

    monkeypatch.setattr(source_cli, "write_prepared_corpus", checked_write)
    (tmp_path / "source").mkdir()
    command = manifest() | ({"admitted_byte_limit": 1} if stop == "budget" else {})
    write_json(tmp_path, "request.json", command, max_bytes=10_000)
    failed = stop in ("invalid_revision", "close_error")
    try:
        assert main(
            ["h-prepare-source", "--root", str(tmp_path), "--manifest", "request.json"]
        ) == (1 if failed else 0)
        assert closed == [True]
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == ("execution_defect" if failed else "source_verified")
        if failed:
            assert not list((tmp_path / "source").iterdir())
    finally:
        try:
            iterator.close()
        except OSError:
            assert stop == "close_error"  # Only dispose this test's retained negative fixture.
