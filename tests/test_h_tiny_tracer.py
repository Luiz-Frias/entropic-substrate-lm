"""Local end-to-end tiny-probe tracer: a remote_tiny profile shape on this host.

The tracer exists to reproduce a remote probe's numerical path locally with the
observability mirror wired exactly as the H worker wires it. It must change only
the execution regime (run class, device, optional precision, re-sealed source
digest), never a scientific field, and it must own its sink's lifecycle.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tests.test_sentence_profile import profile
from tests.test_sentence_sampling import corpus

from entropic_substrate.composition.wiring.h_tiny_tracer import (
    localize_profile,
    main,
    reseal_foreign_corpus,
    run_tiny_trace,
)
from entropic_substrate.core.sentence.corpus import SplitPolicy, prepare_documents
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.kernel.lifecycle import ContentHash
from entropic_substrate.ladder.capture.capture import read_prepared_corpus, write_prepared_corpus
from entropic_substrate.ladder.capture.ports.run_descriptor import TrackingRunDescriptor
from entropic_substrate.ladder.run.execution import SentenceExecution

pytestmark = pytest.mark.unit

LIMIT = 10_000_000
_REGIME_FIELDS = {"run_class", "device", "precision", "source_digest"}


def _remote(value: HProfile) -> HProfile:
    manifest = value.manifest | {"run_class": "remote_tiny", "device": "cuda", "precision": "bf16"}
    return HProfile.from_manifest(manifest)


def _foreign_capture(root: Path) -> tuple[Path, str]:
    """A v2 capture whose reader identity belongs to another regime."""
    local = root / "local-source"
    local.mkdir()
    original = corpus()
    prepared = prepare_documents(
        original.candidates,
        replace(original.spec, splits=SplitPolicy(validation_buckets=2000, test_buckets=1000)),
    )
    write_prepared_corpus(local, prepared, max_bytes=LIMIT)
    payload = json.loads((local / "prepared-corpus.json").read_text())
    payload["reader_id"] = "f" * 64
    foreign = root / "foreign" / "prepared-corpus.json"
    foreign.parent.mkdir()
    foreign.write_text(json.dumps(payload))
    return foreign, prepared.identity


class _FakeSink:
    def __init__(self) -> None:
        self.records: list[Mapping[str, Any]] = []
        self.finished = False

    def log(self, record: Mapping[str, Any]) -> None:
        self.records.append(record)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        raise AssertionError("the tracer never uploads artifacts")

    def finish(self) -> None:
        self.finished = True


class _Builder:
    """Records exactly how the tracer composes its sink."""

    def __init__(self) -> None:
        self.sink = _FakeSink()
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, descriptor: TrackingRunDescriptor, *, results_path: Path, metric_extractor: Any
    ) -> _FakeSink:
        self.calls.append(
            {"descriptor": descriptor, "results_path": results_path, "extractor": metric_extractor}
        )
        return self.sink


def test_localize_changes_only_the_execution_regime() -> None:
    remote = _remote(profile())
    local = localize_profile(remote, source_digest="a" * 64)
    assert (local.run_class, local.device, local.precision) == ("local_tiny", "cpu", "bf16")
    assert local.source_digest == "a" * 64
    kept = {k: v for k, v in remote.manifest.items() if k not in _REGIME_FIELDS}
    assert {k: v for k, v in local.manifest.items() if k not in _REGIME_FIELDS} == kept
    # The run identity hashes the manifest, so a localized run is a new identity.
    assert local.identity != remote.identity
    assert localize_profile(remote, source_digest="a" * 64, precision="fp32").precision == "fp32"


def test_localize_refuses_non_tiny_authority() -> None:
    pilot = HProfile.from_manifest(
        _remote(profile()).manifest | {"name": "h0-fisher-pilot", "run_class": "pilot"}
    )
    with pytest.raises(ValueError, match="tiny"):
        localize_profile(pilot, source_digest="a" * 64)


def test_reseal_replays_foreign_candidates_under_the_local_reader(tmp_path: Path) -> None:
    foreign, identity = _foreign_capture(tmp_path)
    # Control: the local reader refuses the foreign regime's bytes as-is.
    with pytest.raises(ValueError, match="reader identity"):
        _read_foreign(foreign)
    output = tmp_path / "source"
    output.mkdir()
    artifact = reseal_foreign_corpus(foreign, output, max_bytes=LIMIT)
    replayed = read_prepared_corpus(output, artifact.content_hash, max_bytes=LIMIT)
    assert replayed.identity == identity


def _read_foreign(foreign: Path) -> None:
    digest = hashlib.sha256(foreign.read_bytes()).hexdigest()
    read_prepared_corpus(foreign.parent, ContentHash("sha256", digest), max_bytes=LIMIT)


@pytest.mark.parametrize("field", ["corpus_id", "text", "summary"])
def test_reseal_refuses_a_capture_that_does_not_replay(tmp_path: Path, field: str) -> None:
    foreign, _ = _foreign_capture(tmp_path)
    payload = json.loads(foreign.read_text())
    if field == "corpus_id":
        payload["corpus_id"] = "0" * 64
    elif field == "text":
        payload["candidates"][0]["text"] += " tampered"
    else:
        payload["summary"]["admitted_bytes"] += 1
    foreign.write_text(json.dumps(payload))
    output = tmp_path / "source"
    output.mkdir()
    with pytest.raises(ValueError):
        reseal_foreign_corpus(foreign, output, max_bytes=LIMIT)
    assert not tuple(output.iterdir()), "a refused reseal must not publish a capture"


def _local_profile(tmp_path: Path) -> tuple[HProfile, Path]:
    foreign, _ = _foreign_capture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    artifact = reseal_foreign_corpus(foreign, source, max_bytes=LIMIT)
    return localize_profile(
        _remote(profile()), source_digest=artifact.content_hash.digest, precision="fp32"
    ), source


def test_trace_routes_every_row_through_the_worker_shaped_sink(tmp_path: Path) -> None:
    local, source = _local_profile(tmp_path)
    builder = _Builder()
    result = run_tiny_trace(
        local,
        source,
        tmp_path / "trace",
        max_source_bytes=LIMIT,
        max_artifact_bytes=LIMIT,
        sink_builder=builder,
    )
    assert result.status == "training_complete"
    (call,) = builder.calls
    assert call["results_path"] == tmp_path / "trace" / "observability" / "metrics.jsonl"
    assert call["descriptor"].tiny is True
    training = [r for r in builder.sink.records if r["schema"] == "h-training-step-v2"]
    assert [(r["phase"], r["step"]) for r in training] == [
        ("codec_warmup", 1),
        ("codec_warmup", 2),
        *[("prediction", s) for s in range(1, 5)],
    ]
    # One global optimizer-step axis: prediction step n lands at warmup + n.
    steps = [call["extractor"].extract(r).step for r in training]
    assert steps == [1, 2, 3, 4, 5, 6]
    assert builder.sink.finished


def test_trace_finishes_its_sink_when_training_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local, source = _local_profile(tmp_path)
    original = SentenceExecution.advance

    def fail_in_prediction(self: SentenceExecution) -> dict[str, object]:
        if self.phase == "prediction":
            raise RuntimeError("injected prediction failure")
        return original(self)

    monkeypatch.setattr(SentenceExecution, "advance", fail_in_prediction)
    builder = _Builder()
    with pytest.raises(RuntimeError, match="injected"):
        run_tiny_trace(
            local,
            source,
            tmp_path / "trace",
            max_source_bytes=LIMIT,
            max_artifact_bytes=LIMIT,
            sink_builder=builder,
        )
    assert builder.sink.finished
    assert len(list((tmp_path / "trace" / "run").glob("*-failure.json"))) == 1


def test_command_writes_source_run_observability_and_trace_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    foreign, identity = _foreign_capture(tmp_path)
    remote = tmp_path / "remote-profile.json"
    remote.write_text(json.dumps(_remote(profile()).manifest))
    out = tmp_path / "trace"
    status = main(
        [
            "--profile",
            str(remote),
            "--corpus",
            str(foreign),
            "--out",
            str(out),
            "--precision",
            "fp32",
            "--max-source-bytes",
            str(LIMIT),
            "--max-artifact-bytes",
            str(LIMIT),
        ]
    )
    assert status == 0
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "training_complete"
    manifest = json.loads((out / "trace.json").read_text())
    assert manifest["schema"] == "h-tiny-trace-v1"
    assert manifest["corpus_identity"] == identity
    assert manifest["scientific_authority"] == "none"
    rows = [json.loads(line) for line in (out / "observability" / "metrics.jsonl").open()]
    assert sum(r["schema"] == "h-training-step-v2" for r in rows) == 6
    assert (out / "source" / "prepared-corpus.json").is_file()


def test_command_reports_a_training_failure_as_an_execution_defect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    original = SentenceExecution.advance

    def fail_in_prediction(self: SentenceExecution) -> dict[str, object]:
        if self.phase == "prediction":
            raise RuntimeError("injected prediction failure")
        return original(self)

    monkeypatch.setattr(SentenceExecution, "advance", fail_in_prediction)
    foreign, _ = _foreign_capture(tmp_path)
    remote = tmp_path / "remote-profile.json"
    remote.write_text(json.dumps(_remote(profile()).manifest))
    status = main(
        ["--profile", str(remote), "--corpus", str(foreign), "--out", str(tmp_path / "t")]
    )
    assert status == 1
    report = json.loads((tmp_path / "t" / "result.json").read_text())
    assert report["status"] == "execution_defect" and report["error_type"] == "RuntimeError"
    assert report["failure"].endswith("-failure.json")
