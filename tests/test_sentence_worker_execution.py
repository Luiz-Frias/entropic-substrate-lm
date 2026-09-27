"""One materialized request traverses actual training and observer stages."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from shutil import copytree

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_span_state_collection import model

from entropic_substrate.core.probe.predictive_probe import ProbeConfig
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.gate.ladder_campaign import artifact_from_mapping
from entropic_substrate.ladder.run.journal import read_json
from entropic_substrate.ladder.run.worker_execution import execute_probe_request
from entropic_substrate.ladder.run.worker_request import ProbeRequest, TransferLimits
from entropic_substrate.ladder.run.worker_transport import (
    materialize_probe_request,
    publish_probe_request,
)

pytestmark = pytest.mark.unit


def prepared(root: Path, *, pause: tuple[str, int] | None = None) -> tuple[ProbeRequest, Path]:
    template = model(root)
    source = ArtifactIdentity(
        "source/prepared-corpus.json", ContentHash("sha256", template.profile.source_digest)
    )
    request = ProbeRequest(
        template.profile,
        (source,),
        "1" * 64,
        TransferLimits(10_000_000, 50_000_000, 100),
        ProbeConfig(train_positions=8, eval_positions=8, updates=2, batch_size=4),
        None,
        None,
        None,
        None,
        None,
    )
    request = replace(request, pause=pause)
    store = _ObjectStore()
    marker = publish_probe_request(store, request, root)
    worker = root / "worker"
    worker.mkdir()
    materialize_probe_request(store, marker, worker, max_request_bytes=1_000_000)
    return request, worker


def test_worker_runs_one_lineage_through_native_verification(tmp_path: Path) -> None:
    request, root = prepared(tmp_path)
    checkpoints = []
    result = execute_probe_request(request, root, on_checkpoint=checkpoints.append)
    assert result["status"] == "probe_workload_complete"
    assert result["scientific_authority"] == "none"
    assert result["successful_optimizer_steps"] == 6
    assert len(checkpoints) == 3
    native = read_json(root, artifact_from_mapping(result["verification"]), max_bytes=10_000_000)
    assert native["request_sha256"] == request.digest
    assert native["status"] == "native_reader_passed"
    capture = read_json(root, artifact_from_mapping(result["capture"]), max_bytes=10_000_000)
    assert capture["run_identity"] == request.profile.identity
    assert capture["diagnostic_roots"]


def test_worker_pause_stops_before_generation_or_diagnostics(tmp_path: Path) -> None:
    paused, root = prepared(tmp_path, pause=("prediction", 2))
    result = execute_probe_request(paused, root)
    assert result["status"] == "paused"
    assert result["successful_optimizer_steps"] == 4
    assert result["generation"] is None and result["capture"] is None
    assert not (root / "capture").exists()
    staging = tmp_path / "resume-inputs"
    staging.mkdir()
    copytree(root / "source", staging / "source")
    copytree(root / "run", staging / "run")
    artifacts = tuple(
        ArtifactIdentity(
            path.relative_to(staging).as_posix(),
            ContentHash("sha256", sha256(path.read_bytes()).hexdigest()),
        )
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    )
    resumed = replace(
        paused,
        inputs=artifacts,
        nonce="2" * 64,
        resume=artifact_from_mapping(result["training"]),
        pause=None,
    )
    store = _ObjectStore()
    marker = publish_probe_request(store, resumed, staging)
    destination = tmp_path / "resumed"
    destination.mkdir()
    materialize_probe_request(store, marker, destination, max_request_bytes=1_000_000)
    completed = execute_probe_request(resumed, destination)
    assert completed["status"] == "probe_workload_complete"
    assert completed["successful_optimizer_steps"] == 6
    final = read_json(
        destination, artifact_from_mapping(completed["training"]), max_bytes=10_000_000
    )
    assert final["parent_receipt"]["sha256"] == result["training"]["sha256"]


def test_checkpoint_observer_failure_cannot_be_reported_complete(tmp_path: Path) -> None:
    request, root = prepared(tmp_path)

    def fail(artifact: ArtifactIdentity) -> None:
        assert read_json(root, artifact, max_bytes=10_000_000)["status"] == "checkpoint"
        raise RuntimeError("checkpoint transport unavailable")

    with pytest.raises(RuntimeError, match="transport unavailable"):
        execute_probe_request(request, root, on_checkpoint=fail)
    failure = json.loads(next((root / "run").glob("*-failure.json")).read_text())
    assert failure["status"] == "execution_defect"
    assert failure["error_type"] == failure["error_code"] == "RuntimeError"
    assert not (root / "capture").exists()


def test_changed_materialized_request_never_starts_training(tmp_path: Path) -> None:
    request, root = prepared(tmp_path)
    (root / "request.json").write_bytes(request.payload + b" ")
    with pytest.raises(ValueError, match="identity"):
        execute_probe_request(request, root)
    assert not list((root / "run").iterdir())
