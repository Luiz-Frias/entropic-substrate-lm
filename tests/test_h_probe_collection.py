"""One real local worker lineage is collected without promoting partial evidence."""

import hashlib
import json
from pathlib import Path

import pytest
from tests.latent_usage_support import POSTERIOR_MEAN
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_execution import prepared

from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.ladder.run import object_worker
from entropic_substrate.ladder.run.h_probe_collection import collect_probe_terminal
from entropic_substrate.ladder.run.span_states import SpanPopulationGap
from entropic_substrate.ladder.run.worker_transport import publish_probe_request

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("pause", [None, ("prediction", 2)])
def test_real_worker_completion_and_pause_keep_their_distinct_meaning(
    tmp_path: Path, pause: tuple[str, int] | None
) -> None:
    request, inputs = prepared(tmp_path, pause=pause)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    destination, worker = tmp_path / "collected", tmp_path / "executed"
    destination.mkdir()
    worker.mkdir()
    assert (
        collect_probe_terminal(
            objects, request, destination, expected_pod_id=None, max_marker_bytes=1_000_000
        )
        is None
    )
    assert not list(destination.iterdir())
    object_worker.run_object_worker(
        objects, marker, worker, max_request_bytes=1_000_000, latent_usage=POSTERIOR_MEAN
    )
    report = collect_probe_terminal(
        objects, request, destination, expected_pod_id=None, max_marker_bytes=1_000_000
    )
    assert report["workload_verified"] is (pause is None)
    assert report["reported_optimizer_steps"] == (6 if pause is None else 4)
    assert report["reported_outcome"] == ("probe_workload_complete" if pause is None else "paused")
    assert report["scientific_authority"] == "none" and (destination / "collection.json").is_file()


@pytest.mark.parametrize("kind", ["execution_defect", "scientific_gap", "diagnostic_gap"])
def test_reported_failure_kinds_survive_capture_without_claiming_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker, destination = tmp_path / "failed", tmp_path / "collected"
    worker.mkdir()
    destination.mkdir()
    error = {
        "execution_defect": RuntimeError("injected outcome"),
        "scientific_gap": UndefinedDirection(1),
        "diagnostic_gap": SpanPopulationGap({2: 1}, {2: 0}),
    }[kind]

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise error

    monkeypatch.setattr(object_worker, "execute_probe_request", fail)
    with pytest.raises(type(error)):
        object_worker.run_object_worker(
            objects, marker, worker, max_request_bytes=1_000_000, latent_usage=POSTERIOR_MEAN
        )
    report = collect_probe_terminal(
        objects, request, destination, expected_pod_id=None, max_marker_bytes=1_000_000
    )
    assert report["reported_outcome"] == kind and report["workload_verified"] is False
    assert report["capture_status"] == "failure_bytes_verified"
    assert report["reported_optimizer_steps"] is None
    assert (destination / "worker/failure.json").is_file()


def test_multiple_terminal_claims_cannot_be_selected_by_preference(tmp_path: Path) -> None:
    from tests.test_h_probe_events import event, store

    request, _ = prepared(tmp_path)
    failed = event(
        request,
        "failure",
        status="execution_defect",
        stage="execution",
        error_type="RuntimeError",
        error_code="fixture",
        evidence_delivery="failed",
    )
    failed.update(snapshot=None, delivery_error_type="OSError")
    objects = store(request, "failure", failed)
    paused = event(
        request,
        "paused",
        status="paused",
        result={"path": "worker/result.json", "sha256": "3" * 64},
    )
    objects.values.update(store(request, "paused", paused).values)
    destination = tmp_path / "collected"
    destination.mkdir()
    with pytest.raises(ValueError, match="conflicting"):
        collect_probe_terminal(
            objects, request, destination, expected_pod_id=None, max_marker_bytes=1_000_000
        )
    assert not list(destination.iterdir())


def test_missing_failure_delivery_is_an_explicit_gap_not_an_empty_success(tmp_path: Path) -> None:
    from tests.test_h_probe_events import event, store

    request, _ = prepared(tmp_path)
    failed = event(
        request,
        "failure",
        status="scientific_gap",
        stage="execution",
        error_type="UndefinedDirection",
        error_code="fixture",
        evidence_delivery="failed",
    )
    failed.update(snapshot=None, delivery_error_type="OSError")
    destination = tmp_path / "collected"
    destination.mkdir()
    report = collect_probe_terminal(
        store(request, "failure", failed),
        request,
        destination,
        expected_pod_id=None,
        max_marker_bytes=1_000_000,
    )
    assert (
        report["capture_status"] == "missing_failure_snapshot" and not report["workload_verified"]
    )


@pytest.mark.parametrize("tamper", ["missing", "foreign_request"])
def test_failed_recovery_retains_terminal_bytes_without_a_collection_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker, destination = tmp_path / "failed", tmp_path / "collected"
    worker.mkdir()
    destination.mkdir()

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("injected outcome")

    monkeypatch.setattr(object_worker, "execute_probe_request", fail)
    with pytest.raises(RuntimeError):
        object_worker.run_object_worker(
            objects, marker, worker, max_request_bytes=1_000_000, latent_usage=POSTERIOR_MEAN
        )
    key = f"{request.prefix}/outputs/worker/failure.json"
    if tamper == "missing":
        del objects.values[key]
    else:

        def encode(value: object) -> bytes:
            return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

        body = json.loads(objects.values[key])
        body["request_sha256"] = "0" * 64
        objects.values[key] = encode(body)
        event_key = f"{request.prefix}/failure.json"
        event = json.loads(objects.values[event_key])
        snapshot_key = event["snapshot"]["key"]
        snapshot = json.loads(objects.values[snapshot_key])
        snapshot["files"][0]["sha256"] = hashlib.sha256(objects.values[key]).hexdigest()
        objects.values[snapshot_key] = encode(snapshot)
        event["snapshot"]["sha256"] = hashlib.sha256(objects.values[snapshot_key]).hexdigest()
        objects.values[event_key] = encode(event)
    with pytest.raises(ValueError):
        collect_probe_terminal(
            objects, request, destination, expected_pod_id=None, max_marker_bytes=1_000_000
        )
    assert (destination / "terminal.json").read_bytes() == objects.values[
        f"{request.prefix}/failure.json"
    ]
    assert not (destination / "collection.json").exists()


def test_pause_cannot_offer_an_uncaptured_resume_artifact(tmp_path: Path) -> None:
    from dataclasses import asdict

    from tests.test_h_probe_events import event, store

    from entropic_substrate.ladder.run.journal import write_json
    from entropic_substrate.ladder.run.worker_snapshot import publish_worker_snapshot

    request, _ = prepared(tmp_path, pause=("prediction", 2))
    producer, destination = tmp_path / "producer", tmp_path / "collected"
    (producer / "worker").mkdir(parents=True)
    destination.mkdir()
    result = {
        "schema": "h-probe-worker-result-v1",
        "request_sha256": request.digest,
        "run_identity": request.profile.identity,
        "scientific_authority": "none",
        "status": "paused",
        "runtime": {"kind": "local_tiny", "pod_id": None},
        "successful_optimizer_steps": 4,
        "training": {"path": "run/missing.json", "sha256": "a" * 64},
        "generation": None,
        "diagnostics": None,
        "capture": None,
        "verification": None,
        "wall_seconds": 0,
    }
    reference = write_json(producer, "worker/result.json", result, max_bytes=10_000)
    objects = _ObjectStore()
    snapshot = publish_worker_snapshot(
        objects, request, producer, ("worker/result.json",), name="paused"
    )
    terminal = event(
        request,
        "paused",
        status="paused",
        result={"path": "worker/result.json", "sha256": reference.content_hash.digest},
    )
    terminal["snapshot"] = asdict(snapshot)
    objects.values.update(store(request, "paused", terminal).values)
    with pytest.raises(ValueError):
        collect_probe_terminal(
            objects, request, destination, expected_pod_id=None, max_marker_bytes=1_000_000
        )
    assert not (destination / "collection.json").exists()
