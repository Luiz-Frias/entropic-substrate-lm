"""Host completion admission binds transferred bytes, native receipts, and request."""

import json
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_execution import prepared

from entropic_substrate.sentence_prediction.object_worker import run_object_worker
from entropic_substrate.sentence_prediction.worker_completion import receive_worker_completion
from entropic_substrate.sentence_prediction.worker_transport import publish_probe_request

pytestmark = pytest.mark.unit


def test_external_completion_admission_matches_one_worker_lineage(tmp_path: Path) -> None:
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    terminal = run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    host = tmp_path / "host"
    host.mkdir()
    admitted = receive_worker_completion(
        objects, request, terminal, host, expected_pod_id=None, max_marker_bytes=1_000_000
    )
    assert admitted["status"] == "probe_completion_verified"
    assert admitted["scientific_authority"] == "none"
    assert admitted["native_validation"] == "bound_executor_receipt_not_host_rerun"
    assert admitted["successful_optimizer_steps"] == 6
    original = objects.values[terminal.key]
    raw = json.loads(original)
    raw["request_sha256"] = "0" * 64
    forged = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    objects.values[terminal.key] = forged
    wrong = objects._identity(terminal.key, forged)
    empty = tmp_path / "wrong"
    empty.mkdir()
    with pytest.raises(ValueError, match="identity"):
        receive_worker_completion(
            objects, request, wrong, empty, expected_pod_id=None, max_marker_bytes=1_000_000
        )
    assert not list(empty.iterdir())
    objects.values[terminal.key] = original
    verification_key = next(key for key in objects.values if "/outputs/verification/" in key)
    del objects.values[verification_key]
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(ValueError, match="identity"):
        receive_worker_completion(
            objects, request, terminal, missing, expected_pod_id=None, max_marker_bytes=1_000_000
        )
    assert not (missing / "snapshot.json").exists()
