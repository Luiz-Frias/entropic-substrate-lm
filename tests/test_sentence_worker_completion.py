"""Host completion admission binds transferred bytes, native receipts, and request."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_execution import prepared
from tests.test_sentence_worker_request import request as local_request

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.object_worker import run_object_worker
from entropic_substrate.ladder.run.worker_completion import (
    admit_worker_runtime,
    receive_worker_completion,
)
from entropic_substrate.ladder.run.worker_transport import publish_probe_request

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
    rejected = tmp_path / "wrong-pod"
    rejected.mkdir()
    with pytest.raises(ValueError, match="provider allocation"):
        receive_worker_completion(
            objects,
            request,
            terminal,
            rejected,
            expected_pod_id="unexpected-pod",
            max_marker_bytes=1_000_000,
        )
    assert (rejected / "snapshot.json").is_file()
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


def test_remote_runtime_binding_is_a_metadata_contract_not_a_remote_run() -> None:
    local = local_request()
    bundle = ArtifactIdentity("code.bundle", ContentHash("sha256", "b" * 64))
    request = replace(
        local,
        profile=replace(local.profile, run_class="remote_tiny", device="cuda", precision="bf16"),
        inputs=(*local.inputs, bundle),
        code_bundle=bundle,
        code_commit="c" * 40,
        image="ghcr.io/luiz-frias/entropic-substrate-lm@sha256:" + "d" * 64,
    )
    runtime = {
        "kind": "remote_tiny",
        "pod_id": "fixture-pod",
        "source_attestation": "exact_git_tree_match",
        "code_commit": request.code_commit,
        "requested_image": request.image,
        "accelerator": {"model": "fixture-GPU", "count": 1},
        "precision": {},
    }
    assert admit_worker_runtime(runtime, request, "fixture-pod") == runtime
    for key, value in (
        ("pod_id", "other-pod"),
        ("code_commit", "e" * 40),
        ("requested_image", "other-image"),
        ("source_attestation", "unverified"),
        ("kind", "local_tiny"),
    ):
        with pytest.raises(ValueError, match="allocated construction"):
            admit_worker_runtime(runtime | {key: value}, request, "fixture-pod")
    with pytest.raises(ValueError):
        admit_worker_runtime(runtime, request, None)
