"""Host completion admission binds transferred bytes, native receipts, and request."""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_execution import prepared
from tests.test_sentence_worker_request import request as local_request

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.object_worker import run_object_worker
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
)
from entropic_substrate.ladder.run.worker_completion import (
    admit_worker_runtime,
    receive_worker_completion,
)
from entropic_substrate.ladder.run.worker_request import ProbeRequest
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


@pytest.fixture(scope="module")
def _complete_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ProbeRequest, D1TrainerMailboxObjectIdentity, dict[str, bytes]]:
    """Run the REAL worker pipeline exactly ONCE for this whole module (same
    pattern as ``tests/test_h_probe_events.py``'s ``_checkpoint_fixture``):
    every publication-block arm below mutates its OWN in-memory copy of the
    resulting object values, never re-running the (slow) training pipeline."""
    tmp_path = tmp_path_factory.mktemp("worker-completion-pub")
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    terminal = run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    return request, terminal, dict(objects.values)


def test_a_complete_event_is_admitted_with_or_without_a_publication_block(
    tmp_path: Path,
    _complete_fixture: tuple[ProbeRequest, D1TrainerMailboxObjectIdentity, dict[str, bytes]],
) -> None:
    """Read-old/write-new (FOR-1567/pre-for-1567c, DN-1567-C): the REAL
    worker's `complete` event carries a `publication` block and admits
    (write-new); a `complete` event recorded BEFORE this door existed -- no
    `publication` key at all -- is still admitted (read-old). Mutation: drop
    the extra-key admission in `worker_completion.py` and the write-new arm
    below goes RED (`recovery payload fields differ from the versioned
    schema`), independent of the read-old arm, which would still pass."""
    request, terminal, values = _complete_fixture
    write_new = _ObjectStore(values=dict(values))
    row = json.loads(write_new.values[terminal.key])
    assert "publication" in row  # write-new sanity: the fixture really has one
    host_new = tmp_path / "host-write-new"
    host_new.mkdir()
    admitted_new = receive_worker_completion(
        write_new, request, terminal, host_new, expected_pod_id=None, max_marker_bytes=1_000_000
    )
    assert admitted_new["status"] == "probe_completion_verified"

    read_old = _ObjectStore(values=dict(values))
    del row["publication"]
    old_payload = json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    read_old.values[terminal.key] = old_payload
    old_terminal = read_old._identity(terminal.key, old_payload)
    host_old = tmp_path / "host-read-old"
    host_old.mkdir()
    admitted_old = receive_worker_completion(
        read_old, request, old_terminal, host_old, expected_pod_id=None, max_marker_bytes=1_000_000
    )
    assert admitted_old["status"] == "probe_completion_verified"


@pytest.mark.parametrize(
    "corruption",
    [
        lambda pub: {**pub, "retries": -1},
        lambda pub: {**pub, "l1_seconds": -0.1},
        lambda pub: {k: v for k, v in pub.items() if k != "retries"},
    ],
)
def test_a_malformed_publication_block_on_complete_is_refused(
    tmp_path: Path,
    _complete_fixture: tuple[ProbeRequest, D1TrainerMailboxObjectIdentity, dict[str, bytes]],
    corruption: Callable[[dict[str, object]], dict[str, object]],
) -> None:
    request, terminal, values = _complete_fixture
    objects = _ObjectStore(values=dict(values))
    row = json.loads(objects.values[terminal.key])
    row["publication"] = corruption(row["publication"])
    payload = json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    objects.values[terminal.key] = payload
    bad_terminal = objects._identity(terminal.key, payload)
    host = tmp_path / "host-malformed"
    host.mkdir()
    with pytest.raises(ValueError):
        receive_worker_completion(
            objects, request, bad_terminal, host, expected_pod_id=None, max_marker_bytes=1_000_000
        )


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


def test_a_pair_result_v2_is_refused_by_name_at_the_single_profile_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.ladder.run import worker_completion
    from entropic_substrate.ladder.run.worker_request import RESULT_SCHEMA_V2

    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    terminal = run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    host = tmp_path / "host-honest"
    host.mkdir()
    # control: the single-profile result is admitted when it is what the worker wrote
    assert (
        receive_worker_completion(
            objects, request, terminal, host, expected_pod_id=None, max_marker_bytes=1_000_000
        )["status"]
        == "probe_completion_verified"
    )
    real_read_json = worker_completion.read_json

    def relabelled(root: Path, artifact: ArtifactIdentity, *, max_bytes: int) -> dict[str, object]:
        row = real_read_json(root, artifact, max_bytes=max_bytes)
        if artifact.path == "result.json":
            return row | {"schema": RESULT_SCHEMA_V2}  # a pair's result, presented to one request
        return row

    monkeypatch.setattr(worker_completion, "read_json", relabelled)
    rejected = tmp_path / "host-relabelled"
    rejected.mkdir()
    # refused BY NAME with the one shared text, never relabelled into a single-profile result
    with pytest.raises(ValueError, match="names a pair's cells; a single-profile request admits"):
        receive_worker_completion(
            objects, request, terminal, rejected, expected_pod_id=None, max_marker_bytes=1_000_000
        )
