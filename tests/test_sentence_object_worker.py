"""Object-store completion must follow the actual complete worker lineage."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_execution import prepared
from tests.test_sentence_worker_transport import _FlakyReadStore, _ReadFailure

from entropic_substrate.ladder.run import object_worker
from entropic_substrate.ladder.run.object_worker import run_object_worker
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
)
from entropic_substrate.ladder.run.worker_snapshot import materialize_worker_snapshot
from entropic_substrate.ladder.run.worker_transport import publish_probe_request

pytestmark = pytest.mark.unit


def test_completion_marker_follows_verified_outputs_and_recovers_externally(tmp_path: Path) -> None:
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    terminal = run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    assert objects.publishes[-1] == f"{request.prefix}/complete.json"
    row = json.loads(objects.values[terminal.key])
    assert row["status"] == "probe_workload_complete"
    assert row["scientific_authority"] == "none"
    assert any("checkpoint-00000002.json" in key for key in objects.publishes)
    snapshot = D1TrainerMailboxObjectIdentity(**row["snapshot"])
    host = tmp_path / "host"
    host.mkdir()
    materialize_worker_snapshot(objects, request, snapshot, host, max_marker_bytes=1_000_000)
    result = json.loads((host / "worker/result.json").read_text())
    assert result["successful_optimizer_steps"] == 6
    assert result["request_sha256"] == request.digest
    assert (host / result["verification"]["path"]).is_file()


def test_checkpoint_upload_failure_retains_failure_marker_not_completion(tmp_path: Path) -> None:
    class BrokenCheckpoint(_ObjectStore):
        def publish(
            self, key: str, payload: bytes, *, expected_sha256: str
        ) -> D1TrainerMailboxObjectIdentity:
            if key.endswith("/snapshots/checkpoint-00000002.json"):
                raise RuntimeError("injected checkpoint upload failure")
            return super().publish(key, payload, expected_sha256=expected_sha256)

    request, inputs = prepared(tmp_path)
    objects = BrokenCheckpoint()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    with pytest.raises(RuntimeError, match="checkpoint upload failure"):
        run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    assert f"{request.prefix}/complete.json" not in objects.values
    failure = json.loads(objects.values[f"{request.prefix}/failure.json"])
    assert failure["status"] == "execution_defect"
    assert failure["snapshot"] is not None


def test_missing_native_receipt_cannot_be_published_as_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    original = object_worker.execute_probe_request

    def broken(*args: object, **kwargs: object) -> dict[str, object]:
        return original(*args, **kwargs) | {"verification": None}

    monkeypatch.setattr(object_worker, "execute_probe_request", broken)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    with pytest.raises(ValueError, match="native verification receipts"):
        run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    assert f"{request.prefix}/complete.json" not in objects.values
    failure = json.loads(objects.values[f"{request.prefix}/failure.json"])
    assert failure["status"] == "execution_defect" and failure["error_type"] == "ValueError"


def test_snapshot_budget_failure_is_an_explicit_partial_terminal_result(tmp_path: Path) -> None:
    request, inputs = prepared(tmp_path)
    minimum = len(request.payload) + (inputs / "source/prepared-corpus.json").stat().st_size + 1024
    request = replace(request, limits=replace(request.limits, max_total_bytes=minimum))
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    with pytest.raises(ExceptionGroup, match="worker failure"):
        run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    failure = json.loads(objects.values[f"{request.prefix}/failure.json"])
    assert failure["snapshot"] is None and failure["evidence_delivery"] == "failed"
    assert f"{request.prefix}/complete.json" not in objects.values


def test_paused_worker_publishes_recovery_not_completion(tmp_path: Path) -> None:
    request, inputs = prepared(tmp_path, pause=("prediction", 2))
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    result = run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    assert result.key == f"{request.prefix}/paused.json"
    assert json.loads(objects.values[result.key])["status"] == "paused"
    assert f"{request.prefix}/complete.json" not in objects.values


_PUBLICATION_KEYS = {
    "l1_seconds",
    "l2_seconds",
    "l3_seconds",
    "objects_put",
    "objects_skipped",
    "verification_reads",
    "retries",
}


def _assert_well_shaped_publication(publication: dict[str, object]) -> None:
    assert set(publication) == _PUBLICATION_KEYS
    for seconds_field in ("l1_seconds", "l2_seconds", "l3_seconds"):
        assert isinstance(publication[seconds_field], (int, float))
        assert publication[seconds_field] >= 0
    for count_field in ("objects_put", "objects_skipped", "verification_reads", "retries"):
        assert type(publication[count_field]) is int and publication[count_field] >= 0


def test_the_worker_always_writes_a_publication_block_on_the_complete_event(
    tmp_path: Path,
) -> None:
    """Write-new (FOR-1567/pre-for-1567c, DN-1567-C): the final-snapshot leg
    carries a `publication` block on `complete`, unconditionally. l1/l2 are
    pinned to exactly 0.0 -- NOT MEASURED for this leg (see the comment at
    its call site in object_worker.py for why reusing the last checkpoint's
    values would double-report one measurement)."""
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    terminal = run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    publication = json.loads(objects.values[terminal.key])["publication"]
    _assert_well_shaped_publication(publication)
    assert publication["l1_seconds"] == 0.0
    assert publication["l2_seconds"] == 0.0


def test_the_worker_always_writes_a_publication_block_on_the_paused_event(tmp_path: Path) -> None:
    """Write-new (DN-1567-C): the same final-snapshot call site, reached via
    the pause path, carries a `publication` block too -- no branch on the
    terminal name."""
    request, inputs = prepared(tmp_path, pause=("prediction", 2))
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    result = run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    publication = json.loads(objects.values[result.key])["publication"]
    _assert_well_shaped_publication(publication)


def test_two_transient_reads_on_the_final_leg_publish_once_and_record_two_retries(
    tmp_path: Path,
) -> None:
    """ACCEPTANCE: on the final leg, two transient verification-read failures
    then success publish once and record `retries: 2`. The fault targets the
    terminal snapshot MARKER key -- the one key only the final leg publishes
    -- never a PUT (house rule 3's ruled carve-out)."""
    request, inputs = prepared(tmp_path)
    fault_key = f"{request.prefix}/snapshots/complete.json"
    objects = _FlakyReadStore(fail_reads_remaining={fault_key: 2})
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    terminal = run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    publication = json.loads(objects.values[terminal.key])["publication"]
    assert publication["retries"] == 2
    assert objects.publish_calls.count(fault_key) == 1


def test_three_transient_reads_on_the_final_leg_raise_the_original_class(tmp_path: Path) -> None:
    """ACCEPTANCE: three failures raise the original error class; `put_object`
    (here, ``publish``) is still called exactly once for the faulted key."""
    request, inputs = prepared(tmp_path)
    fault_key = f"{request.prefix}/snapshots/complete.json"
    objects = _FlakyReadStore(fail_reads_remaining={fault_key: 3})
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    with pytest.raises(_ReadFailure):
        run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    assert objects.publish_calls.count(fault_key) == 1
    assert f"{request.prefix}/complete.json" not in objects.values
    failure = json.loads(objects.values[f"{request.prefix}/failure.json"])
    assert failure["status"] == "execution_defect" and failure["stage"] == "final_snapshot"
