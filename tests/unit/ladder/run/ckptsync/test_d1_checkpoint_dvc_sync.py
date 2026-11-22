"""Checkpoint cadence sync proves DVC/R2 durability before mailbox publication."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from entropic_substrate.ladder.run.ckptsync.d1_checkpoint_dvc_sync import (
    D1CheckpointDvcSyncAdapter,
    D1CheckpointDvcSyncError,
    d1_checkpoint_sync_mailbox_path,
    parse_d1_checkpoint_dvc_sync_receipt,
)
from entropic_substrate.ladder.run.trainer.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)

pytestmark = pytest.mark.unit

_POD = "pod-sync-123"
_ATTEMPT = "a" * 64
_OBJECT = "b" * 32 + ".dir"


@dataclass
class _Dvc:
    repo: Path
    events: list[str]
    fail: str | None = None

    def add(self, *, tracked_path: Path, pointer_path: Path) -> None:
        self.events.append("add")
        if self.fail == "add":
            raise RuntimeError("provider-private-detail")
        pointer_path.write_text(
            f"outs:\n- md5: {_OBJECT}\n  size: 42\n  nfiles: 2\n  path: {tracked_path.name}\n",
            encoding="utf-8",
        )

    def push(self, *, pointer_path: Path) -> None:
        self.events.append("push")
        if self.fail == "push":
            raise RuntimeError("provider-private-detail")


@dataclass
class _Cloud:
    events: list[str]
    present: bool = True

    def object_is_present(self, *, pointer_path: Path, object_id: str) -> bool:
        self.events.append("cloud")
        assert pointer_path.name.endswith(".dvc") and object_id == _OBJECT
        return self.present


@dataclass
class _Objects:
    events: list[str]
    values: dict[str, bytes] = field(default_factory=dict)

    def publish(
        self,
        key: str,
        payload: bytes,
        *,
        expected_sha256: str,
    ) -> D1TrainerMailboxObjectIdentity:
        self.events.append("mailbox")
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        if key in self.values and self.values[key] != payload:
            raise ValueError("immutable collision")
        self.values[key] = payload
        return D1TrainerMailboxObjectIdentity(key, len(payload), expected_sha256)

    def read(self, key: str) -> D1TrainerMailboxObjectRead | None:
        payload = self.values.get(key)
        if payload is None:
            return None
        identity = D1TrainerMailboxObjectIdentity(
            key,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        return D1TrainerMailboxObjectRead(identity, payload)

    def list(self, prefix: str) -> tuple[D1TrainerMailboxObjectIdentity, ...]:
        del prefix
        return ()


def _case(tmp_path: Path, *, fail: str | None = None, cloud: bool = True):
    repo = tmp_path / "repo"
    run_root = repo / "results/d1-run-abc"
    run_root.mkdir(parents=True)
    checkpoint_path = run_root / "checkpoint.pt"
    journal_path = run_root / "journal.jsonl"
    checkpoint_path.write_bytes(b"checkpoint bytes")
    journal_path.write_bytes(b'{"event":"checkpoint"}\n')
    checkpoint = D1CheckpointArtifactReceipt(
        checkpoint_path,
        checkpoint_path.stat().st_size,
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    )
    events: list[str] = []
    objects = _Objects(events)
    adapter = D1CheckpointDvcSyncAdapter(
        repo_root=repo,
        dvc=_Dvc(repo, events, fail),
        cloud=_Cloud(events, cloud),
        mailbox=objects,
    )
    return adapter, objects, events, run_root, checkpoint, journal_path


def test_sync_pushes_verifies_then_publishes_self_contained_receipt(tmp_path: Path) -> None:
    adapter, objects, events, run_root, checkpoint, journal = _case(tmp_path)

    published = adapter.sync(
        pod_id=_POD,
        attempt_sha256=_ATTEMPT,
        logical_run_id="d1-confirmatory-both-s101",
        successful_step=1000,
        run_root=run_root,
        checkpoint=checkpoint,
        journal_path=journal,
    )

    assert events == ["add", "push", "cloud", "mailbox"]
    receipt = published.receipt
    assert receipt.successful_step == 1000
    assert receipt.checkpoint_sha256 == checkpoint.sha256
    assert receipt.journal_sha256 == hashlib.sha256(journal.read_bytes()).hexdigest()
    assert receipt.pointer.object_id == _OBJECT
    assert base64.b64decode(receipt.pointer_bytes_base64) == (
        run_root.with_name(f"{run_root.name}.dvc").read_bytes()
    )
    assert published.mailbox_object.key == d1_checkpoint_sync_mailbox_path(_POD, _ATTEMPT, 1000)
    assert (
        parse_d1_checkpoint_dvc_sync_receipt(objects.values[published.mailbox_object.key])
        == receipt
    )


def test_sync_is_idempotent_for_unchanged_checkpoint_and_pointer(tmp_path: Path) -> None:
    adapter, _objects, events, run_root, checkpoint, journal = _case(tmp_path)
    kwargs = {
        "pod_id": _POD,
        "attempt_sha256": _ATTEMPT,
        "logical_run_id": "d1-confirmatory-both-s101",
        "successful_step": 1000,
        "run_root": run_root,
        "checkpoint": checkpoint,
        "journal_path": journal,
    }

    first = adapter.sync(**kwargs)
    second = adapter.sync(**kwargs)

    assert second == first
    assert events == ["add", "push", "cloud", "mailbox"] * 2


@pytest.mark.parametrize(
    ("fail", "cloud", "code"),
    [
        ("add", True, "dvc_failure"),
        ("push", True, "dvc_failure"),
        (None, False, "cloud_unverified"),
    ],
)
def test_sync_refuses_before_mailbox_on_dvc_or_cloud_failure(
    tmp_path: Path,
    fail: str | None,
    cloud: bool,
    code: str,
) -> None:
    adapter, objects, _events, run_root, checkpoint, journal = _case(
        tmp_path,
        fail=fail,
        cloud=cloud,
    )
    with pytest.raises(D1CheckpointDvcSyncError) as raised:
        adapter.sync(
            pod_id=_POD,
            attempt_sha256=_ATTEMPT,
            logical_run_id="d1-run",
            successful_step=1000,
            run_root=run_root,
            checkpoint=checkpoint,
            journal_path=journal,
        )
    assert raised.value.code == code
    assert objects.values == {}
    assert "provider-private-detail" not in str(raised.value)


def test_sync_rejects_checkpoint_drift_and_unsafe_run_root_before_dvc(tmp_path: Path) -> None:
    adapter, objects, events, run_root, checkpoint, journal = _case(tmp_path)
    checkpoint.path.write_bytes(b"drift")
    with pytest.raises(D1CheckpointDvcSyncError, match="checkpoint_identity"):
        adapter.sync(
            pod_id=_POD,
            attempt_sha256=_ATTEMPT,
            logical_run_id="d1-run",
            successful_step=1000,
            run_root=run_root,
            checkpoint=checkpoint,
            journal_path=journal,
        )
    assert events == [] and objects.values == {}

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(D1CheckpointDvcSyncError, match="run_root"):
        adapter.sync(
            pod_id=_POD,
            attempt_sha256=_ATTEMPT,
            logical_run_id="d1-run",
            successful_step=1000,
            run_root=outside,
            checkpoint=checkpoint,
            journal_path=journal,
        )
