"""D1 crash recovery re-admits immutable sync receipts before targeted DVC pull."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from entropic_substrate.ladder.capture.custody.d1_capture_ports import D1DvcPointerIdentity
from entropic_substrate.ladder.run.ckptsync.d1_checkpoint_dvc_recovery import (
    D1CheckpointDvcRecoveryError,
    D1CheckpointDvcRecoveryService,
)
from entropic_substrate.ladder.run.ckptsync.d1_checkpoint_dvc_sync import (
    D1CheckpointDvcSyncReceipt,
    d1_checkpoint_sync_mailbox_path,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)

pytestmark = pytest.mark.unit

_LOGICAL = "d1-confirmatory-both-s101"
_RUN_DIGEST = "1" * 64
_TRACKED = f"results/d1-run-{_RUN_DIGEST}"
_POD_A = "pod-recovery-a"
_POD_B = "pod-recovery-b"
_ATTEMPT_A = "a" * 64
_ATTEMPT_B = "b" * 64


def _receipt(
    *,
    step: int,
    checkpoint: bytes,
    journal: bytes,
    logical_run_id: str = _LOGICAL,
    object_digest: str = "8" * 32,
) -> D1CheckpointDvcSyncReceipt:
    object_id = f"{object_digest}.dir"
    pointer_bytes = (
        f"outs:\n- md5: {object_id}\n"
        f"  size: {len(checkpoint) + len(journal)}\n"
        "  nfiles: 2\n"
        f"  path: d1-run-{_RUN_DIGEST}\n"
    ).encode()
    pointer = D1DvcPointerIdentity(
        f"{_TRACKED}.dvc",
        hashlib.sha256(pointer_bytes).hexdigest(),
        object_id,
    )
    return D1CheckpointDvcSyncReceipt(
        logical_run_id,
        step,
        _TRACKED,
        len(checkpoint),
        hashlib.sha256(checkpoint).hexdigest(),
        len(journal),
        hashlib.sha256(journal).hexdigest(),
        pointer,
        base64.b64encode(pointer_bytes).decode("ascii"),
    )


@dataclass
class _Objects:
    values: dict[str, bytes] = field(default_factory=dict)
    reads: list[str] = field(default_factory=list)

    @staticmethod
    def _identity(key: str, payload: bytes) -> D1TrainerMailboxObjectIdentity:
        return D1TrainerMailboxObjectIdentity(
            key,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )

    def seed(self, key: str, payload: bytes) -> None:
        self.values[key] = payload

    def publish(
        self,
        key: str,
        payload: bytes,
        *,
        expected_sha256: str,
    ) -> D1TrainerMailboxObjectIdentity:
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        self.seed(key, payload)
        return self._identity(key, payload)

    def read(self, key: str) -> D1TrainerMailboxObjectRead | None:
        self.reads.append(key)
        payload = self.values.get(key)
        return (
            None
            if payload is None
            else D1TrainerMailboxObjectRead(self._identity(key, payload), payload)
        )

    def list(self, prefix: str) -> tuple[D1TrainerMailboxObjectIdentity, ...]:
        return tuple(
            self._identity(key, payload)
            for key, payload in sorted(self.values.items())
            if key.startswith(prefix)
        )


@dataclass
class _Dvc:
    payloads: dict[str, tuple[bytes, bytes]]
    calls: list[Path] = field(default_factory=list)
    drift: str | None = None
    fail: bool = False

    def pull(self, *, pointer_path: Path) -> None:
        self.calls.append(pointer_path)
        if self.fail:
            raise RuntimeError("provider-private-detail")
        pointer_bytes = pointer_path.read_bytes()
        checkpoint, journal = self.payloads[hashlib.sha256(pointer_bytes).hexdigest()]
        root = pointer_path.with_suffix("")
        root.mkdir(parents=True, exist_ok=True)
        root.joinpath("checkpoint.pt").write_bytes(
            b"checkpoint drift" if self.drift == "checkpoint" else checkpoint
        )
        root.joinpath("journal.jsonl").write_bytes(
            b"journal drift\n" if self.drift == "journal" else journal
        )


def _seed(
    objects: _Objects,
    receipt: D1CheckpointDvcSyncReceipt,
    *,
    pod: str,
    attempt: str,
    key_step: int | None = None,
) -> str:
    key = d1_checkpoint_sync_mailbox_path(
        pod,
        attempt,
        receipt.successful_step if key_step is None else key_step,
    )
    objects.seed(key, receipt.canonical_bytes)
    return key


def _service(repo: Path, objects: _Objects, dvc: _Dvc) -> D1CheckpointDvcRecoveryService:
    return D1CheckpointDvcRecoveryService(repo_root=repo, objects=objects, dvc=dvc)


def test_latest_selects_highest_valid_receipt_and_allows_identical_duplicate(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    objects = _Objects()
    first = _receipt(step=1000, checkpoint=b"checkpoint-1", journal=b"journal-1\n")
    latest = _receipt(
        step=2000,
        checkpoint=b"checkpoint-2",
        journal=b"journal-2\n",
        object_digest="9" * 32,
    )
    _seed(objects, first, pod=_POD_A, attempt=_ATTEMPT_A)
    _seed(objects, latest, pod=_POD_A, attempt=_ATTEMPT_A)
    _seed(objects, latest, pod=_POD_B, attempt=_ATTEMPT_B)
    objects.seed(
        f"d1-mailbox/v1/{_POD_A}/{_ATTEMPT_A}/commands/bootstrap.json",
        b"not a sync receipt",
    )

    selected = _service(repo, objects, _Dvc({})).latest(logical_run_id=_LOGICAL)

    assert selected == latest
    assert len(objects.reads) == 3


def test_latest_rejects_conflicting_receipts_at_one_step(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    objects = _Objects()
    left = _receipt(step=1000, checkpoint=b"checkpoint", journal=b"journal-a\n")
    right = _receipt(step=1000, checkpoint=b"checkpoint", journal=b"journal-b\n")
    _seed(objects, left, pod=_POD_A, attempt=_ATTEMPT_A)
    _seed(objects, right, pod=_POD_B, attempt=_ATTEMPT_B)

    with pytest.raises(D1CheckpointDvcRecoveryError) as raised:
        _service(repo, objects, _Dvc({})).latest(logical_run_id=_LOGICAL)

    assert raised.value.code == "receipt_conflict"


def test_latest_rejects_key_receipt_cursor_mismatch_and_absent_run(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    objects = _Objects()
    receipt = _receipt(step=1000, checkpoint=b"checkpoint", journal=b"journal\n")
    _seed(objects, receipt, pod=_POD_A, attempt=_ATTEMPT_A, key_step=2000)
    service = _service(repo, objects, _Dvc({}))

    with pytest.raises(D1CheckpointDvcRecoveryError) as mismatch:
        service.latest(logical_run_id=_LOGICAL)
    assert mismatch.value.code == "receipt_key"

    empty = _service(repo, _Objects(), _Dvc({}))
    with pytest.raises(D1CheckpointDvcRecoveryError) as missing:
        empty.latest(logical_run_id=_LOGICAL)
    assert missing.value.code == "receipt_missing"


def test_restore_reconstructs_exact_pointer_pulls_and_verifies_artifacts(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint, journal = b"checkpoint bytes", b'{"event":"checkpoint"}\n'
    receipt = _receipt(step=1000, checkpoint=checkpoint, journal=journal)
    objects = _Objects()
    _seed(objects, receipt, pod=_POD_A, attempt=_ATTEMPT_A)
    dvc = _Dvc({receipt.pointer.pointer_sha256: (checkpoint, journal)})

    recovered = _service(repo, objects, dvc).restore(
        logical_run_id=_LOGICAL,
        expected_checkpoint_sha256=receipt.checkpoint_sha256,
    )

    pointer_path = repo / receipt.pointer.pointer_path
    assert pointer_path.read_bytes() == base64.b64decode(receipt.pointer_bytes_base64)
    assert dvc.calls == [pointer_path]
    assert recovered.receipt == receipt
    assert recovered.checkpoint.sha256 == receipt.checkpoint_sha256
    assert recovered.journal_path.read_bytes() == journal


@pytest.mark.parametrize("drift", ["checkpoint", "journal"])
def test_restore_refuses_drift_after_pull(tmp_path: Path, drift: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint, journal = b"checkpoint bytes", b"journal\n"
    receipt = _receipt(step=1000, checkpoint=checkpoint, journal=journal)
    objects = _Objects()
    _seed(objects, receipt, pod=_POD_A, attempt=_ATTEMPT_A)
    dvc = _Dvc(
        {receipt.pointer.pointer_sha256: (checkpoint, journal)},
        drift=drift,
    )

    with pytest.raises(D1CheckpointDvcRecoveryError) as raised:
        _service(repo, objects, dvc).restore(
            logical_run_id=_LOGICAL,
            expected_checkpoint_sha256=receipt.checkpoint_sha256,
        )

    assert raised.value.code == "restored_identity"


def test_restore_refuses_unapproved_checkpoint_and_sanitizes_pull_failure(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    receipt = _receipt(step=1000, checkpoint=b"checkpoint", journal=b"journal\n")
    objects = _Objects()
    _seed(objects, receipt, pod=_POD_A, attempt=_ATTEMPT_A)
    dvc = _Dvc({receipt.pointer.pointer_sha256: (b"checkpoint", b"journal\n")})
    service = _service(repo, objects, dvc)

    with pytest.raises(D1CheckpointDvcRecoveryError) as source:
        service.restore(logical_run_id=_LOGICAL, expected_checkpoint_sha256="f" * 64)
    assert source.value.code == "checkpoint_source" and dvc.calls == []

    dvc.fail = True
    with pytest.raises(D1CheckpointDvcRecoveryError) as failed:
        service.restore(
            logical_run_id=_LOGICAL,
            expected_checkpoint_sha256=receipt.checkpoint_sha256,
        )
    assert failed.value.code == "dvc_pull"
    assert "provider-private-detail" not in str(failed.value)
