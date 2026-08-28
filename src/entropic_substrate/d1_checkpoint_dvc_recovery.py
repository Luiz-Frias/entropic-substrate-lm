"""Strict receipt discovery and DVC/R2 materialization for D1 crash recovery."""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from entropic_substrate.d1_artifact_snapshot import (
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
    snapshot_d1_artifact,
)
from entropic_substrate.d1_checkpoint_dvc_sync import (
    D1CheckpointDvcSyncReceipt,
    parse_d1_checkpoint_dvc_sync_receipt,
)
from entropic_substrate.d1_operator_values import require_d1_pod_id
from entropic_substrate.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.d1_trainer_object_transport import D1TrainerMailboxObjectPort
from entropic_substrate.evidence.d1_capture_ports import D1DvcPullPort
from entropic_substrate.evidence.d1_capture_service import read_d1_dvc_pointer
from entropic_substrate.experiments.d1_primitives import require_sha256
from entropic_substrate.predictive_state.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
)

__all__ = [
    "D1CheckpointDvcRecoveryError",
    "D1CheckpointDvcRecoveryService",
    "D1RecoveredCheckpointDvcState",
]

_MAILBOX_PREFIX = "d1-mailbox/v1/"
_SYNC_NAME = re.compile(r"^checkpoint-(\d{9})\.json$")
_TRACKED = re.compile(r"^results/d1-run-[0-9a-f]{64}$")


class D1CheckpointDvcRecoveryError(RuntimeError):
    """A bounded receipt-selection, pointer, DVC, or artifact refusal."""

    def __init__(self, *, code: str, path: str) -> None:
        self.code = code
        self.path = path
        super().__init__(f"D1 checkpoint DVC recovery {code} at {path}")


def _error(code: str, path: object) -> D1CheckpointDvcRecoveryError:
    return D1CheckpointDvcRecoveryError(code=code, path=str(path))


def _logical_run_id(value: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError("D1 recovery logical_run_id must be trimmed text")
    return value


def _checkpoint_sha256(value: str) -> str:
    require_sha256(value, label="D1 recovery checkpoint")
    if value != value.lower():
        raise ValueError("D1 recovery checkpoint must use lowercase hexadecimal")
    return value


@dataclass(frozen=True, slots=True)
class D1RecoveredCheckpointDvcState:
    receipt: D1CheckpointDvcSyncReceipt
    checkpoint: D1CheckpointArtifactReceipt
    journal_path: Path

    def __post_init__(self) -> None:
        if type(self.receipt) is not D1CheckpointDvcSyncReceipt:
            raise TypeError("D1 recovered state requires a sync receipt")
        if type(self.checkpoint) is not D1CheckpointArtifactReceipt:
            raise TypeError("D1 recovered state requires a checkpoint receipt")
        if not isinstance(cast(object, self.journal_path), Path):
            raise TypeError("D1 recovered state journal_path must be a Path")
        if self.checkpoint.path.name != "checkpoint.pt":
            raise ValueError("D1 recovered state checkpoint path is unsupported")
        if (
            self.journal_path.name != "journal.jsonl"
            or self.journal_path.parent != self.checkpoint.path.parent
        ):
            raise ValueError("D1 recovered state artifact paths differ")
        if (
            self.checkpoint.size_bytes != self.receipt.checkpoint_size_bytes
            or self.checkpoint.sha256 != self.receipt.checkpoint_sha256
        ):
            raise ValueError("D1 recovered state checkpoint identity differs")


class D1CheckpointDvcRecoveryService:
    """Select one immutable sync lineage and materialize its exact run tree."""

    def __init__(
        self,
        *,
        repo_root: Path,
        objects: D1TrainerMailboxObjectPort,
        dvc: D1DvcPullPort,
    ) -> None:
        root = Path(os.path.abspath(repo_root))
        if root.is_symlink() or not root.is_dir():
            raise ValueError("D1 recovery repo_root must be a real directory")
        if not all(callable(getattr(objects, name, None)) for name in ("list", "read")):
            raise TypeError("D1 recovery requires the mailbox object port")
        if not callable(getattr(dvc, "pull", None)):
            raise TypeError("D1 recovery requires the DVC pull port")
        self._root, self._objects, self._dvc = root, objects, dvc

    @staticmethod
    def _sync_step(identity: D1TrainerMailboxObjectIdentity) -> int | None:
        if type(identity) is not D1TrainerMailboxObjectIdentity:
            raise _error("object_identity", _MAILBOX_PREFIX)
        key = identity.key
        if not key.startswith(_MAILBOX_PREFIX):
            raise _error("foreign_object", key)
        parts = PurePosixPath(key).parts
        if len(parts) != 6 or parts[:2] != ("d1-mailbox", "v1"):
            if "sync" in parts:
                raise _error("receipt_key", key)
            return None
        if parts[4] != "sync":
            return None
        match = _SYNC_NAME.fullmatch(parts[5])
        if match is None:
            raise _error("receipt_key", key)
        try:
            require_d1_pod_id(parts[2])
            require_sha256(parts[3], label="D1 recovery attempt")
        except (TypeError, ValueError) as cause:
            raise _error("receipt_key", key) from cause
        step = int(match.group(1))
        if step < 1:
            raise _error("receipt_key", key)
        return step

    def _read_receipt(
        self,
        identity: D1TrainerMailboxObjectIdentity,
        *,
        step: int,
    ) -> D1CheckpointDvcSyncReceipt:
        try:
            observed = self._objects.read(identity.key)
        except Exception as cause:
            raise _error("object_read", identity.key) from cause
        if type(observed) is not D1TrainerMailboxObjectRead or observed.identity != identity:
            raise _error("object_read", identity.key)
        try:
            receipt = parse_d1_checkpoint_dvc_sync_receipt(observed.payload)
        except (TypeError, ValueError) as cause:
            raise _error("receipt_invalid", identity.key) from cause
        if receipt.successful_step != step:
            raise _error("receipt_key", identity.key)
        return receipt

    def latest(self, *, logical_run_id: str) -> D1CheckpointDvcSyncReceipt:
        logical = _logical_run_id(logical_run_id)
        try:
            identities = self._objects.list(_MAILBOX_PREFIX)
        except Exception as cause:
            raise _error("object_list", _MAILBOX_PREFIX) from cause
        by_step: dict[int, dict[bytes, D1CheckpointDvcSyncReceipt]] = {}
        for identity in identities:
            step = self._sync_step(identity)
            if step is None:
                continue
            receipt = self._read_receipt(identity, step=step)
            if receipt.logical_run_id != logical:
                continue
            if _TRACKED.fullmatch(receipt.tracked_path) is None:
                raise _error("tracked_path", receipt.tracked_path)
            by_step.setdefault(step, {})[receipt.canonical_bytes] = receipt
        if not by_step:
            raise _error("receipt_missing", logical)
        for step, receipts in by_step.items():
            if len(receipts) != 1:
                raise _error("receipt_conflict", step)
        latest_step = max(by_step)
        return next(iter(by_step[latest_step].values()))

    def _results_directory(self) -> Path:
        results = self._root / "results"
        try:
            results.mkdir(mode=0o700, exist_ok=True)
        except OSError as cause:
            raise _error("results_directory", "results") from cause
        if results.is_symlink() or not results.is_dir():
            raise _error("results_directory", "results")
        return results

    def _materialize_pointer(
        self,
        receipt: D1CheckpointDvcSyncReceipt,
    ) -> tuple[Path, Path]:
        self._results_directory()
        tracked = self._root.joinpath(*PurePosixPath(receipt.tracked_path).parts)
        pointer = self._root.joinpath(*PurePosixPath(receipt.pointer.pointer_path).parts)
        if tracked.is_symlink() or (tracked.exists() and not tracked.is_dir()):
            raise _error("tracked_path", receipt.tracked_path)
        payload = base64.b64decode(receipt.pointer_bytes_base64, validate=True)
        if pointer.exists():
            try:
                _, observed, digest = read_d1_artifact_bytes(
                    self._root,
                    receipt.pointer.pointer_path,
                    label="D1 recovery DVC pointer",
                )
            except Exception as cause:
                raise _error("pointer_occupied", receipt.pointer.pointer_path) from cause
            if observed != payload or digest != receipt.pointer.pointer_sha256:
                raise _error("pointer_occupied", receipt.pointer.pointer_path)
        else:
            try:
                publish_d1_atomic_bytes(
                    self._root,
                    receipt.pointer.pointer_path,
                    payload,
                    label="D1 recovery DVC pointer",
                )
            except Exception as cause:
                raise _error("pointer_publish", receipt.pointer.pointer_path) from cause
        try:
            observed_pointer, observed_bytes = read_d1_dvc_pointer(
                repo_root=self._root,
                tracked_path=tracked,
                pointer_path=pointer,
            )
        except Exception as cause:
            raise _error("pointer_identity", receipt.pointer.pointer_path) from cause
        if observed_pointer != receipt.pointer or observed_bytes != payload:
            raise _error("pointer_identity", receipt.pointer.pointer_path)
        return tracked, pointer

    def restore(
        self,
        *,
        logical_run_id: str,
        expected_checkpoint_sha256: str,
    ) -> D1RecoveredCheckpointDvcState:
        expected = _checkpoint_sha256(expected_checkpoint_sha256)
        receipt = self.latest(logical_run_id=logical_run_id)
        if receipt.checkpoint_sha256 != expected:
            raise _error("checkpoint_source", logical_run_id)
        tracked, pointer = self._materialize_pointer(receipt)
        try:
            self._dvc.pull(pointer_path=pointer)
        except Exception as cause:
            raise _error("dvc_pull", receipt.pointer.pointer_path) from cause
        try:
            checkpoint_path, checkpoint_size, checkpoint_sha = snapshot_d1_artifact(
                self._root,
                f"{receipt.tracked_path}/checkpoint.pt",
                label="D1 recovered checkpoint",
            )
            journal_path, journal_size, journal_sha = snapshot_d1_artifact(
                self._root,
                f"{receipt.tracked_path}/journal.jsonl",
                label="D1 recovered journal",
            )
        except Exception as cause:
            raise _error("restored_identity", receipt.tracked_path) from cause
        if (
            checkpoint_path.parent != tracked
            or (checkpoint_size, checkpoint_sha)
            != (receipt.checkpoint_size_bytes, receipt.checkpoint_sha256)
            or (journal_size, journal_sha) != (receipt.journal_size_bytes, receipt.journal_sha256)
        ):
            raise _error("restored_identity", receipt.tracked_path)
        return D1RecoveredCheckpointDvcState(
            receipt,
            D1CheckpointArtifactReceipt(
                checkpoint_path,
                checkpoint_size,
                checkpoint_sha,
            ),
            journal_path,
        )
