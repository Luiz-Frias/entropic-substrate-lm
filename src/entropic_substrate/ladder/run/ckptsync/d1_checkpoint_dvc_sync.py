"""Cadenced D1 checkpoint-to-DVC/R2 sync with immutable mailbox receipt."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from entropic_substrate.kernel.artifacts import snapshot_d1_artifact
from entropic_substrate.kernel.identity.d1_primitives import require_sha256_hex_any_case
from entropic_substrate.ladder.capture.custody.d1_capture_ports import (
    D1CloudObjectVerifierPort,
    D1DvcCommandPort,
    D1DvcPointerIdentity,
)
from entropic_substrate.ladder.capture.custody.d1_capture_service import read_d1_dvc_pointer
from entropic_substrate.ladder.run.campaign.d1_operator_values import require_d1_pod_id
from entropic_substrate.ladder.run.trainer.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    d1_trainer_mailbox_attempt_prefix,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)

__all__ = [
    "D1CheckpointDvcSyncAdapter",
    "D1CheckpointDvcSyncError",
    "D1CheckpointDvcSyncReceipt",
    "D1PublishedCheckpointDvcSync",
    "d1_checkpoint_sync_mailbox_path",
    "parse_d1_checkpoint_dvc_sync_receipt",
]

_SCHEMA = "d1-checkpoint-dvc-sync-v1"
_FIELDS = frozenset(
    "checkpoint_sha256 checkpoint_size_bytes journal_sha256 journal_size_bytes "
    "logical_run_id pointer pointer_bytes_base64 schema_version successful_step tracked_path".split()
)
_POINTER_FIELDS = frozenset({"object_id", "pointer_path", "pointer_sha256"})


class D1CheckpointDvcSyncError(RuntimeError):
    """A bounded local identity, DVC, cloud, or mailbox sync failure."""

    def __init__(self, *, code: str, path: str) -> None:
        self.code = code
        self.path = path
        super().__init__(f"D1 checkpoint DVC sync {code} at {path}")


def _error(code: str, path: object) -> D1CheckpointDvcSyncError:
    return D1CheckpointDvcSyncError(code=code, path=str(path))


def _sha256(value: str, *, label: str) -> str:
    require_sha256_hex_any_case(value, label=label)
    if value != value.lower():
        raise ValueError(f"{label} must use lowercase hexadecimal")
    return value


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pointer_bytes(value: str, pointer: D1DvcPointerIdentity) -> bytes:
    if type(value) is not str or not value:
        raise ValueError("D1 checkpoint sync pointer base64 must be nonempty text")
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("D1 checkpoint sync pointer base64 is invalid") from error
    if base64.b64encode(payload).decode("ascii") != value:
        raise ValueError("D1 checkpoint sync pointer base64 is noncanonical")
    if not payload or hashlib.sha256(payload).hexdigest() != pointer.pointer_sha256:
        raise ValueError("D1 checkpoint sync pointer bytes differ from identity")
    return payload


@dataclass(frozen=True, slots=True)
class D1CheckpointDvcSyncReceipt:
    logical_run_id: str
    successful_step: int
    tracked_path: str
    checkpoint_size_bytes: int
    checkpoint_sha256: str
    journal_size_bytes: int
    journal_sha256: str
    pointer: D1DvcPointerIdentity
    pointer_bytes_base64: str
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.logical_run_id) is not str
            or not self.logical_run_id
            or self.logical_run_id.strip() != self.logical_run_id
        ):
            raise ValueError("D1 checkpoint sync logical_run_id must be trimmed text")
        if type(self.successful_step) is not int or self.successful_step < 1:
            raise ValueError("D1 checkpoint sync successful_step must be positive")
        if type(self.tracked_path) is not str or not self.tracked_path.startswith(
            "results/d1-run-"
        ):
            raise ValueError("D1 checkpoint sync tracked_path is unsupported")
        if type(self.checkpoint_size_bytes) is not int or self.checkpoint_size_bytes < 1:
            raise ValueError("D1 checkpoint sync checkpoint size must be positive")
        if type(self.journal_size_bytes) is not int or self.journal_size_bytes < 1:
            raise ValueError("D1 checkpoint sync journal size must be positive")
        _sha256(self.checkpoint_sha256, label="D1 checkpoint sync checkpoint")
        _sha256(self.journal_sha256, label="D1 checkpoint sync journal")
        if type(self.pointer) is not D1DvcPointerIdentity:
            raise TypeError("D1 checkpoint sync requires a DVC pointer identity")
        if self.pointer.pointer_path != f"{self.tracked_path}.dvc":
            raise ValueError("D1 checkpoint sync pointer path differs from tracked path")
        _pointer_bytes(self.pointer_bytes_base64, self.pointer)
        if self.schema_version != _SCHEMA:
            raise ValueError("D1 checkpoint sync schema_version is unsupported")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "journal_sha256": self.journal_sha256,
            "journal_size_bytes": self.journal_size_bytes,
            "logical_run_id": self.logical_run_id,
            "pointer": {
                "object_id": self.pointer.object_id,
                "pointer_path": self.pointer.pointer_path,
                "pointer_sha256": self.pointer.pointer_sha256,
            },
            "pointer_bytes_base64": self.pointer_bytes_base64,
            "schema_version": self.schema_version,
            "successful_step": self.successful_step,
            "tracked_path": self.tracked_path,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical(self.manifest)


@dataclass(frozen=True, slots=True)
class D1PublishedCheckpointDvcSync:
    receipt: D1CheckpointDvcSyncReceipt
    mailbox_object: D1TrainerMailboxObjectIdentity

    def __post_init__(self) -> None:
        if type(self.receipt) is not D1CheckpointDvcSyncReceipt:
            raise TypeError("published checkpoint sync requires a nominal receipt")
        if type(self.mailbox_object) is not D1TrainerMailboxObjectIdentity:
            raise TypeError("published checkpoint sync requires a mailbox object")
        payload = self.receipt.canonical_bytes
        if (
            self.mailbox_object.size_bytes != len(payload)
            or self.mailbox_object.sha256 != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError("published checkpoint sync mailbox identity differs")


def d1_checkpoint_sync_mailbox_path(
    pod_id: str,
    attempt_sha256: str,
    successful_step: int,
) -> str:
    prefix = d1_trainer_mailbox_attempt_prefix(pod_id, attempt_sha256)
    if type(successful_step) is not int or successful_step < 1:
        raise ValueError("checkpoint sync mailbox step must be positive")
    return f"{prefix}/sync/checkpoint-{successful_step:09d}.json"


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate checkpoint sync key {key!r}")
        value[key] = item
    return value


def parse_d1_checkpoint_dvc_sync_receipt(payload: bytes) -> D1CheckpointDvcSyncReceipt:
    if type(payload) is not bytes:
        raise TypeError("checkpoint sync receipt payload must be bytes")
    try:
        value: object = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"nonfinite checkpoint sync value {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint sync receipt must be UTF-8 JSON") from error
    if type(value) is not dict or frozenset(value) != _FIELDS:
        raise ValueError("checkpoint sync receipt fields must be exact")
    root = cast(dict[str, object], value)
    raw_pointer = root["pointer"]
    if type(raw_pointer) is not dict or frozenset(raw_pointer) != _POINTER_FIELDS:
        raise ValueError("checkpoint sync pointer fields must be exact")
    item = cast(dict[str, object], raw_pointer)
    try:
        receipt = D1CheckpointDvcSyncReceipt(
            cast(str, root["logical_run_id"]),
            cast(int, root["successful_step"]),
            cast(str, root["tracked_path"]),
            cast(int, root["checkpoint_size_bytes"]),
            cast(str, root["checkpoint_sha256"]),
            cast(int, root["journal_size_bytes"]),
            cast(str, root["journal_sha256"]),
            D1DvcPointerIdentity(
                cast(str, item["pointer_path"]),
                cast(str, item["pointer_sha256"]),
                cast(str, item["object_id"]),
            ),
            cast(str, root["pointer_bytes_base64"]),
            cast(str, root["schema_version"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint sync receipt is invalid: {error}") from error
    if receipt.canonical_bytes != payload:
        raise ValueError("checkpoint sync receipt must use canonical JSON")
    return receipt


class D1CheckpointDvcSyncAdapter:
    """Sync one checkpointed run directory and publish its recovery receipt last."""

    def __init__(
        self,
        *,
        repo_root: Path,
        dvc: D1DvcCommandPort,
        cloud: D1CloudObjectVerifierPort,
        mailbox: D1TrainerMailboxObjectPort,
    ) -> None:
        root = Path(os.path.abspath(repo_root))
        if root.is_symlink() or not root.is_dir():
            raise ValueError("checkpoint sync repo_root must be a real directory")
        if not all(callable(getattr(dvc, name, None)) for name in ("add", "push")):
            raise TypeError("checkpoint sync requires the DVC command port")
        if not callable(getattr(cloud, "object_is_present", None)):
            raise TypeError("checkpoint sync requires the cloud verifier port")
        if not callable(getattr(mailbox, "publish", None)):
            raise TypeError("checkpoint sync requires the mailbox object port")
        self._root, self._dvc, self._cloud, self._mailbox = root, dvc, cloud, mailbox

    def _run_root(self, value: Path) -> tuple[Path, str]:
        path = Path(os.path.abspath(value))
        try:
            relative = path.relative_to(self._root)
        except ValueError as error:
            raise _error("run_root", path) from error
        if (
            path.is_symlink()
            or not path.is_dir()
            or relative.parent != Path("results")
            or not relative.name.startswith("d1-run-")
        ):
            raise _error("run_root", relative)
        return path, relative.as_posix()

    def sync(
        self,
        *,
        pod_id: str,
        attempt_sha256: str,
        logical_run_id: str,
        successful_step: int,
        run_root: Path,
        checkpoint: D1CheckpointArtifactReceipt,
        journal_path: Path,
    ) -> D1PublishedCheckpointDvcSync:
        require_d1_pod_id(pod_id)
        _sha256(attempt_sha256, label="D1 checkpoint sync attempt")
        root, tracked = self._run_root(run_root)
        if type(checkpoint) is not D1CheckpointArtifactReceipt:
            raise TypeError("checkpoint sync requires a checkpoint receipt")
        checkpoint_relative = f"{tracked}/checkpoint.pt"
        journal_relative = f"{tracked}/journal.jsonl"
        if checkpoint.path != root / "checkpoint.pt" or journal_path != root / "journal.jsonl":
            raise _error("artifact_path", tracked)
        try:
            _, checkpoint_size, checkpoint_sha = snapshot_d1_artifact(
                self._root,
                checkpoint_relative,
                label="D1 checkpoint sync checkpoint",
            )
            _, journal_size, journal_sha = snapshot_d1_artifact(
                self._root,
                journal_relative,
                label="D1 checkpoint sync journal",
            )
        except Exception as error:
            raise _error("artifact_snapshot", tracked) from error
        if (checkpoint_size, checkpoint_sha) != (checkpoint.size_bytes, checkpoint.sha256):
            raise _error("checkpoint_identity", checkpoint.path)
        pointer_path = root.with_name(f"{root.name}.dvc")
        try:
            self._dvc.add(tracked_path=root, pointer_path=pointer_path)
            self._dvc.push(pointer_path=pointer_path)
        except Exception as error:
            raise _error("dvc_failure", tracked) from error
        pointer, pointer_bytes = read_d1_dvc_pointer(
            repo_root=self._root,
            tracked_path=root,
            pointer_path=pointer_path,
        )
        try:
            present = self._cloud.object_is_present(
                pointer_path=pointer_path,
                object_id=pointer.object_id,
            )
        except Exception as error:
            raise _error("cloud_failure", pointer.pointer_path) from error
        if present is not True:
            raise _error("cloud_unverified", pointer.pointer_path)
        receipt = D1CheckpointDvcSyncReceipt(
            logical_run_id,
            successful_step,
            tracked,
            checkpoint_size,
            checkpoint_sha,
            journal_size,
            journal_sha,
            pointer,
            base64.b64encode(pointer_bytes).decode("ascii"),
        )
        payload = receipt.canonical_bytes
        key = d1_checkpoint_sync_mailbox_path(pod_id, attempt_sha256, successful_step)
        expected = D1TrainerMailboxObjectIdentity(
            key,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        try:
            observed = self._mailbox.publish(
                key,
                payload,
                expected_sha256=expected.sha256,
            )
        except Exception as error:
            raise _error("mailbox_failure", key) from error
        if type(observed) is not D1TrainerMailboxObjectIdentity or observed != expected:
            raise _error("mailbox_identity", key)
        return D1PublishedCheckpointDvcSync(receipt, observed)
