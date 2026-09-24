"""Atomic compare-and-swap repository for canonical D1 campaign checkpoints."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from entropic_substrate.ladder.run.campaign.d1_campaign_checkpoint import (
    D1CampaignCheckpoint,
    D1CampaignCheckpointError,
    parse_campaign_checkpoint,
    serialize_campaign_checkpoint,
)

_CHUNK_SIZE = 1024 * 1024


class D1CampaignCheckpointRepositoryError(RuntimeError):
    """Checkpoint bytes are missing, corrupt, stale, forked, or not durable."""

    def __init__(self, *, code: str, path: Path, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = {"code": code, "path": str(path)}
        super().__init__(f"D1 campaign checkpoint repository {code} at {path}: {detail}")


@dataclass(frozen=True, slots=True)
class D1CampaignCheckpointReceipt:
    path: Path
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.path), Path):
            raise TypeError("campaign checkpoint receipt path must be a Path")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("campaign checkpoint receipt size_bytes must be positive")
        _expected_digest(self.sha256)


def _error(code: str, path: Path, detail: str) -> D1CampaignCheckpointRepositoryError:
    return D1CampaignCheckpointRepositoryError(code=code, path=path, detail=detail)


def _raise(code: str, path: Path, detail: str, cause: Exception) -> NoReturn:
    raise _error(code, path, detail) from cause


def _expected_digest(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("expected checkpoint SHA-256 must be 64 lowercase hexadecimal")
    return value


def _validate_parent(path: Path) -> None:
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise _error("invalid_parent", path, "parent must be an existing real directory")


def _read_regular(path: Path) -> bytes:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise _error("missing_checkpoint", path, "checkpoint file is absent") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise _error(
            "invalid_checkpoint_path",
            path,
            "checkpoint must be a regular non-symlink file",
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        _raise("checkpoint_io", path, f"cannot open checkpoint: {error}", error)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _error("invalid_checkpoint_path", path, "opened checkpoint is not regular")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, _CHUNK_SIZE):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_if_present(path: Path) -> bytes | None:
    if not path.exists():
        if path.is_symlink():
            raise _error("invalid_checkpoint_path", path, "checkpoint symlink is forbidden")
        return None
    return _read_regular(path)


def _decode(data: bytes, *, path: Path) -> D1CampaignCheckpoint:
    if not data or not data.endswith(b"\n") or data[:-1].endswith(b"\n"):
        raise _error("corrupt_checkpoint", path, "checkpoint requires one canonical JSON line")
    try:
        payload = data[:-1].decode("utf-8")
        return parse_campaign_checkpoint(payload)
    except (UnicodeError, D1CampaignCheckpointError) as error:
        _raise("corrupt_checkpoint", path, f"checkpoint is corrupt: {error}", error)


def _receipt(path: Path, data: bytes) -> D1CampaignCheckpointReceipt:
    return D1CampaignCheckpointReceipt(path, len(data), hashlib.sha256(data).hexdigest())


def _receipt_from_path(path: Path) -> D1CampaignCheckpointReceipt:
    data = _read_regular(path)
    _decode(data, path=path)
    return _receipt(path, data)


def _write_all(descriptor: int, data: bytes, *, path: Path) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise _error(
                "partial_write",
                path,
                f"checkpoint write made no progress at {written}/{len(view)}",
            )
        written += count


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class D1CampaignCheckpointRepository:
    """Persist one canonical checkpoint with lock-serialized CAS replacement."""

    def __init__(self, path: Path) -> None:
        if not isinstance(cast(object, path), Path):
            raise TypeError("campaign checkpoint repository path must be a Path")
        _validate_parent(path)
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise _error("invalid_checkpoint_path", path, "target must be regular")
        self.path = path
        self._lock_path = path.parent / f".{path.name}.lock"

    def load(
        self, *, expected_sha256: str | None = None
    ) -> tuple[D1CampaignCheckpoint, D1CampaignCheckpointReceipt]:
        expected = _expected_digest(expected_sha256)
        data = _read_regular(self.path)
        receipt = _receipt(self.path, data)
        if expected is not None and receipt.sha256 != expected:
            raise _error("hash_mismatch", self.path, "checkpoint SHA-256 mismatch")
        return _decode(data, path=self.path), receipt

    def save(
        self,
        checkpoint: D1CampaignCheckpoint,
        *,
        expected_previous_sha256: str | None,
    ) -> D1CampaignCheckpointReceipt:
        if type(checkpoint) is not D1CampaignCheckpoint:
            raise TypeError("campaign checkpoint save requires D1CampaignCheckpoint")
        expected = _expected_digest(expected_previous_sha256)
        _validate_parent(self.path)
        lock = self._open_lock()
        temporary: Path | None = None
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            before = _read_if_present(self.path)
            self._admit_compare_and_swap(before, expected)
            payload = (serialize_campaign_checkpoint(checkpoint) + "\n").encode("utf-8")
            descriptor, name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            temporary = Path(name)
            try:
                _write_all(descriptor, payload, path=self.path)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            current = _read_if_present(self.path)
            if current != before:
                raise _error(
                    "forked_checkpoint",
                    self.path,
                    "checkpoint changed after compare-and-swap admission",
                )
            os.replace(temporary, self.path)
            temporary = None
            _fsync_parent(self.path)
            return _receipt_from_path(self.path)
        except D1CampaignCheckpointRepositoryError:
            raise
        except OSError as error:
            _raise("atomic_write", self.path, f"atomic checkpoint save failed: {error}", error)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            try:
                fcntl.flock(lock, fcntl.LOCK_UN)
            finally:
                os.close(lock)

    def _admit_compare_and_swap(self, current: bytes | None, expected: str | None) -> None:
        if current is None:
            if expected is not None:
                raise _error(
                    "compare_and_swap",
                    self.path,
                    "compare-and-swap expected a prior checkpoint but target is missing",
                )
            return
        _decode(current, path=self.path)
        actual = hashlib.sha256(current).hexdigest()
        if expected is None or actual != expected:
            raise _error(
                "compare_and_swap",
                self.path,
                "compare-and-swap previous checkpoint digest differs",
            )

    def _open_lock(self) -> int:
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._lock_path, flags, 0o600)
        except OSError as error:
            _raise("lock_io", self.path, f"cannot open repository lock: {error}", error)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise _error("lock_io", self.path, "repository lock is not a regular file")
        return descriptor
