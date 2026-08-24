"""Durable atomic persistence adapter for validated D1 checkpoint mappings."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import tempfile
from collections.abc import Buffer
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import torch

from entropic_substrate.data.batching import SequenceSampler
from entropic_substrate.experiments.d1_execution import D1ExecutionProtocol
from entropic_substrate.experiments.d1_runs import D1RunSpec
from entropic_substrate.predictive_state.checkpoint_envelope import validate_d1_checkpoint
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem

__all__ = [
    "D1CheckpointArtifactReceipt",
    "D1CheckpointRepositoryError",
    "load_and_validate",
    "save_atomic",
]

_CHUNK_SIZE = 1024 * 1024


class D1CheckpointRepositoryError(RuntimeError):
    """Checkpoint bytes cannot be persisted or recovered without ambiguity."""


@dataclass(frozen=True, slots=True)
class D1CheckpointArtifactReceipt:
    """Detached identity of the durable artifact occupying ``path``."""

    path: Path
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.path), Path):
            raise TypeError("checkpoint receipt path must be a Path")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("checkpoint receipt size_bytes must be positive")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or self.sha256 != self.sha256.lower()
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("checkpoint receipt sha256 must be a lowercase SHA-256 digest")


class _ExactFdWriter(io.RawIOBase):
    """File-like torch sink that completes partial writes or fails on zero progress."""

    def __init__(self, fd: int) -> None:
        super().__init__()
        self._fd = fd

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def write(self, data: Buffer) -> int:
        if self.closed:
            raise ValueError("write to closed checkpoint temporary")
        view = memoryview(data)
        total = 0
        while total < len(view):
            written = os.write(self._fd, view[total:])
            if written <= 0:
                raise OSError(f"short checkpoint write made no progress at {total}/{len(view)}")
            total += written
        return total

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return os.lseek(self._fd, offset, whence)

    def tell(self) -> int:
        return os.lseek(self._fd, 0, os.SEEK_CUR)

    def fileno(self) -> int:
        return self._fd

    def flush(self) -> None:
        if self.closed:
            raise ValueError("flush of closed checkpoint temporary")

    def close(self) -> None:
        if not self.closed:
            os.close(self._fd)
        super().close()


def _target_path(value: str | os.PathLike[str]) -> Path:
    target = Path(value)
    parent = target.parent
    if not parent.exists() or not parent.is_dir() or parent.is_symlink():
        raise D1CheckpointRepositoryError(f"checkpoint parent is not a real directory: {parent}")
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise D1CheckpointRepositoryError(f"checkpoint target is not a regular file: {target}")
    return target


def _fsync_parent(parent: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(parent, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _open_regular(path: Path) -> int:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise D1CheckpointRepositoryError(f"checkpoint does not exist: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise D1CheckpointRepositoryError(f"checkpoint is not a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise D1CheckpointRepositoryError(f"cannot open checkpoint {path}: {exc}") from exc
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise D1CheckpointRepositoryError(f"opened checkpoint is not a regular file: {path}")
    return fd


def _receipt_from_handle(path: Path, handle: BinaryIO) -> D1CheckpointArtifactReceipt:
    handle.seek(0)
    digest = hashlib.sha256()
    size = 0
    while chunk := handle.read(_CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
    handle.seek(0)
    return D1CheckpointArtifactReceipt(path=path, size_bytes=size, sha256=digest.hexdigest())


def _receipt_from_path(path: Path) -> D1CheckpointArtifactReceipt:
    fd = _open_regular(path)
    with os.fdopen(fd, "rb") as handle:
        return _receipt_from_handle(path, handle)


def _require_expected_digest(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D1CheckpointRepositoryError("expected_sha256 must be a lowercase SHA-256 digest")
    return value


def save_atomic(
    target: str | os.PathLike[str], checkpoint: dict[str, object]
) -> D1CheckpointArtifactReceipt:
    """Durably replace ``target`` and then identify the artifact now at that path."""
    if type(checkpoint) is not dict:
        raise D1CheckpointRepositoryError("atomic checkpoint save requires a validated mapping")
    path = _target_path(target)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    writer: _ExactFdWriter | None = _ExactFdWriter(fd)
    try:
        torch.save(checkpoint, cast(BinaryIO, writer))
        writer.flush()
        os.fsync(writer.fileno())
        writer.close()
        writer = None
        os.replace(temporary, path)
        _fsync_parent(path.parent)
        return _receipt_from_path(path)
    except Exception as exc:
        raise D1CheckpointRepositoryError(
            f"atomic checkpoint save failed for {path}: {exc}"
        ) from exc
    finally:
        if writer is not None:
            try:
                writer.close()
            except OSError:
                pass
        temporary.unlink(missing_ok=True)


def load_and_validate(
    path: str | os.PathLike[str],
    *,
    system: D1PredictiveSystem,
    optimizer: torch.optim.Optimizer,
    sampler: SequenceSampler,
    run: D1RunSpec,
    protocol: D1ExecutionProtocol,
    for_resume: bool = False,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], D1CheckpointArtifactReceipt]:
    """Load one regular file, verify its receipt, and apply D1 admission."""
    if type(for_resume) is not bool:
        raise D1CheckpointRepositoryError("for_resume must be a bool")
    expected = _require_expected_digest(expected_sha256)
    target = Path(path)
    fd = _open_regular(target)
    with os.fdopen(fd, "rb") as handle:
        receipt = _receipt_from_handle(target, handle)
        if expected is not None and receipt.sha256 != expected:
            raise D1CheckpointRepositoryError(
                f"checkpoint SHA-256 mismatch: expected {expected}, got {receipt.sha256}"
            )
        try:
            raw = torch.load(handle, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise D1CheckpointRepositoryError(f"cannot load checkpoint {target}: {exc}") from exc
    validated = validate_d1_checkpoint(
        raw,
        system=system,
        optimizer=optimizer,
        sampler=sampler,
        run=run,
        protocol=protocol,
        for_resume=for_resume,
    )
    return validated, receipt
