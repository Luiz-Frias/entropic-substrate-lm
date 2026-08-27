"""Descriptor-bound artifact snapshots beneath a symlink-free D1 export root."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash

__all__ = [
    "D1ArtifactSnapshotError",
    "normalize_d1_artifact_root",
    "normalize_d1_relative_artifact_path",
    "publish_d1_atomic_bytes",
    "read_d1_artifact_bytes",
    "relative_d1_artifact_path",
    "snapshot_d1_artifact",
]

_CHUNK_SIZE = 1024 * 1024
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class D1ArtifactSnapshotError(ValueError):
    """An artifact path cannot be bound to one regular file beneath its root."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1ArtifactSnapshotError(message)
    raise D1ArtifactSnapshotError(message) from error


def _absolute_path(value: Path, *, label: str) -> Path:
    if not isinstance(cast(object, value), Path):
        raise TypeError(f"{label} must be a Path")
    return Path(os.path.abspath(value))


def _open_directory_path(path: Path, *, label: str) -> int:
    descriptor: int | None = None
    try:
        descriptor = os.open(path.anchor, _DIRECTORY_FLAGS)
        for part in path.parts[1:]:
            next_descriptor = os.open(
                part,
                _DIRECTORY_FLAGS,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            _fail(f"{label} must be a real directory")
        return descriptor
    except (OSError, ValueError) as error:
        if descriptor is not None:
            os.close(descriptor)
        _fail(f"{label} could not be opened without a symlink", error)


def normalize_d1_artifact_root(value: Path) -> Path:
    """Admit one existing absolute directory path without following symlinks."""
    root = _absolute_path(value, label="D1 artifact root")
    descriptor = _open_directory_path(root, label="D1 artifact root")
    os.close(descriptor)
    return root


def normalize_d1_relative_artifact_path(value: object, *, label: str) -> str:
    """Admit one canonical nonempty relative POSIX artifact path."""
    if type(value) is not str or not value or "\\" in value:
        _fail(f"{label} path must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        _fail(f"{label} path must remain normalized below its artifact root")
    return value


def relative_d1_artifact_path(root: Path, value: Path, *, label: str) -> str:
    """Derive a canonical relative path without using resolution for containment."""
    admitted_root = normalize_d1_artifact_root(root)
    artifact = _absolute_path(value, label=f"{label} artifact")
    try:
        relative = artifact.relative_to(admitted_root).as_posix()
    except ValueError as error:
        _fail(f"{label} path escapes its artifact root", error)
    return normalize_d1_relative_artifact_path(relative, label=label)


def _measure_file_descriptor(descriptor: int, *, label: str) -> tuple[int, str]:
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        _fail(f"{label} artifact must be a regular file")
    digest = hashlib.sha256()
    size = 0
    while chunk := os.read(descriptor, _CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


def _open_artifact_descriptor(
    root: Path,
    relative_path: object,
    *,
    label: str,
) -> tuple[Path, int, int]:
    admitted_root = _absolute_path(root, label="D1 artifact root")
    relative = normalize_d1_relative_artifact_path(relative_path, label=label)
    parts = PurePosixPath(relative).parts
    directory_fd: int | None = None
    try:
        directory_fd = _open_directory_path(admitted_root, label="D1 artifact root")
        for part in parts[:-1]:
            next_descriptor = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_descriptor
        file_fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=directory_fd)
        return admitted_root.joinpath(*parts), directory_fd, file_fd
    except (OSError, ValueError) as error:
        if directory_fd is not None:
            os.close(directory_fd)
        _fail(f"{label} artifact could not be opened beneath its root without a symlink", error)


def snapshot_d1_artifact(
    root: Path,
    relative_path: object,
    *,
    label: str,
) -> tuple[Path, int, str]:
    """Hash and size one regular file from the same no-follow descriptor."""
    path, directory_fd, file_fd = _open_artifact_descriptor(
        root,
        relative_path,
        label=label,
    )
    try:
        size, digest = _measure_file_descriptor(file_fd, label=label)
        return path, size, digest
    finally:
        os.close(file_fd)
        os.close(directory_fd)


def read_d1_artifact_bytes(
    root: Path,
    relative_path: object,
    *,
    label: str,
    max_bytes: int | None = None,
) -> tuple[Path, bytes, str]:
    """Read and hash one regular artifact through the same no-follow descriptor."""
    if max_bytes is not None:
        if type(max_bytes) is not int:
            raise TypeError(f"{label} max_bytes must be an integer or None")
        if max_bytes < 1:
            raise ValueError(f"{label} max_bytes must be positive")
    path, directory_fd, file_fd = _open_artifact_descriptor(
        root,
        relative_path,
        label=label,
    )
    try:
        try:
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode):
                _fail(f"{label} artifact must be a regular file")
            if max_bytes is not None and metadata.st_size > max_bytes:
                _fail(f"{label} artifact exceeds its byte limit")
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            size = 0
            while True:
                read_size = (
                    _CHUNK_SIZE if max_bytes is None else min(_CHUNK_SIZE, max_bytes - size + 1)
                )
                chunk = os.read(file_fd, read_size)
                if not chunk:
                    break
                size += len(chunk)
                if max_bytes is not None and size > max_bytes:
                    _fail(f"{label} artifact exceeds its byte limit")
                chunks.append(chunk)
                digest.update(chunk)
            return path, b"".join(chunks), digest.hexdigest()
        except OSError as error:
            _fail(f"{label} artifact descriptor read failed", error)
    finally:
        os.close(file_fd)
        os.close(directory_fd)


def _write_all(descriptor: int, payload: bytes, *, label: str) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            _fail(f"{label} atomic publication partial write stalled at {offset}/{len(view)}")
        offset += written


def _create_temporary(directory_fd: int, digest: str, *, label: str) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(128):
        name = f".{digest}.{secrets.token_hex(8)}.tmp"
        try:
            return os.open(name, flags, 0o600, dir_fd=directory_fd), name
        except FileExistsError:
            continue
    _fail(f"{label} atomic publication could not allocate a sibling temporary")


def publish_d1_atomic_bytes(
    root: Path,
    relative_path: object,
    payload: bytes,
    *,
    label: str,
) -> ArtifactIdentity:
    """Publish complete immutable bytes without replacing an occupied target."""
    if type(payload) is not bytes or not payload:
        raise TypeError(f"{label} atomic publication requires nonempty bytes")
    admitted_root = _absolute_path(root, label="D1 artifact root")
    relative = normalize_d1_relative_artifact_path(relative_path, label=label)
    parts = PurePosixPath(relative).parts
    directory_fd: int | None = None
    temporary_fd: int | None = None
    temporary_name: str | None = None
    published_fd: int | None = None
    digest = hashlib.sha256(payload).hexdigest()
    try:
        directory_fd = _open_directory_path(admitted_root, label="D1 artifact root")
        for part in parts[:-1]:
            next_descriptor = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_descriptor
        try:
            os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _fail(f"{label} atomic publication target is occupied")
        temporary_fd, temporary_name = _create_temporary(
            directory_fd,
            digest,
            label=label,
        )
        _write_all(temporary_fd, payload, label=label)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        os.link(
            temporary_name,
            parts[-1],
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_name = None
        os.fsync(directory_fd)
        published_fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=directory_fd)
        size, observed_digest = _measure_file_descriptor(published_fd, label=label)
        if (size, observed_digest) != (len(payload), digest):
            _fail(f"{label} atomic publication bytes differ after durable link")
        path = admitted_root.joinpath(*parts)
        return ArtifactIdentity(str(path), ContentHash("sha256", digest))
    except D1ArtifactSnapshotError:
        raise
    except OSError as error:
        _fail(f"{label} atomic publication failed: {error}", error)
    finally:
        if published_fd is not None:
            os.close(published_fd)
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_name is not None and directory_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        if directory_fd is not None:
            os.close(directory_fd)
