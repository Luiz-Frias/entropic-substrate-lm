"""Descriptor-bound artifact snapshots beneath a symlink-free D1 export root."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

__all__ = [
    "D1ArtifactSnapshotError",
    "normalize_d1_artifact_root",
    "normalize_d1_relative_artifact_path",
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
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


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
    except OSError as error:
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


def snapshot_d1_artifact(
    root: Path,
    relative_path: object,
    *,
    label: str,
) -> tuple[Path, int, str]:
    """Hash and size one regular file from the same no-follow descriptor."""
    admitted_root = _absolute_path(root, label="D1 artifact root")
    relative = normalize_d1_relative_artifact_path(relative_path, label=label)
    parts = PurePosixPath(relative).parts
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = _open_directory_path(admitted_root, label="D1 artifact root")
        for part in parts[:-1]:
            next_descriptor = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_descriptor
        file_fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=directory_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            _fail(f"{label} artifact must be a regular file")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(file_fd, _CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
        return admitted_root.joinpath(*parts), size, digest.hexdigest()
    except OSError as error:
        _fail(f"{label} artifact could not be opened beneath its root without a symlink", error)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)
