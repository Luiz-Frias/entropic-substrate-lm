"""Descriptor-bound artifact snapshots beneath a symlink-free D1 export root."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash

__all__ = [
    "D1ArtifactSnapshotError",
    "artifact_from_mapping",
    "artifact_mapping",
    "normalize_d1_artifact_directory",
    "normalize_d1_artifact_root",
    "normalize_d1_relative_artifact_path",
    "object_fields",
    "publish_d1_atomic_bytes",
    "read_bound_artifact",
    "read_d1_artifact_bytes",
    "read_json_artifact",
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


def _descend_directories(descriptor: int, parts: Sequence[str]) -> int:
    """Walk ``parts`` beneath ``descriptor`` without following symlinks, owning every handle.

    Consumes ``descriptor``: on return it is closed and the result names the last directory; on
    any raise every descriptor the walk held is closed. Callers must therefore never close or
    track the argument afterwards, only the result.
    """
    try:
        for part in parts:
            next_descriptor = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_directory_path(path: Path, *, label: str) -> int:
    descriptor: int | None = None
    try:
        anchor_descriptor = os.open(path.anchor, _DIRECTORY_FLAGS)
        descriptor = _descend_directories(anchor_descriptor, path.parts[1:])
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


def normalize_d1_artifact_directory(root: Path, value: object, *, label: str) -> Path:
    """Admit one existing directory named by a canonical relative path beneath ``root``."""
    relative = normalize_d1_relative_artifact_path(value, label=label)
    return normalize_d1_artifact_root(root / relative)


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
        root_fd = _open_directory_path(admitted_root, label="D1 artifact root")
        directory_fd = _descend_directories(root_fd, parts[:-1])
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
        root_fd = _open_directory_path(admitted_root, label="D1 artifact root")
        directory_fd = _descend_directories(root_fd, parts[:-1])
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


def read_bound_artifact(root: Path, artifact: ArtifactIdentity) -> bytes:
    """Read and hash the same no-follow descriptor; never trust a caller summary."""
    if type(artifact) is not ArtifactIdentity or artifact.content_hash.algorithm != "sha256":
        raise ValueError("ladder artifacts require exact SHA-256 identities")
    name = artifact.path
    if Path(name).is_absolute():
        name = relative_d1_artifact_path(root, Path(name), label="ladder artifact")
    _, data, observed = read_d1_artifact_bytes(root, name, label="ladder artifact")
    if observed != artifact.content_hash.digest:
        raise ValueError("ladder artifact digest mismatch")
    return data


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate ladder JSON key")
        result[key] = value
    return result


def object_fields(value: object, fields: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(fields.split()):
        raise ValueError("ladder artifact fields must be exact")
    return cast(dict[str, object], value)


def read_json_artifact(root: Path, artifact: ArtifactIdentity) -> dict[str, object]:
    value = json.loads(read_bound_artifact(root, artifact), object_pairs_hook=_unique)
    if type(value) is not dict:
        raise ValueError("ladder artifact must be a JSON object")
    return cast(dict[str, object], value)


def artifact_from_mapping(value: object) -> ArtifactIdentity:
    row = object_fields(value, "path sha256")
    if type(row["path"]) is not str or type(row["sha256"]) is not str:
        raise ValueError("artifact path and sha256 must be strings")
    return ArtifactIdentity(row["path"], ContentHash("sha256", row["sha256"]))


def artifact_mapping(value: ArtifactIdentity) -> dict[str, str]:
    return {"path": value.path, "sha256": value.content_hash.digest}
