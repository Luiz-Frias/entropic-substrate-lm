"""Atomic content-addressed filesystem custody for finalized D1 cells."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from pathlib import Path
from typing import NoReturn, cast

from entropic_substrate.d1_finalized_cell_custody import (
    D1FinalizedCellCustodyError,
    D1FinalizedCellCustodyRecord,
    parse_d1_finalized_cell_custody,
    serialize_d1_finalized_cell_custody,
)
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash

_CHUNK_SIZE = 1024 * 1024


class D1FinalizedCellCustodyRepositoryError(RuntimeError):
    """Finalized-cell storage is unsafe, corrupt, or not durably published."""

    def __init__(self, *, code: str, path: Path, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = {"code": code, "path": str(path)}
        super().__init__(f"D1 finalized-cell repository {code} at {path}: {detail}")


def _error(code: str, path: Path, detail: str) -> D1FinalizedCellCustodyRepositoryError:
    return D1FinalizedCellCustodyRepositoryError(code=code, path=path, detail=detail)


def _raise(code: str, path: Path, detail: str, cause: Exception) -> NoReturn:
    raise _error(code, path, detail) from cause


def _open_root(root: Path) -> tuple[Path, int]:
    if not isinstance(cast(object, root), Path):
        raise TypeError("finalized-cell repository root must be a Path")
    absolute = Path(os.path.abspath(root))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        _raise("invalid_root", absolute, "root must be a real non-symlink directory", error)
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise _error("invalid_root", absolute, "opened root is not a directory")
    return absolute, descriptor


def _read_regular(root_fd: int, name: str, path: Path, *, synchronize: bool = False) -> bytes:
    try:
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError as error:
        raise _error("missing_record", path, "finalized-cell record is absent") from error
    if stat.S_ISLNK(info.st_mode):
        raise _error("symlink_record", path, "finalized-cell symlink is forbidden")
    if not stat.S_ISREG(info.st_mode):
        raise _error("invalid_record", path, "finalized-cell record must be regular")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=root_fd)
    except OSError as error:
        _raise("record_io", path, f"cannot open finalized-cell record: {error}", error)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _error("invalid_record", path, "opened finalized-cell record is not regular")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, _CHUNK_SIZE):
            chunks.append(chunk)
        if synchronize:
            os.fsync(descriptor)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_if_present(root_fd: int, name: str, path: Path) -> bytes | None:
    try:
        os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return _read_regular(root_fd, name, path)


def _decode(data: bytes, path: Path) -> D1FinalizedCellCustodyRecord:
    try:
        payload = data.decode("utf-8")
        record = parse_d1_finalized_cell_custody(payload)
    except (UnicodeError, D1FinalizedCellCustodyError) as error:
        _raise("corrupt_record", path, f"finalized-cell bytes are invalid: {error}", error)
    if hashlib.sha256(data).hexdigest() != record.record_digest:
        raise _error("hash_mismatch", path, "record bytes differ from their content address")
    return record


def _write_all(descriptor: int, data: bytes, target: Path) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        count = os.write(descriptor, view[offset:])
        if count <= 0:
            raise _error(
                "partial_write",
                target,
                f"partial write made no progress at {offset}/{len(view)}",
            )
        offset += count


def _create_temporary(root_fd: int, digest: str, target: Path) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(128):
        name = f".{digest}.{secrets.token_hex(8)}.tmp"
        try:
            return os.open(name, flags, 0o600, dir_fd=root_fd), name
        except FileExistsError:
            continue
    raise _error("temporary_collision", target, "cannot allocate a unique sibling temp file")


class D1FinalizedCellCustodyRepository:
    """Publish immutable finalized-cell records under their canonical SHA-256."""

    def __init__(self, root: Path) -> None:
        self.root, self._root_fd = _open_root(root)

    def close(self) -> None:
        descriptor = self._root_fd
        if descriptor >= 0:
            self._root_fd = -1
            os.close(descriptor)

    def __del__(self) -> None:
        descriptor = getattr(self, "_root_fd", -1)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _descriptor(self) -> int:
        if self._root_fd < 0:
            raise _error("closed_repository", self.root, "repository directory is closed")
        return self._root_fd

    @staticmethod
    def _name(digest: str) -> str:
        return f"{digest}.json"

    def _target(self, digest: str) -> Path:
        return self.root / self._name(digest)

    @staticmethod
    def _artifact(target: Path, digest: str) -> ArtifactIdentity:
        return ArtifactIdentity(str(target), ContentHash("sha256", digest))

    def save(self, record: D1FinalizedCellCustodyRecord) -> ArtifactIdentity:
        if type(record) is not D1FinalizedCellCustodyRecord:
            raise TypeError("finalized-cell save requires its nominal record")
        record.require_factory_attested()
        root_fd = self._descriptor()
        payload = serialize_d1_finalized_cell_custody(record).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record.record_digest:
            raise _error("hash_mismatch", self.root, "record digest differs before persistence")
        name, target = self._name(digest), self._target(digest)
        current = _read_if_present(root_fd, name, target)
        if current is not None:
            return self._admit_durable_existing(name, target, payload, digest)
        descriptor, temporary = _create_temporary(root_fd, digest, target)
        temporary_present = True
        try:
            try:
                _write_all(descriptor, payload, target)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                os.link(
                    temporary,
                    name,
                    src_dir_fd=root_fd,
                    dst_dir_fd=root_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                os.unlink(temporary, dir_fd=root_fd)
                temporary_present = False
                return self._admit_durable_existing(name, target, payload, digest)
            os.unlink(temporary, dir_fd=root_fd)
            temporary_present = False
            os.fsync(root_fd)
            persisted = _read_regular(root_fd, name, target)
            return self._admit_existing(target, persisted, payload, digest)
        except D1FinalizedCellCustodyRepositoryError:
            raise
        except OSError as error:
            _raise("atomic_write", target, f"atomic publication failed: {error}", error)
        finally:
            if temporary_present:
                try:
                    os.unlink(temporary, dir_fd=root_fd)
                    os.fsync(root_fd)
                except FileNotFoundError:
                    pass

    def _admit_durable_existing(
        self, name: str, target: Path, expected: bytes, digest: str
    ) -> ArtifactIdentity:
        current = _read_regular(self._descriptor(), name, target, synchronize=True)
        artifact = self._admit_existing(target, current, expected, digest)
        os.fsync(self._descriptor())
        return artifact

    def _admit_existing(
        self, target: Path, current: bytes, expected: bytes, digest: str
    ) -> ArtifactIdentity:
        if current != expected or hashlib.sha256(current).hexdigest() != digest:
            raise _error("content_collision", target, "content-addressed collision is immutable")
        _decode(current, target).require_factory_attested()
        return self._artifact(target, digest)

    def load(self, artifact: ArtifactIdentity) -> D1FinalizedCellCustodyRecord:
        if type(artifact) is not ArtifactIdentity:
            raise TypeError("finalized-cell load requires ArtifactIdentity")
        if artifact.content_hash.algorithm != "sha256":
            raise _error("artifact_identity", self.root, "artifact must use SHA-256")
        target = self._target(artifact.content_hash.digest)
        if artifact.path != str(target):
            raise _error("artifact_identity", target, "artifact path differs from content address")
        data = _read_regular(
            self._descriptor(),
            self._name(artifact.content_hash.digest),
            target,
        )
        if hashlib.sha256(data).hexdigest() != artifact.content_hash.digest:
            raise _error("hash_mismatch", target, "artifact SHA-256 differs from stored bytes")
        return _decode(data, target)
