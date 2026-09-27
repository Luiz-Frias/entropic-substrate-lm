"""Shared atomic content-addressed byte custody for D1 evidence adapters."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from pathlib import Path
from typing import NoReturn, cast

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash

__all__ = ["D1ContentAddressedCustodyError", "D1ContentAddressedCustodyStore"]

_CHUNK_SIZE = 1024 * 1024


class D1ContentAddressedCustodyError(RuntimeError):
    """The byte store is unsafe, corrupt, closed, or not durably published."""

    def __init__(self, *, code: str, path: Path, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = {"code": code, "path": str(path)}
        super().__init__(f"D1 content-addressed custody {code} at {path}: {detail}")


def _error(code: str, path: Path, detail: str) -> D1ContentAddressedCustodyError:
    return D1ContentAddressedCustodyError(code=code, path=path, detail=detail)


def _raise(code: str, path: Path, detail: str, cause: Exception) -> NoReturn:
    raise _error(code, path, detail) from cause


def _sha256(value: str, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _open_root(root: Path) -> tuple[Path, int]:
    if not isinstance(cast(object, root), Path):
        raise TypeError("content-addressed custody root must be a Path")
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
        raise _error("missing_record", path, "content-addressed record is absent") from error
    if stat.S_ISLNK(info.st_mode):
        raise _error("symlink_record", path, "content-addressed record symlink is forbidden")
    if not stat.S_ISREG(info.st_mode):
        raise _error("invalid_record", path, "content-addressed record must be regular")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=root_fd)
    except OSError as error:
        _raise("record_io", path, f"cannot open content-addressed record: {error}", error)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _error("invalid_record", path, "opened record is not regular")
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


class D1ContentAddressedCustodyStore:
    """Publish immutable bytes under their SHA-256 with durable race-safe reuse."""

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
            raise _error("closed_store", self.root, "content-addressed custody store is closed")
        return self._root_fd

    @staticmethod
    def _name(digest: str) -> str:
        return f"{digest}.json"

    def _target(self, digest: str) -> Path:
        return self.root / self._name(digest)

    @staticmethod
    def _artifact(target: Path, digest: str) -> ArtifactIdentity:
        return ArtifactIdentity(str(target), ContentHash("sha256", digest))

    def publish(self, payload: bytes, *, expected_sha256: str) -> ArtifactIdentity:
        if type(payload) is not bytes or not payload:
            raise ValueError("content-addressed custody payload must be nonempty bytes")
        expected = _sha256(expected_sha256, label="expected_sha256")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise _error("hash_mismatch", self.root, "payload hash differs from expected hash")
        root_fd = self._descriptor()
        name, target = self._name(digest), self._target(digest)
        current = _read_if_present(root_fd, name, target)
        if current is not None:
            return self._admit_durable_existing(name, target, current, payload, digest)
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
                current = _read_regular(root_fd, name, target, synchronize=True)
                return self._admit_durable_existing(name, target, current, payload, digest)
            os.unlink(temporary, dir_fd=root_fd)
            temporary_present = False
            os.fsync(root_fd)
            current = _read_regular(root_fd, name, target)
            return self._admit_existing(target, current, payload, digest)
        except D1ContentAddressedCustodyError:
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
        self, name: str, target: Path, current: bytes, expected: bytes, digest: str
    ) -> ArtifactIdentity:
        if current != expected:
            current = _read_regular(self._descriptor(), name, target, synchronize=True)
        artifact = self._admit_existing(target, current, expected, digest)
        os.fsync(self._descriptor())
        return artifact

    @staticmethod
    def _admit_existing(
        target: Path, current: bytes, expected: bytes, digest: str
    ) -> ArtifactIdentity:
        if current != expected or hashlib.sha256(current).hexdigest() != digest:
            raise _error("content_collision", target, "content-addressed collision is immutable")
        return D1ContentAddressedCustodyStore._artifact(target, digest)

    def read(self, artifact: ArtifactIdentity) -> bytes:
        root_fd = self._descriptor()
        if type(artifact) is not ArtifactIdentity or artifact.content_hash.algorithm != "sha256":
            raise _error("artifact_identity", self.root, "artifact must be SHA-256 identity")
        digest = _sha256(artifact.content_hash.digest, label="artifact SHA-256")
        target = self._target(digest)
        if artifact.path != str(target):
            raise _error("artifact_identity", target, "artifact path differs from content address")
        data = _read_regular(root_fd, self._name(digest), target)
        if hashlib.sha256(data).hexdigest() != digest:
            raise _error("hash_mismatch", target, "artifact SHA-256 differs from stored bytes")
        return data
