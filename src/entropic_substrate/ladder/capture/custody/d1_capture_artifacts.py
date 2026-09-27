"""Filesystem identities and deferred DVC bindings for D1 capture artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final

__all__ = [
    "D1ArtifactIdentity",
    "D1ArtifactRole",
    "D1CaptureArtifactError",
    "D1DvcBinding",
    "D1DvcBindingState",
    "identify_d1_artifact",
    "verify_d1_artifact",
]

_CHUNK_SIZE: Final = 1024 * 1024


class D1CaptureArtifactError(RuntimeError):
    """A local artifact cannot be identified without ambiguity."""


def _hex(value: str | None, length: int, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != length
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be {length} lowercase hexadecimal characters")


def _normalized_path(value: str, label: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty path")
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or path.as_posix() != value or ".." in path.parts:
        raise ValueError(f"{label} must be a normalized repository-relative POSIX path")


class D1DvcBindingState(StrEnum):
    PENDING = "pending"
    BOUND = "bound"


@dataclass(frozen=True, slots=True)
class D1DvcBinding:
    """Placeholder or later materialized DVC pointer/object identity."""

    state: D1DvcBindingState
    pointer_path: str
    pointer_sha256: str | None
    object_id: str | None

    def __post_init__(self) -> None:
        if type(self.state) is not D1DvcBindingState:
            raise TypeError("DVC binding state must be nominal")
        _normalized_path(self.pointer_path, "DVC pointer_path")
        if not self.pointer_path.endswith(".dvc"):
            raise ValueError("DVC pointer_path must name a .dvc file")
        if self.state is D1DvcBindingState.PENDING:
            if self.pointer_sha256 is not None or self.object_id is not None:
                raise ValueError("pending DVC binding must retain null pointer/object placeholders")
            return
        _hex(self.pointer_sha256, 64, "DVC pointer_sha256")
        if type(self.object_id) is not str:
            raise ValueError("bound DVC object_id must be present")
        object_digest = self.object_id.removesuffix(".dir")
        _hex(object_digest, 32, "DVC object_id")

    @classmethod
    def pending(cls, pointer_path: str) -> D1DvcBinding:
        return cls(
            state=D1DvcBindingState.PENDING,
            pointer_path=pointer_path,
            pointer_sha256=None,
            object_id=None,
        )

    def bind(self, *, pointer_sha256: str, object_id: str) -> D1DvcBinding:
        if self.state is not D1DvcBindingState.PENDING:
            raise ValueError("DVC identity is already bound")
        return D1DvcBinding(
            state=D1DvcBindingState.BOUND,
            pointer_path=self.pointer_path,
            pointer_sha256=pointer_sha256,
            object_id=object_id,
        )

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "pointer_path": self.pointer_path,
            "pointer_sha256": self.pointer_sha256,
            "state": self.state.value,
        }


class D1ArtifactRole(StrEnum):
    JOURNAL = "authoritative_journal"
    CHECKPOINT = "terminal_checkpoint"
    NUMERICAL_DEFECT_CHECKPOINT = "numerical_defect_prior_checkpoint"
    BILLING_SIDECAR = "provider_billing_sidecar"
    QUARANTINED_SIBLING = "quarantined_sibling"


@dataclass(frozen=True, slots=True)
class D1ArtifactIdentity:
    """Exact local bytes plus their deferred DVC address."""

    normalized_path: str
    role: D1ArtifactRole
    size_bytes: int
    md5: str
    sha256: str
    dvc: D1DvcBinding

    def __post_init__(self) -> None:
        _normalized_path(self.normalized_path, "artifact normalized_path")
        if type(self.role) is not D1ArtifactRole:
            raise TypeError("artifact role must be nominal")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("artifact size_bytes must be positive")
        _hex(self.md5, 32, "artifact md5")
        _hex(self.sha256, 64, "artifact sha256")
        if type(self.dvc) is not D1DvcBinding:
            raise TypeError("artifact requires a D1DvcBinding")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "dvc": self.dvc.manifest,
            "md5": self.md5,
            "path": self.normalized_path,
            "role": self.role.value,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _root(value: str | os.PathLike[str]) -> tuple[Path, Path]:
    lexical = Path(os.path.abspath(os.fspath(value)))
    if lexical.is_symlink() or not lexical.exists() or not lexical.is_dir():
        raise D1CaptureArtifactError(f"capture root is not a real directory: {lexical}")
    return lexical, lexical.resolve(strict=True)


def _regular_path(root: str | os.PathLike[str], value: str | os.PathLike[str]) -> tuple[Path, str]:
    lexical_root, resolved_root = _root(root)
    target = Path(os.path.abspath(os.fspath(value)))
    try:
        info = target.lstat()
    except FileNotFoundError as error:
        raise D1CaptureArtifactError(f"capture artifact does not exist: {target}") from error
    if stat.S_ISLNK(info.st_mode):
        raise D1CaptureArtifactError(f"capture artifact is a symlink: {target}")
    cursor = target.parent
    while cursor != lexical_root:
        if cursor.is_symlink():
            raise D1CaptureArtifactError(f"capture artifact parent is a symlink: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    resolved = target.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise D1CaptureArtifactError(f"capture artifact is outside root: {target}") from error
    if not stat.S_ISREG(info.st_mode):
        raise D1CaptureArtifactError(f"capture artifact is not a regular file: {target}")
    normalized = relative.as_posix()
    _normalized_path(normalized, "artifact normalized_path")
    return resolved, normalized


def _hash_handle(handle: BinaryIO) -> tuple[int, str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    while chunk := handle.read(_CHUNK_SIZE):
        size += len(chunk)
        md5.update(chunk)
        sha256.update(chunk)
    if size < 1:
        raise D1CaptureArtifactError("capture artifact is empty")
    return size, md5.hexdigest(), sha256.hexdigest()


def identify_d1_artifact(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    *,
    role: D1ArtifactRole,
    dvc_pointer_path: str,
) -> D1ArtifactIdentity:
    """Read and hash one real file; missing existing files are never skipped."""
    if type(role) is not D1ArtifactRole:
        raise TypeError("role must be a D1ArtifactRole")
    target, normalized = _regular_path(root, path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as error:
        raise D1CaptureArtifactError(f"cannot open capture artifact: {target}") from error
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise D1CaptureArtifactError(f"opened capture artifact is not regular: {target}")
        size, md5, sha256 = _hash_handle(handle)
    return D1ArtifactIdentity(
        normalized_path=normalized,
        role=role,
        size_bytes=size,
        md5=md5,
        sha256=sha256,
        dvc=D1DvcBinding.pending(dvc_pointer_path),
    )


def verify_d1_artifact(
    root: str | os.PathLike[str], identity: D1ArtifactIdentity
) -> D1ArtifactIdentity:
    """Re-read bytes and reject any drift from a previously captured identity."""
    if type(identity) is not D1ArtifactIdentity:
        raise TypeError("identity must be a D1ArtifactIdentity")
    target = Path(root) / identity.normalized_path
    actual = identify_d1_artifact(
        root,
        target,
        role=identity.role,
        dvc_pointer_path=identity.dvc.pointer_path,
    )
    if (
        actual.normalized_path != identity.normalized_path
        or actual.size_bytes != identity.size_bytes
        or actual.md5 != identity.md5
        or actual.sha256 != identity.sha256
    ):
        raise D1CaptureArtifactError(
            f"capture artifact identity changed: {identity.normalized_path}"
        )
    return identity
