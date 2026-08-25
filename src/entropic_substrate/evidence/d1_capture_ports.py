"""Ports and transport values for the D1 authoritative capture application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1DvcBinding,
    D1DvcBindingState,
)

__all__ = [
    "D1CaptureTransportError",
    "D1CloudObjectVerifierPort",
    "D1DvcCommandPort",
    "D1DvcPointerIdentity",
    "D1EvidenceExportSourcePort",
    "D1GitPointerCommitEvidence",
    "D1GitPointerStatusVerifierPort",
]


class D1CaptureTransportError(RuntimeError):
    """Typed fail-loud refusal at a capture infrastructure boundary."""

    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = MappingProxyType({"code": code, "path": path})
        super().__init__(f"D1 capture transport {code} at {path}: {detail}")


@dataclass(frozen=True, slots=True)
class D1DvcPointerIdentity:
    """Filesystem-read identity of one materialized DVC pointer and object."""

    pointer_path: str
    pointer_sha256: str
    object_id: str

    def __post_init__(self) -> None:
        D1DvcBinding(
            state=D1DvcBindingState.BOUND,
            pointer_path=self.pointer_path,
            pointer_sha256=self.pointer_sha256,
            object_id=self.object_id,
        )

    @property
    def binding(self) -> D1DvcBinding:
        return D1DvcBinding(
            state=D1DvcBindingState.BOUND,
            pointer_path=self.pointer_path,
            pointer_sha256=self.pointer_sha256,
            object_id=self.object_id,
        )


@dataclass(frozen=True, slots=True)
class D1GitPointerCommitEvidence:
    """Git-read proof that one commit contains the exact pointer and object identity."""

    commit: str
    pointer_path: str
    pointer_sha256: str
    object_id: str

    def __post_init__(self) -> None:
        if (
            type(self.commit) is not str
            or not 7 <= len(self.commit) <= 64
            or self.commit != self.commit.lower()
            or any(character not in "0123456789abcdef" for character in self.commit)
        ):
            raise ValueError("pointer commit must be lowercase hexadecimal")
        D1DvcPointerIdentity(self.pointer_path, self.pointer_sha256, self.object_id)

    @property
    def binding(self) -> D1DvcBinding:
        return D1DvcBinding(
            state=D1DvcBindingState.BOUND,
            pointer_path=self.pointer_path,
            pointer_sha256=self.pointer_sha256,
            object_id=self.object_id,
        )


class D1EvidenceExportSourcePort(Protocol):
    """Materialize exact run-custody bytes without consulting MLflow."""

    def export(
        self,
        *,
        artifacts: tuple[D1ArtifactIdentity, ...],
        destination_root: Path,
    ) -> tuple[D1ArtifactIdentity, ...]: ...


class D1DvcCommandPort(Protocol):
    """The mutating DVC add/push boundary."""

    def add(self, *, tracked_path: Path, pointer_path: Path) -> None: ...

    def push(self, *, pointer_path: Path) -> None: ...


class D1CloudObjectVerifierPort(Protocol):
    """Affirmatively prove the remote contains the pointer's object."""

    def object_is_present(self, *, pointer_path: Path, object_id: str) -> bool: ...


class D1GitPointerStatusVerifierPort(Protocol):
    """Resolve the commit that contains these exact pointer bytes."""

    def committed_pointer(
        self,
        *,
        pointer_path: Path,
        pointer_bytes: bytes,
        pointer_sha256: str,
        object_id: str,
    ) -> D1GitPointerCommitEvidence | None: ...
