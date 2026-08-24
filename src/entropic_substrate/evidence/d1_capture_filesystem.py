"""Direct local-custody evidence export for the D1 capture application."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1CaptureArtifactError,
    identify_d1_artifact,
    verify_d1_artifact,
)
from entropic_substrate.evidence.d1_capture_ports import D1CaptureTransportError

__all__ = ["LocalD1EvidenceExportSource"]


def _error(code: str, path: str, detail: str) -> D1CaptureTransportError:
    return D1CaptureTransportError(code=code, path=path, detail=detail)


def _real_source_root(root: Path) -> Path:
    lexical = Path(os.path.abspath(root))
    if lexical.is_symlink() or not lexical.is_dir():
        raise _error("source_root", str(lexical), "must be an existing real directory")
    return lexical


def _prepare_destination_root(root: Path) -> Path:
    lexical = Path(os.path.abspath(root))
    if lexical.exists() and lexical.is_symlink():
        raise _error("destination_boundary", str(lexical), "capture root may not be a symlink")
    lexical.mkdir(parents=True, exist_ok=True)
    if not lexical.is_dir() or lexical.is_symlink():
        raise _error("destination_boundary", str(lexical), "capture root must be a real directory")
    return lexical


def _prepare_parent(root: Path, relative: str) -> Path:
    current = root
    for part in Path(relative).parent.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise _error(
                "destination_boundary", relative, f"destination parent is a symlink: {current}"
            )
        current.mkdir(exist_ok=True)
        if not current.is_dir():
            raise _error(
                "destination_boundary",
                relative,
                f"destination parent is not a directory: {current}",
            )
    return current


def _same_identity(actual: D1ArtifactIdentity, expected: D1ArtifactIdentity) -> bool:
    return (
        actual.normalized_path == expected.normalized_path
        and actual.role is expected.role
        and actual.size_bytes == expected.size_bytes
        and actual.md5 == expected.md5
        and actual.sha256 == expected.sha256
    )


def _validate_existing(root: Path, artifact: D1ArtifactIdentity) -> None:
    target = root / artifact.normalized_path
    if not target.exists() and not target.is_symlink():
        return
    try:
        actual = identify_d1_artifact(
            root,
            target,
            role=artifact.role,
            dvc_pointer_path=artifact.dvc.pointer_path,
        )
    except (D1CaptureArtifactError, OSError) as error:
        raise _error("destination_boundary", artifact.normalized_path, str(error)) from error
    if not _same_identity(actual, artifact):
        raise _error(
            "destination_drift",
            artifact.normalized_path,
            "existing captured bytes differ from the admitted source identity",
        )


def _copy_atomically(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".d1-capture-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with (
            os.fdopen(source_descriptor, "rb") as source_handle,
            os.fdopen(descriptor, "wb") as destination_handle,
        ):
            if not stat.S_ISREG(os.fstat(source_handle.fileno()).st_mode):
                raise OSError(f"source is no longer a regular file: {source}")
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class LocalD1EvidenceExportSource:
    """Copy exact bytes from a directly held local/RunPod export directory."""

    def __init__(self, source_root: str | os.PathLike[str]) -> None:
        self._source_root = _real_source_root(Path(source_root))

    def export(
        self,
        *,
        artifacts: tuple[D1ArtifactIdentity, ...],
        destination_root: Path,
    ) -> tuple[D1ArtifactIdentity, ...]:
        if type(artifacts) is not tuple or not artifacts:
            raise TypeError("D1 evidence export requires a nonempty artifact tuple")
        if len({artifact.normalized_path for artifact in artifacts}) != len(artifacts):
            raise ValueError("D1 evidence export artifact paths must be unique")
        destination = _prepare_destination_root(destination_root)
        for artifact in artifacts:
            try:
                verify_d1_artifact(self._source_root, artifact)
            except (D1CaptureArtifactError, OSError) as error:
                raise _error("source_artifact", artifact.normalized_path, str(error)) from error
            parent = _prepare_parent(destination, artifact.normalized_path)
            _validate_existing(destination, artifact)
            try:
                _copy_atomically(
                    self._source_root / artifact.normalized_path,
                    parent / Path(artifact.normalized_path).name,
                )
                verify_d1_artifact(destination, artifact)
            except (D1CaptureArtifactError, OSError) as error:
                raise _error("source_artifact", artifact.normalized_path, str(error)) from error
        return artifacts
