"""Capture-preparation codec adapter over shared content-addressed custody."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NoReturn

from entropic_substrate.d1_content_addressed_custody import (
    D1ContentAddressedCustodyError,
    D1ContentAddressedCustodyStore,
)
from entropic_substrate.evidence.d1_capture_preparation_custody import (
    D1CapturePreparationCustodyError,
    D1CapturePreparationCustodyRecord,
    parse_d1_capture_preparation_custody,
    serialize_d1_capture_preparation_custody,
)
from entropic_substrate.evidence.lifecycle import ArtifactIdentity

__all__ = [
    "D1CapturePreparationCustodyRepository",
    "D1CapturePreparationCustodyRepositoryError",
]


class D1CapturePreparationCustodyRepositoryError(RuntimeError):
    """Preparation custody storage is unsafe, corrupt, or not durable."""

    def __init__(self, *, code: str, path: Path, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = {"code": code, "path": str(path)}
        super().__init__(f"D1 capture-preparation repository {code} at {path}: {detail}")


def _translate(error: D1ContentAddressedCustodyError) -> NoReturn:
    raise D1CapturePreparationCustodyRepositoryError(
        code=error.code,
        path=error.path,
        detail=str(error),
    ) from error


def _decode(data: bytes, path: Path) -> D1CapturePreparationCustodyRecord:
    try:
        record = parse_d1_capture_preparation_custody(data.decode("utf-8"))
    except (UnicodeError, D1CapturePreparationCustodyError) as error:
        raise D1CapturePreparationCustodyRepositoryError(
            code="corrupt_record",
            path=path,
            detail=f"capture-preparation bytes are invalid: {error}",
        ) from error
    if hashlib.sha256(data).hexdigest() != record.record_sha256:
        raise D1CapturePreparationCustodyRepositoryError(
            code="hash_mismatch",
            path=path,
            detail="record bytes differ from their content address",
        )
    return record


class D1CapturePreparationCustodyRepository:
    """Persist COMMIT_REQUIRED records through the shared atomic byte store."""

    def __init__(self, root: Path) -> None:
        try:
            self._store = D1ContentAddressedCustodyStore(root)
        except D1ContentAddressedCustodyError as error:
            _translate(error)
        self.root = self._store.root

    def close(self) -> None:
        self._store.close()

    def save(self, record: D1CapturePreparationCustodyRecord) -> ArtifactIdentity:
        if type(record) is not D1CapturePreparationCustodyRecord:
            raise TypeError("capture-preparation save requires its nominal record")
        payload = serialize_d1_capture_preparation_custody(record).encode("utf-8")
        try:
            return self._store.publish(payload, expected_sha256=record.record_sha256)
        except D1ContentAddressedCustodyError as error:
            _translate(error)

    def load(self, artifact: ArtifactIdentity) -> D1CapturePreparationCustodyRecord:
        try:
            data = self._store.read(artifact)
        except D1ContentAddressedCustodyError as error:
            _translate(error)
        return _decode(data, Path(artifact.path))
