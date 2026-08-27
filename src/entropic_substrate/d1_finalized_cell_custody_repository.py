"""Finalized-cell codec adapter over shared content-addressed byte custody."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NoReturn

from entropic_substrate.d1_content_addressed_custody import (
    D1ContentAddressedCustodyError,
    D1ContentAddressedCustodyStore,
)
from entropic_substrate.d1_finalized_cell_custody import (
    D1FinalizedCellCustodyError,
    D1FinalizedCellCustodyRecord,
    parse_d1_finalized_cell_custody,
    serialize_d1_finalized_cell_custody,
)
from entropic_substrate.evidence.lifecycle import ArtifactIdentity


class D1FinalizedCellCustodyRepositoryError(RuntimeError):
    """Finalized-cell storage is unsafe, corrupt, or not durably published."""

    def __init__(self, *, code: str, path: Path, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = {"code": code, "path": str(path)}
        super().__init__(f"D1 finalized-cell repository {code} at {path}: {detail}")


def _translate(error: D1ContentAddressedCustodyError) -> NoReturn:
    raise D1FinalizedCellCustodyRepositoryError(
        code=error.code,
        path=error.path,
        detail=str(error),
    ) from error


def _decode(data: bytes, path: Path) -> D1FinalizedCellCustodyRecord:
    try:
        payload = data.decode("utf-8")
        record = parse_d1_finalized_cell_custody(payload)
    except (UnicodeError, D1FinalizedCellCustodyError) as error:
        raise D1FinalizedCellCustodyRepositoryError(
            code="corrupt_record",
            path=path,
            detail=f"finalized-cell bytes are invalid: {error}",
        ) from error
    if hashlib.sha256(data).hexdigest() != record.record_digest:
        raise D1FinalizedCellCustodyRepositoryError(
            code="hash_mismatch",
            path=path,
            detail="record bytes differ from their content address",
        )
    return record


class D1FinalizedCellCustodyRepository:
    """Persist finalized-cell values through the shared atomic byte store."""

    def __init__(self, root: Path) -> None:
        try:
            self._store = D1ContentAddressedCustodyStore(root)
        except D1ContentAddressedCustodyError as error:
            _translate(error)
        self.root = self._store.root

    def close(self) -> None:
        self._store.close()

    def save(self, record: D1FinalizedCellCustodyRecord) -> ArtifactIdentity:
        if type(record) is not D1FinalizedCellCustodyRecord:
            raise TypeError("finalized-cell save requires its nominal record")
        record.require_factory_attested()
        payload = serialize_d1_finalized_cell_custody(record).encode("utf-8")
        try:
            return self._store.publish(payload, expected_sha256=record.record_digest)
        except D1ContentAddressedCustodyError as error:
            _translate(error)

    def load(self, artifact: ArtifactIdentity) -> D1FinalizedCellCustodyRecord:
        try:
            data = self._store.read(artifact)
        except D1ContentAddressedCustodyError as error:
            _translate(error)
        return _decode(data, Path(artifact.path))
