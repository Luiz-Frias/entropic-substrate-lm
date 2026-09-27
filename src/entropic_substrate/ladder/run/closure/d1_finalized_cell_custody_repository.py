"""Finalized-cell codec adapter over shared content-addressed byte custody."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import NoReturn

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.capture.custody.d1_content_addressed_custody import (
    D1ContentAddressedCustodyError,
    D1ContentAddressedCustodyStore,
)
from entropic_substrate.ladder.run.closure.d1_finalized_cell_custody import (
    D1FinalizedCellCustodyError,
    D1FinalizedCellCustodyRecord,
    parse_d1_finalized_cell_custody,
    serialize_d1_finalized_cell_custody,
)

_RECORD_NAME = re.compile(r"^([0-9a-f]{64})\.json$")


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

    def discover(
        self,
        *,
        exclude: tuple[ArtifactIdentity, ...] = (),
    ) -> tuple[tuple[D1FinalizedCellCustodyRecord, ArtifactIdentity], ...]:
        """Enumerate a closed set of canonical custody files for crash adoption."""
        if type(exclude) is not tuple or any(
            type(item) is not ArtifactIdentity for item in exclude
        ):
            raise TypeError("finalized-cell discovery exclude must be an artifact tuple")
        excluded: set[tuple[str, str]] = set()
        for artifact in exclude:
            digest = artifact.content_hash.digest
            expected = self.root / f"{digest}.json"
            if (
                artifact.content_hash.algorithm != "sha256"
                or artifact.path != str(expected)
                or _RECORD_NAME.fullmatch(expected.name) is None
            ):
                raise D1FinalizedCellCustodyRepositoryError(
                    code="discover_exclude",
                    path=expected,
                    detail="excluded artifact differs from content-addressed custody",
                )
            excluded.add((artifact.path, digest))
        try:
            entries = sorted(os.scandir(self.root), key=lambda item: item.name)
        except OSError as error:
            raise D1FinalizedCellCustodyRepositoryError(
                code="discover_io",
                path=self.root,
                detail=f"cannot enumerate custody root: {error}",
            ) from error
        discovered: list[tuple[D1FinalizedCellCustodyRecord, ArtifactIdentity]] = []
        for entry in entries:
            match = _RECORD_NAME.fullmatch(entry.name)
            path = self.root / entry.name
            if match is None or entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise D1FinalizedCellCustodyRepositoryError(
                    code="discover_entry",
                    path=path,
                    detail="discovery found a noncanonical custody entry",
                )
            digest = match.group(1)
            artifact = ArtifactIdentity(str(path), ContentHash("sha256", digest))
            if (artifact.path, digest) in excluded:
                continue
            discovered.append((self.load(artifact), artifact))
        return tuple(discovered)
