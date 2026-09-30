"""Descriptor-bound, byte-limited snapshots for authoritative D1 journals."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn, SupportsIndex, cast

from entropic_substrate.kernel.artifacts import (
    normalize_d1_relative_artifact_path,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.identity.d1_primitives import RunClass, require_sha256_hex_any_case
from entropic_substrate.kernel.identity.factory_attestation import FactoryAttestation
from entropic_substrate.ladder.capture.records.d1_telemetry_query import (
    read_d1_run_evidence_bytes,
)
from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1RunEvidence

__all__ = [
    "D1JournalSnapshot",
    "read_d1_journal_snapshot",
    "read_d1_local_journal_snapshot",
]


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class D1JournalSnapshot:
    """One exact regular-file payload and the run evidence folded from it."""

    path: Path
    relative_path: str
    byte_count: int
    sha256: str
    run: D1RunEvidence

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("D1JournalSnapshot is factory-only")

    def __copy__(self) -> NoReturn:
        raise TypeError("D1 journal snapshot cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("D1 journal snapshot cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("D1 journal snapshot cannot be copied")

    def require_factory_attested(self) -> None:
        if not _SNAPSHOT_ATTESTATION.contains(self):
            raise TypeError("D1 journal snapshot is not factory-attested")


def _validate_snapshot(value: D1JournalSnapshot) -> None:
    if type(value) is not D1JournalSnapshot:
        raise TypeError("D1 journal snapshot must be nominal")
    if not isinstance(cast(object, value.path), Path):
        raise TypeError("D1 journal snapshot path must be a Path")
    normalize_d1_relative_artifact_path(
        value.relative_path,
        label="D1 journal snapshot",
    )
    if type(value.byte_count) is not int or value.byte_count < 1:
        raise ValueError("D1 journal snapshot byte_count must be positive")
    require_sha256_hex_any_case(value.sha256, label="D1 journal snapshot sha256")
    if type(value.run) is not D1RunEvidence or value.run.path != value.path:
        raise ValueError("D1 journal snapshot run evidence differs from its path")


def _snapshot_fingerprint(value: D1JournalSnapshot) -> tuple[object, ...]:
    return (
        value.path,
        value.relative_path,
        value.byte_count,
        value.sha256,
        value.run.authority_digest,
    )


_SNAPSHOT_ATTESTATION = FactoryAttestation(_snapshot_fingerprint)


def _mint_snapshot(
    root: Path,
    path: Path,
    relative_path: str,
    payload: bytes,
    *,
    expected_logical_run_id: str | None,
    expected_family_digest: str | None,
    expected_run_class: RunClass | None,
    expected_plan_digest: str | None,
    allow_incomplete: bool,
) -> D1JournalSnapshot:
    relative = normalize_d1_relative_artifact_path(
        relative_path,
        label="D1 journal snapshot",
    )
    expected_path = Path(os.path.abspath(root)).joinpath(*PurePosixPath(relative).parts)
    if path != expected_path:
        raise ValueError("D1 journal snapshot path differs from root-relative identity")
    digest = hashlib.sha256(payload).hexdigest()
    run = read_d1_run_evidence_bytes(
        payload,
        journal_path=path,
        expected_logical_run_id=expected_logical_run_id,
        expected_family_digest=expected_family_digest,
        expected_run_class=expected_run_class,
        expected_plan_digest=expected_plan_digest,
        allow_incomplete=allow_incomplete,
    )
    value = object.__new__(D1JournalSnapshot)
    object.__setattr__(value, "path", path)
    object.__setattr__(value, "relative_path", relative)
    object.__setattr__(value, "byte_count", len(payload))
    object.__setattr__(value, "sha256", digest)
    object.__setattr__(value, "run", run)
    _validate_snapshot(value)
    _SNAPSHOT_ATTESTATION.register(value)
    return value


def read_d1_journal_snapshot(
    root: Path,
    relative_path: object,
    *,
    max_bytes: int,
    expected_logical_run_id: str | None = None,
    expected_family_digest: str | None = None,
    expected_run_class: RunClass | None = None,
    expected_plan_digest: str | None = None,
    allow_incomplete: bool = False,
) -> D1JournalSnapshot:
    """Read one contained path and fold only the bytes held by that descriptor."""
    relative = normalize_d1_relative_artifact_path(
        relative_path,
        label="D1 journal snapshot",
    )
    path, payload, _ = read_d1_artifact_bytes(
        root,
        relative,
        label="D1 journal snapshot",
        max_bytes=max_bytes,
    )
    return _mint_snapshot(
        root,
        path,
        relative,
        payload,
        expected_logical_run_id=expected_logical_run_id,
        expected_family_digest=expected_family_digest,
        expected_run_class=expected_run_class,
        expected_plan_digest=expected_plan_digest,
        allow_incomplete=allow_incomplete,
    )


def read_d1_local_journal_snapshot(
    path: Path,
    *,
    max_bytes: int,
    expected_logical_run_id: str | None = None,
    expected_family_digest: str | None = None,
    expected_run_class: RunClass | None = None,
    expected_plan_digest: str | None = None,
    allow_incomplete: bool = False,
) -> D1JournalSnapshot:
    """Inspect any operator-visible local regular file through the same safe adapter."""
    if not isinstance(cast(object, path), Path):
        raise TypeError("D1 local journal path must be a Path")
    lexical = Path(os.path.abspath(path))
    return read_d1_journal_snapshot(
        lexical.parent,
        lexical.name,
        max_bytes=max_bytes,
        expected_logical_run_id=expected_logical_run_id,
        expected_family_digest=expected_family_digest,
        expected_run_class=expected_run_class,
        expected_plan_digest=expected_plan_digest,
        allow_incomplete=allow_incomplete,
    )
