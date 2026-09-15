"""Authoritative D1 inspection starts from one bounded no-follow journal snapshot."""

from __future__ import annotations

import hashlib
import os
import pickle
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import entropic_substrate.d1_journal_snapshot as snapshot_module
from entropic_substrate.d1_artifact_snapshot import D1ArtifactSnapshotError
from entropic_substrate.d1_journal_snapshot import (
    D1JournalSnapshot,
    read_d1_journal_snapshot,
    read_d1_local_journal_snapshot,
)
from entropic_substrate.evidence.d1_telemetry_query import (
    read_d1_run_evidence_bytes,
)
from entropic_substrate.evidence.d1_telemetry_values import D1RunStatus

from .d1_telemetry_fixtures import LOGICAL_RUN_ID, valid_events, write_events

pytestmark = pytest.mark.unit


def _journal(root: Path) -> tuple[Path, bytes]:
    target = root / "runs" / "run.jsonl"
    target.parent.mkdir(parents=True)
    write_events(target, valid_events())
    return target, target.read_bytes()


def test_bounded_snapshot_binds_same_descriptor_bytes_hash_and_run_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    target, payload = _journal(root)

    snapshot = read_d1_journal_snapshot(
        root,
        "runs/run.jsonl",
        max_bytes=len(payload),
        expected_logical_run_id=LOGICAL_RUN_ID,
    )

    assert type(snapshot) is D1JournalSnapshot
    assert snapshot.path == target
    assert snapshot.relative_path == "runs/run.jsonl"
    assert snapshot.byte_count == len(payload)
    assert snapshot.sha256 == hashlib.sha256(payload).hexdigest()
    assert snapshot.run.status is D1RunStatus.COMPLETE
    assert snapshot.run.path == target
    assert snapshot.run.identity.logical_run_id == LOGICAL_RUN_ID
    snapshot.require_factory_attested()

    target.unlink()
    from_bytes = read_d1_run_evidence_bytes(
        payload,
        journal_path=target,
        expected_logical_run_id=LOGICAL_RUN_ID,
    )
    assert from_bytes == snapshot.run


def test_descriptor_read_failure_is_typed_without_returning_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifacts"
    _journal(root)

    def fail_read(_descriptor: int, _size: int) -> bytes:
        raise OSError("simulated descriptor failure")

    monkeypatch.setattr(os, "read", fail_read)
    with pytest.raises(D1ArtifactSnapshotError, match="descriptor read") as caught:
        read_d1_journal_snapshot(root, "runs/run.jsonl", max_bytes=10_000_000)
    assert type(caught.value.__cause__) is OSError


def test_snapshot_refuses_one_byte_over_limit_without_returning_partial_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    _, payload = _journal(root)

    with pytest.raises(D1ArtifactSnapshotError, match="byte limit"):
        read_d1_journal_snapshot(
            root,
            "runs/run.jsonl",
            max_bytes=len(payload) - 1,
        )


def test_local_path_adapter_allows_arbitrary_regular_operator_visible_journal(
    tmp_path: Path,
) -> None:
    target, payload = _journal(tmp_path / "operator-visible" / "elsewhere")

    snapshot = read_d1_local_journal_snapshot(
        target,
        max_bytes=len(payload),
        expected_logical_run_id=LOGICAL_RUN_ID,
    )

    assert snapshot.path == target
    assert snapshot.relative_path == target.name
    assert snapshot.run.status is D1RunStatus.COMPLETE


def test_snapshot_capability_is_factory_attested_noncopyable_and_drift_sensitive(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    _, payload = _journal(root)
    snapshot = read_d1_journal_snapshot(
        root,
        "runs/run.jsonl",
        max_bytes=len(payload),
    )
    snapshot.require_factory_attested()

    with pytest.raises(TypeError, match="factory-only"):
        D1JournalSnapshot(snapshot.path, snapshot.relative_path, 1, "0" * 64, snapshot.run)
    for copier in (copy, deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="cannot be copied"):
            copier(snapshot)
    with pytest.raises(TypeError):
        replace(snapshot)

    object.__setattr__(snapshot, "sha256", "0" * 64)
    with pytest.raises(TypeError, match="factory-attested"):
        snapshot.require_factory_attested()


def test_snapshot_mint_derives_run_digest_and_root_relative_identity_from_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    target, payload = _journal(root)

    with pytest.raises(ValueError, match="root-relative"):
        snapshot_module._mint_snapshot(  # pyright: ignore[reportPrivateUsage]
            root,
            target,
            "different/run.jsonl",
            payload,
            expected_logical_run_id=None,
            expected_family_digest=None,
            expected_run_class=None,
            expected_plan_digest=None,
            allow_incomplete=False,
        )


@pytest.mark.parametrize(
    "field",
    ["identity", "evals", "checkpoint_receipts", "data_order_segments", "terminal"],
)
def test_nested_run_evidence_drift_invalidates_snapshot_attestation(
    tmp_path: Path,
    field: str,
) -> None:
    root = tmp_path / "artifacts"
    _, payload = _journal(root)
    snapshot = read_d1_journal_snapshot(root, "runs/run.jsonl", max_bytes=len(payload))
    if field == "identity":
        object.__setattr__(snapshot.run.identity, "logical_run_id", "drifted-run")
    elif field == "terminal":
        object.__setattr__(snapshot.run, field, None)
    else:
        object.__setattr__(snapshot.run, field, ())

    with pytest.raises(TypeError, match="factory-attested"):
        snapshot.require_factory_attested()


@pytest.mark.parametrize("kind", ["root", "directory", "file"])
def test_snapshot_refuses_symlinked_root_parent_and_file(tmp_path: Path, kind: str) -> None:
    real_root = tmp_path / "real"
    target, _ = _journal(real_root)
    root = real_root
    relative = "runs/run.jsonl"
    if kind == "root":
        root = tmp_path / "root-link"
        root.symlink_to(real_root, target_is_directory=True)
    elif kind == "directory":
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "run.jsonl").write_bytes(target.read_bytes())
        (real_root / "linked").symlink_to(outside, target_is_directory=True)
        relative = "linked/run.jsonl"
    else:
        link = target.with_name("linked.jsonl")
        link.symlink_to(target)
        relative = "runs/linked.jsonl"

    with pytest.raises(D1ArtifactSnapshotError, match=r"symlink|open"):
        read_d1_journal_snapshot(root, relative, max_bytes=10_000_000)


def test_snapshot_refuses_directory_fifo_nul_and_invalid_limit_as_typed_artifact_gaps(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "directory").mkdir()
    fifo = root / "journal.fifo"
    os.mkfifo(fifo)

    for relative in ("directory", "journal.fifo", "bad\x00name"):
        with pytest.raises(D1ArtifactSnapshotError):
            read_d1_journal_snapshot(root, relative, max_bytes=1024)
    for limit in (0, -1, True):
        with pytest.raises((TypeError, ValueError), match="max_bytes"):
            read_d1_journal_snapshot(
                root,
                "directory",
                max_bytes=limit,  # type: ignore[arg-type]
            )
