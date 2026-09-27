"""Content-addressed finalized-cell custody is atomic and hash-verified."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

import entropic_substrate.ladder.capture.custody.d1_content_addressed_custody as repository_module
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.closure.d1_finalized_cell_custody_repository import (
    D1FinalizedCellCustodyRepository,
    D1FinalizedCellCustodyRepositoryError,
)

from .test_d1_finalized_cell_custody_values import _attest, _finalized_case

pytestmark = pytest.mark.unit


def _temps(root: Path) -> list[Path]:
    return list(root.glob(".*.tmp"))


def test_repository_atomically_saves_loads_and_reuses_exact_content(tmp_path: Path) -> None:
    record = _attest(_finalized_case(tmp_path / "case"))
    root = tmp_path / "custody"
    root.mkdir()
    repository = D1FinalizedCellCustodyRepository(root)

    first = repository.save(record)
    second = repository.save(record)
    restored = repository.load(first)

    assert first == second
    assert first.path == str(root / f"{record.record_digest}.json")
    assert first.content_hash == ContentHash("sha256", record.record_digest)
    assert hashlib.sha256(Path(first.path).read_bytes()).hexdigest() == record.record_digest
    assert restored.manifest == record.manifest
    restored.require_factory_attested()
    assert _temps(root) == []


def test_partial_write_preserves_prior_records_and_publishes_no_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    repository = D1FinalizedCellCustodyRepository(root)
    prior = repository.save(_attest(_finalized_case(tmp_path / "prior")))
    candidate = _attest(_finalized_case(tmp_path / "candidate"))
    target = root / f"{candidate.record_digest}.json"
    real_write = os.write
    calls = 0

    def interrupted(fd: int, data: bytes | bytearray | memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            payload = bytes(data)
            return real_write(fd, payload[: max(1, len(payload) // 2)])
        return 0

    monkeypatch.setattr(repository_module.os, "write", interrupted)
    with pytest.raises(D1FinalizedCellCustodyRepositoryError, match="partial write"):
        repository.save(candidate)

    assert Path(prior.path).is_file()
    assert not target.exists()
    assert _temps(root) == []


def test_file_then_directory_are_fsynced_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    record = _attest(_finalized_case(tmp_path / "case"))
    events: list[str] = []
    real_fsync, real_link = os.fsync, os.link

    def fsync(fd: int) -> None:
        events.append("dir-fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file-fsync")
        real_fsync(fd)

    def link(source: str | Path, target: str | Path, **kwargs: object) -> None:
        events.append("publish")
        real_link(source, target, **kwargs)

    monkeypatch.setattr(repository_module.os, "fsync", fsync)
    monkeypatch.setattr(repository_module.os, "link", link)
    D1FinalizedCellCustodyRepository(root).save(record)

    assert events == ["file-fsync", "publish", "dir-fsync"]


def test_raced_existing_publication_is_file_and_directory_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    record = _attest(_finalized_case(tmp_path / "case"))
    events: list[str] = []
    real_fsync, real_link = os.fsync, os.link

    def fsync(fd: int) -> None:
        events.append("dir-fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file-fsync")
        real_fsync(fd)

    def raced_link(source: str | Path, target: str | Path, **kwargs: object) -> None:
        events.append("publish-race")
        real_link(source, target, **kwargs)
        raise FileExistsError("simulated competing publication")

    monkeypatch.setattr(repository_module.os, "fsync", fsync)
    monkeypatch.setattr(repository_module.os, "link", raced_link)
    artifact = D1FinalizedCellCustodyRepository(root).save(record)

    assert Path(artifact.path).is_file()
    assert events == ["file-fsync", "publish-race", "file-fsync", "dir-fsync"]
    assert _temps(root) == []


def test_repository_refuses_symlinks_collisions_and_substituted_receipts(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(D1FinalizedCellCustodyRepositoryError, match="root"):
        D1FinalizedCellCustodyRepository(linked_root)

    record = _attest(_finalized_case(tmp_path / "case"))
    repository = D1FinalizedCellCustodyRepository(real_root)
    target = real_root / f"{record.record_digest}.json"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("substituted", encoding="utf-8")
    target.symlink_to(elsewhere)
    with pytest.raises(D1FinalizedCellCustodyRepositoryError, match="symlink"):
        repository.save(record)
    target.unlink()

    artifact = repository.save(record)
    wrong_hash = replace(artifact, content_hash=ContentHash("sha256", "0" * 64))
    wrong_path = ArtifactIdentity(str(elsewhere), artifact.content_hash)
    for candidate in (wrong_hash, wrong_path):
        with pytest.raises(D1FinalizedCellCustodyRepositoryError):
            repository.load(candidate)


def test_existing_wrong_bytes_at_content_address_are_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    record = _attest(_finalized_case(tmp_path / "case"))
    target = root / f"{record.record_digest}.json"
    target.write_bytes(b"wrong bytes")
    before = target.read_bytes()

    with pytest.raises(D1FinalizedCellCustodyRepositoryError, match="collision"):
        D1FinalizedCellCustodyRepository(root).save(record)

    assert target.read_bytes() == before
    assert _temps(root) == []


def test_discovery_returns_canonical_records_and_supports_exact_exclusion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    repository = D1FinalizedCellCustodyRepository(root)
    first = repository.save(_attest(_finalized_case(tmp_path / "first")))
    second = repository.save(_attest(_finalized_case(tmp_path / "second")))

    discovered = repository.discover()
    remaining = repository.discover(exclude=(first,))

    assert tuple(artifact for _record, artifact in discovered) == tuple(
        sorted((first, second), key=lambda item: item.path)
    )
    assert tuple(artifact for _record, artifact in remaining) == (second,)
    assert all(
        record.record_digest == artifact.content_hash.digest for record, artifact in discovered
    )


@pytest.mark.parametrize("hazard", ["stray", "symlink"])
def test_discovery_refuses_noncanonical_or_symlink_entries(
    tmp_path: Path,
    hazard: str,
) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    repository = D1FinalizedCellCustodyRepository(root)
    repository.save(_attest(_finalized_case(tmp_path / "case")))
    target = root / "stray.txt"
    if hazard == "stray":
        target.write_text("unexpected", encoding="utf-8")
    else:
        source = tmp_path / "source"
        source.write_text("unexpected", encoding="utf-8")
        target.symlink_to(source)

    with pytest.raises(D1FinalizedCellCustodyRepositoryError, match="discover"):
        repository.discover()


def test_discovery_refuses_substituted_exclusion_identity(tmp_path: Path) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    repository = D1FinalizedCellCustodyRepository(root)
    artifact = repository.save(_attest(_finalized_case(tmp_path / "case")))
    substituted = replace(artifact, content_hash=ContentHash("sha256", "0" * 64))

    with pytest.raises(D1FinalizedCellCustodyRepositoryError, match="exclude"):
        repository.discover(exclude=(substituted,))
