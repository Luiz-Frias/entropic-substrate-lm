"""Atomicity and CAS guards for durable D1 campaign checkpoints."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

import entropic_substrate.d1_campaign_checkpoint_repository as repository_module
from entropic_substrate.d1_campaign_checkpoint import D1CampaignDisposition
from entropic_substrate.d1_campaign_checkpoint_repository import (
    D1CampaignCheckpointRepository,
    D1CampaignCheckpointRepositoryError,
)

from .test_d1_campaign_checkpoint import _ready

pytestmark = pytest.mark.unit


def _halted():  # type: ignore[no-untyped-def]
    return replace(
        _ready(),
        disposition=D1CampaignDisposition.HARD_HALT,
        halt_reason="operator stopped the campaign",
    )


def _temps(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*.tmp"))


def test_atomic_create_and_expected_digest_load_return_exact_receipt(tmp_path: Path) -> None:
    target = tmp_path / "campaign.json"
    repository = D1CampaignCheckpointRepository(target)

    receipt = repository.save(_ready(), expected_previous_sha256=None)
    loaded, loaded_receipt = repository.load(expected_sha256=receipt.sha256)

    assert loaded == _ready()
    assert loaded_receipt == receipt
    assert receipt.path == target
    assert receipt.size_bytes == target.stat().st_size > 0
    assert receipt.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert target.read_bytes().endswith(b"\n")
    assert _temps(target) == []


def test_compare_and_swap_replaces_only_the_exact_previous_checkpoint(tmp_path: Path) -> None:
    target = tmp_path / "campaign.json"
    repository = D1CampaignCheckpointRepository(target)
    first = repository.save(_ready(), expected_previous_sha256=None)

    second = repository.save(_halted(), expected_previous_sha256=first.sha256)

    assert second.sha256 != first.sha256
    assert repository.load(expected_sha256=second.sha256)[0] == _halted()
    frozen = target.read_bytes()
    for stale in (None, first.sha256, "0" * 64):
        with pytest.raises(D1CampaignCheckpointRepositoryError, match="compare-and-swap"):
            repository.save(_ready(), expected_previous_sha256=stale)
        assert target.read_bytes() == frozen


def test_expected_previous_digest_on_missing_target_is_a_loud_stale_writer(tmp_path: Path) -> None:
    target = tmp_path / "missing.json"

    with pytest.raises(D1CampaignCheckpointRepositoryError, match="compare-and-swap"):
        D1CampaignCheckpointRepository(target).save(_ready(), expected_previous_sha256="0" * 64)

    assert not target.exists()
    assert _temps(target) == []


def test_interrupted_partial_write_preserves_prior_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "campaign.json"
    repository = D1CampaignCheckpointRepository(target)
    first = repository.save(_ready(), expected_previous_sha256=None)
    before = target.read_bytes()
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
    with pytest.raises(D1CampaignCheckpointRepositoryError, match="no progress"):
        repository.save(_halted(), expected_previous_sha256=first.sha256)

    assert calls == 2
    assert target.read_bytes() == before
    assert repository.load(expected_sha256=first.sha256)[0] == _ready()
    assert _temps(target) == []


def test_replace_failure_preserves_prior_and_removes_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "campaign.json"
    repository = D1CampaignCheckpointRepository(target)
    first = repository.save(_ready(), expected_previous_sha256=None)
    before = target.read_bytes()

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(repository_module.os, "replace", fail_replace)
    with pytest.raises(D1CampaignCheckpointRepositoryError, match="replace failure"):
        repository.save(_halted(), expected_previous_sha256=first.sha256)

    assert target.read_bytes() == before
    assert _temps(target) == []


def test_file_then_directory_are_fsynced_before_post_replace_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "campaign.json"
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    real_receipt = repository_module._receipt_from_path  # pyright: ignore[reportPrivateUsage]

    def fsync(fd: int) -> None:
        events.append("dir-fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file-fsync")
        real_fsync(fd)

    def replace_file(source: str | Path, destination: str | Path) -> None:
        events.append("replace")
        real_replace(source, destination)

    def receipt(path: Path):  # type: ignore[no-untyped-def]
        events.append("receipt")
        return real_receipt(path)

    monkeypatch.setattr(repository_module.os, "fsync", fsync)
    monkeypatch.setattr(repository_module.os, "replace", replace_file)
    monkeypatch.setattr(repository_module, "_receipt_from_path", receipt)

    D1CampaignCheckpointRepository(target).save(_ready(), expected_previous_sha256=None)
    assert events == ["file-fsync", "replace", "dir-fsync", "receipt"]


def test_load_refuses_missing_directory_symlink_corrupt_and_noncanonical(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    regular = tmp_path / "regular.json"
    regular.write_text("payload", encoding="utf-8")
    symlink = tmp_path / "link.json"
    symlink.symlink_to(regular)
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_bytes(b"not-json\n")
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(b" {}\n")

    for path in (tmp_path / "missing.json", directory, symlink, corrupt, noncanonical):
        with pytest.raises(D1CampaignCheckpointRepositoryError):
            D1CampaignCheckpointRepository(path).load()


def test_load_hash_mismatch_and_matching_hash_of_corrupt_current_both_fail(
    tmp_path: Path,
) -> None:
    target = tmp_path / "campaign.json"
    repository = D1CampaignCheckpointRepository(target)
    receipt = repository.save(_ready(), expected_previous_sha256=None)
    with pytest.raises(D1CampaignCheckpointRepositoryError, match="SHA-256"):
        repository.load(expected_sha256="0" * 64)

    target.write_bytes(b"corrupt-but-hashed\n")
    corrupt_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(D1CampaignCheckpointRepositoryError, match="corrupt"):
        repository.save(_halted(), expected_previous_sha256=corrupt_digest)
    assert target.read_bytes() == b"corrupt-but-hashed\n"
    assert receipt.sha256 != corrupt_digest
