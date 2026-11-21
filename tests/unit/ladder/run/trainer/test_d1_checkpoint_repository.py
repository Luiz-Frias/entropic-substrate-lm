"""Durability and admission guards for the atomic D1 checkpoint repository."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Protocol, cast

import pytest

import entropic_substrate.ladder.run.trainer.checkpoint_repository as repository
from d1_checkpoint_fixtures import CheckpointCase, checkpoint_state, set_path
from entropic_substrate.ladder.run.trainer.checkpoint_envelope import D1CheckpointError
from entropic_substrate.ladder.run.trainer.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
    D1CheckpointRepositoryError,
    load_and_validate,
    load_checkpoint_candidate,
    save_atomic,
)

pytestmark = pytest.mark.unit


class _Writable(Protocol):
    def write(self, data: bytes) -> int: ...


def _load(
    path: Path,
    case: CheckpointCase,
    *,
    for_resume: bool = False,
    expected_sha256: str | None = None,
):
    return load_and_validate(
        path,
        system=case.system,
        optimizer=case.optimizer,
        sampler=case.sampler,
        run=case.run,
        protocol=case.protocol,
        for_resume=for_resume,
        expected_sha256=expected_sha256,
    )


def _temporary_siblings(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.*.tmp"))


def test_atomic_save_returns_post_replace_receipt_and_validated_load(tmp_path: Path) -> None:
    state, case = checkpoint_state()
    target = tmp_path / "d1.pt"

    receipt = save_atomic(target, state)
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    loaded, loaded_receipt = _load(target, case, for_resume=True, expected_sha256=expected)

    assert receipt.path == target
    assert receipt.size_bytes == target.stat().st_size > 0
    assert receipt.sha256 == expected == receipt.sha256.lower()
    assert loaded["successful_step"] == state["successful_step"]
    assert loaded_receipt == receipt
    assert _temporary_siblings(target) == []


def test_candidate_load_verifies_receipt_before_runtime_materialization(tmp_path: Path) -> None:
    state, _case = checkpoint_state()
    target = tmp_path / "candidate.pt"
    receipt = save_atomic(target, state)

    candidate, loaded_receipt = load_checkpoint_candidate(
        target,
        expected_sha256=receipt.sha256,
    )

    assert candidate["successful_step"] == state["successful_step"]
    assert loaded_receipt == receipt
    with pytest.raises(D1CheckpointRepositoryError, match="SHA-256 mismatch"):
        load_checkpoint_candidate(target, expected_sha256="0" * 64)


def test_short_os_writes_are_completed_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, case = checkpoint_state()
    target = tmp_path / "short-write.pt"
    real_write = os.write
    partial_writes = 0

    def partial(fd: int, data: bytes | bytearray | memoryview) -> int:
        nonlocal partial_writes
        payload = bytes(data)
        if len(payload) > 1:
            partial_writes += 1
            payload = payload[: max(1, len(payload) // 2)]
        return real_write(fd, payload)

    monkeypatch.setattr(repository.os, "write", partial)
    save_atomic(target, state)

    assert partial_writes > 0
    loaded, _receipt = _load(target, case, for_resume=True)
    assert loaded["successful_step"] == 0


def test_serialization_failure_preserves_target_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, _case = checkpoint_state()
    target = tmp_path / "existing.pt"
    target.write_bytes(b"previous-good-artifact")

    def fail(_state: object, handle: _Writable) -> None:
        handle.write(b"partial")
        raise RuntimeError("simulated torch.save failure")

    monkeypatch.setattr(repository.torch, "save", fail)
    with pytest.raises(D1CheckpointRepositoryError, match=r"torch\.save failure"):
        save_atomic(target, state)

    assert target.read_bytes() == b"previous-good-artifact"
    assert _temporary_siblings(target) == []


def test_replace_failure_preserves_target_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, _case = checkpoint_state()
    target = tmp_path / "replace.pt"
    target.write_bytes(b"old")
    monkeypatch.setattr(repository.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("no")))

    with pytest.raises(D1CheckpointRepositoryError, match="atomic checkpoint save"):
        save_atomic(target, state)
    assert target.read_bytes() == b"old"
    assert _temporary_siblings(target) == []


def test_file_then_parent_are_fsynced_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, _case = checkpoint_state()
    target = tmp_path / "ordered.pt"
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    real_receipt = repository._receipt_from_path

    def fsync(fd: int) -> None:
        events.append("dir-fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file-fsync")
        real_fsync(fd)

    def replace(source: str | Path, destination: str | Path) -> None:
        events.append("replace")
        real_replace(source, destination)

    def receipt(path: Path) -> D1CheckpointArtifactReceipt:
        events.append("receipt")
        return real_receipt(path)

    monkeypatch.setattr(repository.os, "fsync", fsync)
    monkeypatch.setattr(repository.os, "replace", replace)
    monkeypatch.setattr(repository, "_receipt_from_path", receipt)

    save_atomic(target, state)
    assert events == ["file-fsync", "replace", "dir-fsync", "receipt"]


def test_file_fsync_failure_preserves_target_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, _case = checkpoint_state()
    target = tmp_path / "fsync.pt"
    target.write_bytes(b"old")
    monkeypatch.setattr(repository.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("disk")))

    with pytest.raises(D1CheckpointRepositoryError, match="atomic checkpoint save"):
        save_atomic(target, state)
    assert target.read_bytes() == b"old"
    assert _temporary_siblings(target) == []


def test_load_rejects_missing_nonfile_and_symlink(tmp_path: Path) -> None:
    _state, case = checkpoint_state()
    directory = tmp_path / "directory"
    directory.mkdir()
    regular = tmp_path / "regular.pt"
    regular.write_bytes(b"payload")
    symlink = tmp_path / "link.pt"
    symlink.symlink_to(regular)

    for path in (tmp_path / "missing.pt", directory, symlink):
        with pytest.raises(D1CheckpointRepositoryError):
            _load(path, case)


def test_load_rejects_expected_hash_mismatch_and_corruption(tmp_path: Path) -> None:
    state, case = checkpoint_state()
    valid = tmp_path / "valid.pt"
    save_atomic(valid, state)
    with pytest.raises(D1CheckpointRepositoryError, match="SHA-256 mismatch"):
        _load(valid, case, expected_sha256="0" * 64)

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(D1CheckpointRepositoryError, match="cannot load"):
        _load(corrupt, case)


def test_terminal_checkpoint_loads_for_archive_but_not_resume(tmp_path: Path) -> None:
    state, case = checkpoint_state()
    set_path(state, ("lifecycle", "state"), "completed")
    set_path(state, ("lifecycle", "completion_reason"), "terminal eval filed")
    target = tmp_path / "completed.pt"
    save_atomic(target, state)

    archived, _receipt = _load(target, case, for_resume=False)
    assert cast(dict[str, object], archived["lifecycle"])["state"] == "completed"
    with pytest.raises(D1CheckpointError, match="not resumable"):
        _load(target, case, for_resume=True)
