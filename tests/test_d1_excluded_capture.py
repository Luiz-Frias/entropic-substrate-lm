"""Excluded pilot capture proves bytes/cloud/Git without a confirmation roster."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from entropic_substrate.ladder.capture.custody.d1_capture_ports import D1GitPointerCommitEvidence
from entropic_substrate.ladder.run.closure.d1_excluded_capture import D1ExcludedCaptureService

from .test_d1_excluded_operator import pilot_case

pytestmark = pytest.mark.unit


@dataclass
class CapturePorts:
    root: Path
    committed: bool = True
    present: bool = True

    def add(self, *, tracked_path: Path, pointer_path: Path) -> None:
        pointer_path.write_text(f"outs:\n- md5: {'8' * 32}.dir\n  path: {tracked_path.name}\n")

    def push(self, *, pointer_path: Path) -> None:
        assert pointer_path.is_file()

    def object_is_present(self, *, pointer_path: Path, object_id: str) -> bool:
        return self.present

    def committed_pointer(
        self, *, pointer_path: Path, pointer_bytes: bytes, pointer_sha256: str, object_id: str
    ):
        assert hashlib.sha256(pointer_bytes).hexdigest() == pointer_sha256
        if not self.committed:
            return None
        return D1GitPointerCommitEvidence(
            "a" * 40, str(pointer_path.relative_to(self.root)), pointer_sha256, object_id
        )


def captured_pilot(tmp_path: Path):
    operator, intent, _, _, _ = pilot_case(tmp_path)
    result = operator.execute_fresh(intent)
    ports = CapturePorts(tmp_path)
    capture = D1ExcludedCaptureService(
        repo_root=tmp_path, source_root=tmp_path, dvc=ports, cloud=ports, git=ports
    )
    preparation = capture.prepare(result)
    return capture, preparation, ports


def test_real_pilot_capture_retains_exclusion_and_hash_verified_journal(tmp_path: Path) -> None:
    service, preparation, _ports = captured_pilot(tmp_path)
    receipt = service.finalize(preparation)
    assert receipt.plan.run.run_class.value == "pilot"
    assert receipt.plan.run.preregistration is None
    assert receipt.journal_sha256 == hashlib.sha256(receipt.read_journal_bytes()).hexdigest()
    assert receipt.pointer_commit == "a" * 40
    Path(receipt.journal.path).write_bytes(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        receipt.read_journal_bytes()


def test_uncommitted_pointer_cannot_mint_receipt(tmp_path: Path) -> None:
    service, preparation, ports = captured_pilot(tmp_path)
    ports.committed = False
    with pytest.raises(ValueError, match="committed"):
        service.finalize(preparation)


def test_fresh_session_rehydrates_only_exact_installed_pilot_construction(tmp_path: Path) -> None:
    service, preparation, _ports = captured_pilot(tmp_path)
    expected = service.finalize(preparation)
    reloaded = service.load(preparation.tracked_path, catalog=(expected.plan.profile,))
    assert reloaded.capture_digest == expected.capture_digest
    assert reloaded.read_journal_bytes() == expected.read_journal_bytes()
    with pytest.raises(ValueError, match="installed"):
        service.load(preparation.tracked_path, catalog=())


def test_capture_refuses_symlink_parent_before_export(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    operator, intent, _, _, _ = pilot_case(source)
    result = operator.execute_fresh(intent)
    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (repo / "results").symlink_to(external, target_is_directory=True)
    ports = CapturePorts(repo)
    capture = D1ExcludedCaptureService(
        repo_root=repo, source_root=source, dvc=ports, cloud=ports, git=ports
    )
    with pytest.raises(ValueError, match="symlink"):
        capture.prepare(result)
    assert list(external.iterdir()) == []
