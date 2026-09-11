"""Bounded shell-free Git provenance for D1 matrix re-attestation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from entropic_substrate.d1_command_runner import D1BoundedProcessResult
from entropic_substrate.d1_git_provenance_adapter import (
    D1GitProvenanceAdapterError,
    SubprocessD1GitProvenance,
)

pytestmark = pytest.mark.unit

_PREREG = "a" * 40
_CODE = "c" * 40
_PATH = "docs/d1-preregistration.md"


@dataclass
class _Runner:
    responses: dict[tuple[str, ...], D1BoundedProcessResult]
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        input: bytes,
        env: object,
    ) -> D1BoundedProcessResult:
        assert cwd.is_dir()
        assert input == b""
        assert env is not None
        self.calls.append(argv)
        return self.responses[argv]


def _result(stdout: bytes = b"", *, returncode: int = 0) -> D1BoundedProcessResult:
    return D1BoundedProcessResult(returncode, stdout, b"")


def test_adapter_reads_commit_time_raw_blob_hash_and_ancestry(tmp_path: Path) -> None:
    blob = b"# preregistration\r\nexact bytes\n"
    responses = {
        ("git", "cat-file", "-e", f"{_PREREG}^{{commit}}"): _result(),
        ("git", "show", "-s", "--format=%cI", "--no-patch", _PREREG, "--"): _result(
            b"2026-08-24T12:30:45+00:00\n"
        ),
        ("git", "cat-file", "-e", f"{_PREREG}:{_PATH}"): _result(),
        ("git", "cat-file", "blob", f"{_PREREG}:{_PATH}"): _result(blob),
        ("git", "merge-base", "--is-ancestor", _PREREG, _CODE): _result(),
    }
    runner = _Runner(responses)
    adapter = SubprocessD1GitProvenance(repo_root=tmp_path, runner=runner)

    assert adapter.commit_time(_PREREG) == datetime(2026, 8, 24, 12, 30, 45, tzinfo=UTC)
    assert adapter.blob_sha256(_PREREG, _PATH) == hashlib.sha256(blob).hexdigest()
    assert adapter.read_file(_PREREG, _PATH) == blob
    assert adapter.is_ancestor(_PREREG, _CODE) is True
    assert runner.calls == [
        ("git", "cat-file", "-e", f"{_PREREG}^{{commit}}"),
        ("git", "show", "-s", "--format=%cI", "--no-patch", _PREREG, "--"),
        ("git", "cat-file", "-e", f"{_PREREG}:{_PATH}"),
        ("git", "cat-file", "blob", f"{_PREREG}:{_PATH}"),
        ("git", "cat-file", "-e", f"{_PREREG}:{_PATH}"),
        ("git", "cat-file", "blob", f"{_PREREG}:{_PATH}"),
        ("git", "merge-base", "--is-ancestor", _PREREG, _CODE),
    ]


def test_missing_commit_or_blob_and_nonancestor_are_explicit_false_values(
    tmp_path: Path,
) -> None:
    runner = _Runner(
        {
            ("git", "cat-file", "-e", f"{_PREREG}^{{commit}}"): _result(returncode=128),
            ("git", "cat-file", "-e", f"{_PREREG}:{_PATH}"): _result(returncode=128),
            ("git", "merge-base", "--is-ancestor", _PREREG, _CODE): _result(returncode=1),
        }
    )
    adapter = SubprocessD1GitProvenance(repo_root=tmp_path, runner=runner)

    assert adapter.commit_time(_PREREG) is None
    assert adapter.blob_sha256(_PREREG, _PATH) is None
    assert adapter.read_file(_PREREG, _PATH) is None
    assert adapter.is_ancestor(_PREREG, _CODE) is False


def test_tree_inventory_is_nul_delimited_and_refuses_escaped_paths(tmp_path: Path) -> None:
    command = ("git", "ls-tree", "-rz", "--name-only", _CODE, "--", "src", "configs")
    runner = _Runner({command: _result(b"configs/a.yaml\0src/pkg/a.py\0")})
    adapter = SubprocessD1GitProvenance(repo_root=tmp_path, runner=runner)
    assert adapter.tree_paths(_CODE, ("src", "configs")) == ("configs/a.yaml", "src/pkg/a.py")
    runner.responses[command] = _result(b"../outside\0")
    with pytest.raises((ValueError, D1GitProvenanceAdapterError)):
        adapter.tree_paths(_CODE, ("src", "configs"))


@pytest.mark.parametrize(
    ("call", "args"),
    [
        ("commit", ("--help",)),
        ("blob", (_PREREG, "../secret")),
        ("read", (_PREREG, "../secret")),
        ("ancestor", (_PREREG, "not-a-commit")),
    ],
)
def test_invalid_identity_refuses_before_subprocess(
    tmp_path: Path,
    call: str,
    args: tuple[str, ...],
) -> None:
    runner = _Runner({})
    adapter = SubprocessD1GitProvenance(repo_root=tmp_path, runner=runner)

    with pytest.raises((TypeError, ValueError)):
        if call == "commit":
            adapter.commit_time(*args)
        elif call == "blob":
            adapter.blob_sha256(*args)
        elif call == "read":
            adapter.read_file(*args)
        else:
            adapter.is_ancestor(*args)
    assert runner.calls == []


def test_unexpected_git_exit_is_typed_failure(tmp_path: Path) -> None:
    runner = _Runner(
        {("git", "merge-base", "--is-ancestor", _PREREG, _CODE): _result(returncode=2)}
    )

    with pytest.raises(D1GitProvenanceAdapterError, match="ancestry"):
        SubprocessD1GitProvenance(repo_root=tmp_path, runner=runner).is_ancestor(
            _PREREG,
            _CODE,
        )
