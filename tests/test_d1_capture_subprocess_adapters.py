"""Hermetic command-contract tests for D1 DVC/cloud/Git adapters."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from entropic_substrate.evidence.d1_capture_ports import D1CaptureTransportError
from entropic_substrate.evidence.d1_capture_subprocess import (
    SubprocessD1CloudObjectVerifier,
    SubprocessD1DvcAdapter,
    SubprocessD1GitPointerStatusVerifier,
)

pytestmark = pytest.mark.unit

_OBJECT_ID = "8" * 32 + ".dir"


@dataclass
class _Runner:
    outputs: list[str] = field(default_factory=list)
    fail_at: int | None = None
    calls: list[tuple[list[str], Path]] = field(default_factory=list)

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        cwd = kwargs["cwd"]
        assert isinstance(cwd, Path)
        assert kwargs == {
            "cwd": cwd,
            "capture_output": True,
            "text": True,
            "check": True,
        }
        self.calls.append((argv, cwd))
        index = len(self.calls) - 1
        if self.fail_at == index:
            raise subprocess.CalledProcessError(1, argv, output="", stderr="simulated failure")
        stdout = self.outputs[index] if index < len(self.outputs) else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")


def _repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    captured = repo / "results" / "captured"
    captured.mkdir(parents=True)
    pointer = repo / "results" / "captured.dvc"
    pointer.write_text(
        f"outs:\n- md5: {_OBJECT_ID}\n  path: captured\n",
        encoding="utf-8",
    )
    return repo, captured, pointer


def test_dvc_adapter_runs_fixed_add_then_targeted_push_arguments(tmp_path: Path) -> None:
    repo, captured, pointer = _repo(tmp_path)
    runner = _Runner()
    adapter = SubprocessD1DvcAdapter(repo_root=repo, runner=runner)

    adapter.add(tracked_path=captured, pointer_path=pointer)
    adapter.push(pointer_path=pointer)

    assert runner.calls == [
        (["dvc", "add", "--", "results/captured"], repo),
        (["dvc", "push", "results/captured.dvc"], repo),
    ]


def test_dvc_adapter_wraps_command_failure_and_rejects_paths_outside_repo(
    tmp_path: Path,
) -> None:
    repo, captured, pointer = _repo(tmp_path)
    runner = _Runner(fail_at=0)
    adapter = SubprocessD1DvcAdapter(repo_root=repo, runner=runner)

    with pytest.raises(D1CaptureTransportError) as failed:
        adapter.add(tracked_path=captured, pointer_path=pointer)
    assert failed.value.code == "dvc_command"
    with pytest.raises(D1CaptureTransportError) as outside:
        adapter.add(tracked_path=tmp_path / "outside", pointer_path=pointer)
    assert outside.value.code == "command_path"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [("{}", True), ('{"results/captured.dvc":[{"changed outs":{"x":"new"}}]}', False)],
)
def test_cloud_adapter_parses_status_body_instead_of_trusting_exit_zero(
    tmp_path: Path, payload: str, expected: bool
) -> None:
    repo, _, pointer = _repo(tmp_path)
    runner = _Runner(outputs=[payload])

    present = SubprocessD1CloudObjectVerifier(repo_root=repo, runner=runner).object_is_present(
        pointer_path=pointer,
        object_id=_OBJECT_ID,
    )

    assert present is expected
    assert runner.calls == [(["dvc", "status", "--cloud", "--json", "results/captured.dvc"], repo)]


def test_cloud_adapter_refuses_malformed_status_output(tmp_path: Path) -> None:
    repo, _, pointer = _repo(tmp_path)
    runner = _Runner(outputs=["not-json"])

    with pytest.raises(D1CaptureTransportError) as captured:
        SubprocessD1CloudObjectVerifier(repo_root=repo, runner=runner).object_is_present(
            pointer_path=pointer,
            object_id=_OBJECT_ID,
        )

    assert captured.value.code == "cloud_status"


def test_git_adapter_requires_exact_clean_tracked_pointer_and_returns_head(tmp_path: Path) -> None:
    repo, _, pointer = _repo(tmp_path)
    digest = hashlib.sha256(pointer.read_bytes()).hexdigest()
    pointer_bytes = pointer.read_bytes()
    runner = _Runner(
        outputs=["", "results/captured.dvc\n", "abcdef123456\n", pointer_bytes.decode()]
    )

    evidence = SubprocessD1GitPointerStatusVerifier(
        repo_root=repo, runner=runner
    ).committed_pointer(
        pointer_path=pointer,
        pointer_bytes=pointer_bytes,
        pointer_sha256=digest,
        object_id=_OBJECT_ID,
    )

    assert evidence is not None
    assert evidence.commit == "abcdef123456"  # pragma: allowlist secret
    assert evidence.pointer_sha256 == digest
    assert [call[0] for call in runner.calls] == [
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "results/captured.dvc",
        ],
        ["git", "ls-files", "--error-unmatch", "--", "results/captured.dvc"],
        ["git", "rev-parse", "HEAD"],
        ["git", "show", "abcdef123456:results/captured.dvc"],
    ]


@pytest.mark.parametrize("condition", ["dirty", "hash_drift", "untracked"])
def test_git_adapter_returns_no_revision_for_any_uncommitted_pointer_state(
    tmp_path: Path, condition: str
) -> None:
    repo, _, pointer = _repo(tmp_path)
    digest = hashlib.sha256(pointer.read_bytes()).hexdigest()
    outputs = [" M results/captured.dvc\n"] if condition == "dirty" else ["", ""]
    runner = _Runner(outputs=outputs, fail_at=1 if condition == "untracked" else None)
    if condition == "hash_drift":
        digest = "0" * 64

    evidence = SubprocessD1GitPointerStatusVerifier(
        repo_root=repo, runner=runner
    ).committed_pointer(
        pointer_path=pointer,
        pointer_bytes=pointer.read_bytes(),
        pointer_sha256=digest,
        object_id=_OBJECT_ID,
    )

    assert evidence is None
