"""DVC, cloud-status, and Git adapters for the D1 capture ports."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess  # nosec B404 -- fixed argv adapters; shell execution is never used
from pathlib import Path
from typing import Protocol, cast

from entropic_substrate.evidence.d1_capture_ports import D1CaptureTransportError

__all__ = [
    "SubprocessD1CloudObjectVerifier",
    "SubprocessD1DvcAdapter",
    "SubprocessD1GitPointerStatusVerifier",
]


class _CommandResult(Protocol):
    stdout: str
    stderr: str


class _CommandRunner(Protocol):
    def __call__(
        self,
        argv: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> _CommandResult: ...


def _error(code: str, path: str, detail: str) -> D1CaptureTransportError:
    return D1CaptureTransportError(code=code, path=path, detail=detail)


def _root(value: str | os.PathLike[str]) -> Path:
    root = Path(os.path.abspath(value))
    if root.is_symlink() or not root.is_dir():
        raise _error("repo_root", str(root), "must be an existing real directory")
    return root


def _relative(root: Path, value: Path) -> str:
    lexical = Path(os.path.abspath(value))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise _error("command_path", str(lexical), "is outside the repository") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise _error("command_path", relative.as_posix(), "contains a symlink component")
    return relative.as_posix()


def _runner(value: _CommandRunner | None) -> _CommandRunner:
    return value if value is not None else cast(_CommandRunner, subprocess.run)


def _invoke(
    runner: _CommandRunner,
    argv: list[str],
    *,
    root: Path,
    code: str,
    path: str,
) -> _CommandResult:
    try:
        return runner(
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as error:
        stderr = getattr(error, "stderr", None)
        detail = str(stderr).strip() if stderr else str(error)
        raise _error(code, path, detail) from error


class SubprocessD1DvcAdapter:
    """Run repository-scoped DVC add and targeted push commands."""

    def __init__(
        self,
        *,
        repo_root: str | os.PathLike[str],
        runner: _CommandRunner | None = None,
    ) -> None:
        self._root = _root(repo_root)
        self._runner = _runner(runner)

    def add(self, *, tracked_path: Path, pointer_path: Path) -> None:
        tracked = _relative(self._root, tracked_path)
        pointer = _relative(self._root, pointer_path)
        if pointer != f"{tracked}.dvc":
            raise _error("command_path", pointer, "does not address the tracked DVC path")
        _invoke(
            self._runner,
            ["dvc", "add", "--", tracked],
            root=self._root,
            code="dvc_command",
            path=tracked,
        )

    def push(self, *, pointer_path: Path) -> None:
        pointer = _relative(self._root, pointer_path)
        _invoke(
            self._runner,
            ["dvc", "push", pointer],
            root=self._root,
            code="dvc_command",
            path=pointer,
        )


def _object_id(value: str) -> None:
    digest = value.removesuffix(".dir")
    if (
        len(digest) != 32
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("DVC object_id must be 32 lowercase hexadecimal characters")


class SubprocessD1CloudObjectVerifier:
    """Parse DVC's cloud-status body; exit zero alone is not affirmative proof."""

    def __init__(
        self,
        *,
        repo_root: str | os.PathLike[str],
        runner: _CommandRunner | None = None,
    ) -> None:
        self._root = _root(repo_root)
        self._runner = _runner(runner)

    def object_is_present(self, *, pointer_path: Path, object_id: str) -> bool:
        pointer = _relative(self._root, pointer_path)
        try:
            _object_id(object_id)
        except ValueError as error:
            raise _error("cloud_status", pointer, str(error)) from error
        result = _invoke(
            self._runner,
            ["dvc", "status", "--cloud", "--json", pointer],
            root=self._root,
            code="cloud_status",
            path=pointer,
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise _error("cloud_status", pointer, "DVC cloud status was not JSON") from error
        if not isinstance(payload, dict):
            raise _error("cloud_status", pointer, "DVC cloud status must be a JSON object")
        return not payload


def _pointer_hash(path: Path) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SubprocessD1GitPointerStatusVerifier:
    """Prove exact pointer bytes are clean, tracked, and contained in HEAD."""

    def __init__(
        self,
        *,
        repo_root: str | os.PathLike[str],
        runner: _CommandRunner | None = None,
    ) -> None:
        self._root = _root(repo_root)
        self._runner = _runner(runner)

    def committed_revision(self, *, pointer_path: Path, pointer_sha256: str) -> str | None:
        pointer = _relative(self._root, pointer_path)
        if _pointer_hash(pointer_path) != pointer_sha256:
            return None
        status = _invoke(
            self._runner,
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                pointer,
            ],
            root=self._root,
            code="git_status",
            path=pointer,
        )
        if status.stdout:
            return None
        try:
            tracked = self._runner(
                ["git", "ls-files", "--error-unmatch", "--", pointer],
                cwd=self._root,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            return None
        except Exception as error:
            raise _error("git_status", pointer, str(error)) from error
        if tracked.stdout.strip() != pointer:
            return None
        head = _invoke(
            self._runner,
            ["git", "rev-parse", "HEAD"],
            root=self._root,
            code="git_status",
            path=pointer,
        ).stdout.strip()
        if not 7 <= len(head) <= 64 or any(
            character not in "0123456789abcdef" for character in head
        ):
            raise _error("git_status", pointer, "HEAD was not a lowercase hexadecimal commit")
        return head
