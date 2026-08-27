"""Bounded shell-free Git provenance adapter for D1 matrix admission."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import NoReturn, Protocol

from entropic_substrate.d1_artifact_snapshot import (
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
)
from entropic_substrate.d1_command_runner import (
    D1BoundedProcessError,
    D1BoundedProcessResult,
    D1BoundedProcessRunner,
)
from entropic_substrate.experiments.d1_primitives import require_commit

__all__ = ["D1GitProvenanceAdapterError", "SubprocessD1GitProvenance"]

_TIMEOUT_SECONDS = 15.0
_MAX_OUTPUT_BYTES = 20 * 1024 * 1024


class _Runner(Protocol):
    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        input: bytes,
        env: Mapping[str, str],
    ) -> D1BoundedProcessResult: ...


class D1GitProvenanceAdapterError(RuntimeError):
    """Git provenance is unavailable, malformed, or exceeded process bounds."""

    def __init__(self, *, operation: str, detail: str) -> None:
        self.operation = operation
        self.context = {"operation": operation}
        super().__init__(f"D1 Git provenance {operation}: {detail}")


def _fail(operation: str, detail: str, error: Exception | None = None) -> NoReturn:
    failure = D1GitProvenanceAdapterError(operation=operation, detail=detail)
    if error is None:
        raise failure
    raise failure from error


def _commit(value: str) -> str:
    require_commit(value, label="D1 Git provenance commit")
    return value


class SubprocessD1GitProvenance:
    """Read commit chronology, exact blob bytes, and ancestry through bounded Git."""

    def __init__(
        self,
        *,
        repo_root: Path,
        runner: _Runner | None = None,
    ) -> None:
        self._root = normalize_d1_artifact_root(repo_root)
        self._runner = runner or D1BoundedProcessRunner(
            timeout_seconds=_TIMEOUT_SECONDS,
            max_output_bytes=_MAX_OUTPUT_BYTES,
        )

    def _run(self, operation: str, argv: tuple[str, ...]) -> D1BoundedProcessResult:
        try:
            return self._runner.run(
                argv,
                cwd=self._root,
                input=b"",
                env=dict(os.environ),
            )
        except (D1BoundedProcessError, OSError, ValueError) as error:
            _fail(operation, str(error), error)

    def _exists(self, operation: str, specification: str) -> bool:
        result = self._run(operation, ("git", "cat-file", "-e", specification))
        if result.returncode == 0:
            return True
        if result.returncode == 128:
            return False
        _fail(operation, f"git cat-file exited {result.returncode}")

    def commit_time(self, commit: str) -> datetime | None:
        identity = _commit(commit)
        if not self._exists("commit_time", f"{identity}^{{commit}}"):
            return None
        result = self._run(
            "commit_time",
            ("git", "show", "-s", "--format=%cI", "--no-patch", identity, "--"),
        )
        if result.returncode != 0:
            _fail("commit_time", f"git show exited {result.returncode}")
        try:
            text = result.stdout.decode("ascii")
            if not text.endswith("\n") or "\n" in text[:-1]:
                raise ValueError("commit time output must contain one line")
            value = datetime.fromisoformat(text[:-1])
        except (UnicodeError, ValueError) as error:
            _fail("commit_time", "Git returned an invalid ISO-8601 timestamp", error)
        if value.tzinfo is None:
            _fail("commit_time", "Git commit time is timezone-naive")
        return value

    def blob_sha256(self, commit: str, path: str) -> str | None:
        identity = _commit(commit)
        relative = normalize_d1_relative_artifact_path(path, label="D1 Git blob")
        specification = f"{identity}:{relative}"
        if not self._exists("blob", specification):
            return None
        result = self._run("blob", ("git", "cat-file", "blob", specification))
        if result.returncode != 0:
            _fail("blob", f"git cat-file blob exited {result.returncode}")
        return hashlib.sha256(result.stdout).hexdigest()

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        before = _commit(ancestor)
        after = _commit(descendant)
        result = self._run(
            "ancestry",
            ("git", "merge-base", "--is-ancestor", before, after),
        )
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        _fail("ancestry", f"git merge-base exited {result.returncode}")
