"""Shared shell-free subprocess runner for D1 operator adapters."""

from __future__ import annotations

import os
import subprocess  # nosec B404 -- fixed argv adapter; shell execution is never used
from pathlib import Path
from typing import cast

from entropic_substrate.d1_operator_ports import D1CommandResult

__all__ = ["SubprocessD1CommandRunner"]


def _argv(value: tuple[str, ...]) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise TypeError("D1 command runner argv must be a nonempty tuple")
    if any(type(part) is not str or not part for part in value):
        raise TypeError("D1 command runner argv entries must be nonempty strings")
    return value


def _cwd(value: Path | None) -> Path | None:
    if value is None:
        return None
    if not isinstance(cast(object, value), Path):
        raise TypeError("D1 command runner cwd must be a Path or None")
    lexical = Path(os.path.abspath(value))
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("D1 command runner cwd must be a real directory")
    try:
        root = lexical.resolve(strict=True)
    except OSError as error:
        raise ValueError("D1 command runner cwd must be a real directory") from error
    if not root.is_dir():
        raise ValueError("D1 command runner cwd must be a real directory")
    return root


def _stdin(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError("D1 command runner stdin must be a string or None")
    return value


class SubprocessD1CommandRunner:
    """Execute an explicit argv tuple and capture UTF-8 text streams."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        stdin: str | None,
    ) -> D1CommandResult:
        result = subprocess.run(  # nosec B603 -- argv is explicit and shell-free
            _argv(argv),
            cwd=_cwd(cwd),
            input=_stdin(stdin),
            capture_output=True,
            text=True,
            check=True,
        )
        return D1CommandResult(stdout=result.stdout, stderr=result.stderr)
