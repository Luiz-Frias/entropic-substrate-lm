"""The repository root, found by marker instead of by the caller's depth.

Test modules live at several depths under ``tests/``. Deriving the root as
``Path(__file__).resolve().parents[N]`` encodes each file's depth in the file,
so a move re-points it without any error. ``REPO_ROOT`` walks up from this
module to the first directory holding ``pyproject.toml``.
"""

from __future__ import annotations

from pathlib import Path

MARKER = "pyproject.toml"


def find_root(start: Path) -> Path:
    """First directory at or above ``start`` that holds the project marker."""
    for candidate in (start, *start.parents):
        if (candidate / MARKER).is_file():
            return candidate
    message = f"no {MARKER} at or above {start}"
    raise RuntimeError(message)


REPO_ROOT = find_root(Path(__file__).resolve().parent)
TESTS_ROOT = REPO_ROOT / "tests"
