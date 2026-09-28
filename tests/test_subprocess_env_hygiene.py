"""No test may hand its subprocess to ``uv`` (FOR-1618, measured 2026-09-28).

A bare ``uv run`` inside a test re-syncs the project ``.venv`` from
``uv.lock`` while the suite is using it. Under pytest-xdist with two workers
that is a race: ``tests/test_cold_import.py`` children died mid-import with
``ModuleNotFoundError: No module named 'numpy'`` (runs 36489147202,
36489150096) and ``No module named 'torch._opaque_base'`` (run 36489145501)
while ``test_notebooks.py`` was re-installing those very packages on the
other worker. On one worker the same call merely cost 26.57 s / 55.72 s
(runs 36486275734 / 36486548073) against 1.4 s for ``python -m marimo``.

Tests reach tools through ``sys.executable`` so the interpreter that runs
the suite is the interpreter that runs the tool, and nothing rewrites the
environment underneath the other worker.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

TESTS_DIR = Path(__file__).resolve().parent

# A subprocess argv list whose first element is uv: ["uv", ...] / ['uv', ...].
_UV_ARGV0 = re.compile(r"""\[\s*["']uv["']\s*,""")


def _uv_argv_sites() -> list[str]:
    sites: list[str] = []
    for path in sorted(TESTS_DIR.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _UV_ARGV0.search(line):
                sites.append(f"{path.name}:{lineno}: {line.strip()}")
    return sites


def test_no_test_invokes_uv_as_a_subprocess() -> None:
    sites = _uv_argv_sites()
    assert sites == [], (
        "tests must not shell out to uv (it re-syncs the shared .venv):\n" + "\n".join(sites)
    )


def test_guard_pattern_matches_the_shape_it_guards() -> None:
    # Teeth: the regex must catch both quote styles and survive whitespace.
    assert _UV_ARGV0.search('subprocess.run(["uv", "run", "marimo"])')
    assert _UV_ARGV0.search("subprocess.run([ 'uv' , 'sync'])")
    # and must not fire on strings that merely mention uv, or on sets
    assert not _UV_ARGV0.search('assert "uv run deptry" in text')
    assert not _UV_ARGV0.search('{"uv", "github-actions"}')
    assert not _UV_ARGV0.search('[sys.executable, "-m", "marimo"]')
