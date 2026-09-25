"""ACCEPTANCE instrument for the p12-historical-composition lane (brief
ACCEPTANCE: "Cold import, fresh interpreter per moved leaf, entered FIRST,
stdout marker not exit code" -- mirrors ``tests/test_run_upper_cold_import.py``).

RED-first (house rule 5): before commit 1 lands, p12's own fragment has no
``landed: true`` rows, so ``_fragment_landed_targets()`` is empty and
``test_fragment_has_at_least_one_landed_target`` fails loud rather than
vacuously passing on an empty parametrize set.

SUBJECT-SET DERIVATION, generic across p12's multiple destination trees
(unlike p10-run-upper's single ``ladder/run/{campaign,closure,ckptsync}``
sweep): p12 lands into ``historical/{d_ladder,e_series}``,
``composition/{wiring,entrypoints}`` and ``ladder/gate`` across its six
commits, so this file derives its subject set directly from p12's own
fragment's ``modules`` map (``landed: true`` targets) rather than a
hand-maintained per-subpackage glob -- the fragment IS the authoritative,
live-growing list of "what p12 has landed so far", extended automatically
as each commit adds its own rows, with no further edit to this file needed
per commit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_PATH = (
    REPO_ROOT / ".claude" / "workflows" / "landed-set" / "p12-historical-composition.json"
)

pytestmark = pytest.mark.unit

_MARKER = "COLD_IMPORT_OK"

_COLD_IMPORT_TEMPLATE = """
import importlib
importlib.import_module({dotted!r})
print({marker!r})
"""


def _fragment_landed_targets() -> list[str]:
    """p12's own fragment's ``landed: true`` target rows -- the live,
    per-commit-growing subject set."""
    if not FRAGMENT_PATH.exists():
        return []
    data = json.loads(FRAGMENT_PATH.read_text())
    return sorted(
        entry["target"]
        for entry in (data.get("modules") or {}).values()
        if entry.get("landed", False)
    )


def _run_cold_import(dotted: str) -> subprocess.CompletedProcess[str]:
    code = _COLD_IMPORT_TEMPLATE.format(dotted=dotted, marker=_MARKER)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )


def _assert_cold_import_ok(result: subprocess.CompletedProcess[str], dotted: str) -> None:
    """Positive-arm assertion: the cold-import subprocess must exit 0 AND
    print the marker -- not merely exit 0."""
    assert result.returncode == 0, f"cold import of {dotted} failed:\n{result.stderr}"
    assert _MARKER in result.stdout, (
        f"child process for {dotted} did not report success; "
        f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
    )


def test_fragment_has_at_least_one_landed_target() -> None:
    """RED-first: before commit 1 lands, this is empty and this test fails
    loud rather than the parametrize below silently collecting zero cases."""
    targets = _fragment_landed_targets()
    assert targets, (
        "p12 fragment has no landed:true rows yet -- RED-first: expected "
        "before commit 1 lands, must go GREEN once it does"
    )


@pytest.mark.parametrize("dotted", _fragment_landed_targets())
def test_cold_import_of_landed_p12_target_succeeds_with_nothing_warming_it(
    dotted: str,
) -> None:
    """Fresh interpreter, NOTHING imported first: importing ``dotted``
    directly must succeed and print the marker -- not merely exit 0."""
    _assert_cold_import_ok(_run_cold_import(dotted), dotted)


def test_cold_import_guard_accepts_a_real_import() -> None:
    """Positive arm: the instrument reports the marker for an importable module."""
    result = _run_cold_import("json")
    assert result.returncode == 0
    assert _MARKER in result.stdout


def test_cold_import_guard_rejects_a_failed_import() -> None:
    """Negative arm: a failed import never prints the marker."""
    result = _run_cold_import("entropic_substrate.__no_such_module__")
    assert result.returncode != 0
    assert _MARKER not in result.stdout
    with pytest.raises(AssertionError):
        _assert_cold_import_ok(result, "entropic_substrate.__no_such_module__")


def test_cold_import_guard_is_not_fooled_by_a_process_that_silently_exits_0() -> None:
    """GREEN control for the instrument itself: a zero-exit, markerless
    child must still make the helper raise."""
    result = subprocess.run(
        [sys.executable, "-c", "pass"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert _MARKER not in result.stdout
    with pytest.raises(AssertionError):
        _assert_cold_import_ok(result, "silent zero-exit child")
