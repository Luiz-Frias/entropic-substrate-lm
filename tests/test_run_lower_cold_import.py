"""ACCEPTANCE instrument for the p9-run-lower lane (brief ACCEPTANCE: "Cold
import at the NEW address, fresh interpreter per moved leaf"). Same discipline
as ``tests/test_shim_layer.py`` and ``tests/test_register_gate_cold_import.py``:
a subprocess-plus-stdout-marker check, never exit code alone.

RED-first (house rule 5): at commit 1 this file's own subject set is derived
live from the p9 fragment's ``landed: true`` rows, which do not exist yet, so
the non-vacuity guard below fails until the first ``git mv`` lands
``d1_resource_activation`` with its fragment row. Extended per landing commit
(the brief: "the file is p9's own, so extending it per commit is inside the
envelope").

C7 (design gate): the subject-set equality arm compares TWO INDEPENDENT
derivations -- (a) a live filesystem enumeration of
``ladder/run/{values,transport,provider,resource,trainer,construction}/*.py``
(excluding ``__init__.py``), turned into dotted addresses, and (b) the p9
fragment's own ``landed: true`` target rows, read straight from
``.claude/workflows/landed-set/p9-run-lower.json``. Comparing the fragment
with itself would be a tautology; comparing against the real directory
listing is not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_PATH = REPO_ROOT / ".claude" / "workflows" / "landed-set" / "p9-run-lower.json"
RUN_ROOT = REPO_ROOT / "src" / "entropic_substrate" / "ladder" / "run"
SUBPACKAGES = ("values", "transport", "provider", "resource", "trainer", "construction")

pytestmark = pytest.mark.unit

_MARKER = "COLD_IMPORT_OK"

_COLD_IMPORT_TEMPLATE = """
import importlib
importlib.import_module({dotted!r})
print({marker!r})
"""


def _fragment_landed_targets() -> list[str]:
    """Side (b): the p9 fragment's own ``landed: true`` target rows, read
    directly -- never merged with any other phase's fragment (p9's fragment
    is disjoint, so reading it alone is a correct and simpler derivation)."""
    if not FRAGMENT_PATH.exists():
        return []
    data = json.loads(FRAGMENT_PATH.read_text())
    return sorted(
        entry["target"]
        for entry in (data.get("modules") or {}).values()
        if entry.get("landed", False)
    )


def _filesystem_landed_targets() -> list[str]:
    """Side (a): a live directory listing of the six p9 destination
    sub-packages, turned into dotted addresses -- independent of the JSON
    fragment (reads the actual tree, not a declaration about it)."""
    out = []
    for sub in SUBPACKAGES:
        d = RUN_ROOT / sub
        if not d.is_dir():
            continue
        for f in d.glob("*.py"):
            if f.name == "__init__.py":
                continue
            out.append(f"entropic_substrate.ladder.run.{sub}.{f.stem}")
    return sorted(out)


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


def test_filesystem_and_fragment_derivations_of_the_landed_set_agree() -> None:
    """The C7 equality arm: two INDEPENDENT derivations of "what p9 has
    landed so far" must agree, with a non-vacuity guard so an empty==empty
    pass does not satisfy this test."""
    fs_side = _filesystem_landed_targets()
    fragment_side = _fragment_landed_targets()
    assert fs_side, "no p9 destination modules found on disk -- has the first git mv landed?"
    assert fragment_side, (
        "p9 fragment has no landed:true rows yet -- RED-first: this is expected "
        "before the first landing commit, and must go GREEN once it lands"
    )
    assert fs_side == fragment_side, (
        f"filesystem enumeration {fs_side} != fragment landed targets {fragment_side}"
    )


@pytest.mark.parametrize("dotted", _fragment_landed_targets())
def test_cold_import_of_landed_run_lower_target_succeeds_with_nothing_warming_it(
    dotted: str,
) -> None:
    """Fresh interpreter, NOTHING imported first: importing ``dotted``
    directly must succeed and print the marker -- not merely exit 0."""
    result = _run_cold_import(dotted)
    assert result.returncode == 0, f"cold import of {dotted} failed:\n{result.stderr}"
    assert _MARKER in result.stdout, (
        f"child process for {dotted} did not report success; "
        f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
    )


def test_cold_import_guard_is_not_fooled_by_a_process_that_silently_exits_0() -> None:
    """GREEN control for the instrument itself: a child that exits 0
    WITHOUT printing the marker must not satisfy this test's own check."""
    result = subprocess.run(
        [sys.executable, "-c", "pass"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert _MARKER not in result.stdout
