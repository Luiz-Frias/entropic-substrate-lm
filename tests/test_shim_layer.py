"""Pytest for the p3-quarantine cold-import guard and the facade-retraction
guard tooth (ACCEPTANCE: "Cold-import guard ... a test spawns a FRESH
interpreter (subprocess) per facade package and imports
entropic_substrate.<facade>.<child> for one child of each of the seven
facades, old prefix first, asserting success and sys.modules identity with
the __legacy module; the in-process sweep's sorted() order can warm the
chain and must not be read as evidence on its own").

INSTRUMENT (architect constraint, design gate 2026-09-20): one fresh
subprocess per facade, launched with `sys.executable` (never a hardcoded
`python`, which would silently test the ambient interpreter instead of
THIS venv's) and the parent test's own environment, importing the
old-prefix dotted path FIRST -- nothing warms `__legacy` ahead of it. The
CHILD process itself evaluates the `sys.modules` identity comparison and
prints an explicit marker; the parent asserts on that stdout text AND the
return code, never on the exit code alone (a child that silently skips
the comparison still exits 0, which is the exact defect the ruling was
written against -- p0's own sorted() sweep warms `__legacy` first because
`"_"` sorts before every letter, so an exit-code-only guard would stay
green through the real hazard)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

pytestmark = pytest.mark.unit

# facade package -> one real child module each restored __init__.py imports
# from (see each facade's docstring/import list at src/entropic_substrate/
# <facade>/__init__.py, restored verbatim after the git mv).
FACADE_CHILDREN = {
    "data": "batching",
    "geometry": "kappa",
    "init": "fisher",
    "layers": "kappa_norm",
    "models": "backbone",
    "observation": "byte_codec",
    "predictive_state": "future_law",
}

_MARKER = "COLD_IMPORT_IDENTICAL"

_COLD_IMPORT_CHILD_TEMPLATE = """
import sys
old_dotted = {old_dotted!r}
legacy_dotted = {legacy_dotted!r}
import importlib
old = importlib.import_module(old_dotted)
legacy = importlib.import_module(legacy_dotted)
identical = sys.modules[old_dotted] is sys.modules[legacy_dotted]
print({marker!r} if identical else "COLD_IMPORT_DIFFERENT")
"""


def _run_cold_import(facade: str, child: str) -> subprocess.CompletedProcess[str]:
    old_dotted = f"entropic_substrate.{facade}.{child}"
    legacy_dotted = f"entropic_substrate.__legacy.{facade}.{child}"
    code = _COLD_IMPORT_CHILD_TEMPLATE.format(
        old_dotted=old_dotted, legacy_dotted=legacy_dotted, marker=_MARKER
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )


@pytest.mark.parametrize("facade,child", sorted(FACADE_CHILDREN.items()))
def test_cold_import_of_facade_child_is_sys_modules_identical_to_legacy(
    facade: str, child: str
) -> None:
    """Fresh interpreter, old prefix imported FIRST (nothing else in this
    one-line child script touches __legacy ahead of it): the shim's
    sys.modules rebind must make the old and __legacy module objects
    IDENTICAL, asserted from the child's own stdout marker AND its
    return code -- not the return code alone."""
    result = _run_cold_import(facade, child)
    assert result.returncode == 0, (
        f"cold import of entropic_substrate.{facade}.{child} failed:\n{result.stderr}"
    )
    assert _MARKER in result.stdout, (
        f"child process did not report identity for {facade}.{child}; "
        f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
    )


def test_cold_import_guard_is_not_fooled_by_a_process_that_silently_exits_0() -> None:
    """GREEN control for the instrument itself: a child process that exits 0
    WITHOUT printing the marker (the defect class the ruling is written
    against -- an exit-code-only guard cannot tell this apart from a real
    identity pass) must NOT satisfy this test's own marker assertion."""
    result = subprocess.run(
        [sys.executable, "-c", "pass"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert _MARKER not in result.stdout


# ---------------------------------------------------------------------------
# Guard tooth (house rule 5, negative arm): restoring one moved facade
# copy's ORIGINAL bytes under __legacy/<pkg>/__init__.py must make a COLD
# import of one of its children raise ImportError on a partially
# initialized module. Performed against the REAL committed tree and
# reverted in a `finally`, snapshot-based so a failure mid-test cannot
# leave the working tree dirty.
# ---------------------------------------------------------------------------


def test_restoring_one_facade_original_bytes_breaks_cold_import_RED_case() -> None:
    """RED-first, both arms in one test (house rule 5): the retracted
    docstring-only file at __legacy/data/__init__.py is what MAKES the
    cold import above pass; putting the ORIGINAL re-export body back at
    that __legacy path (the exact state the design-gate ruling retracted)
    must reintroduce the partially-initialized-module ImportError this
    phase's facade-emptying ruling exists to prevent. The GREEN case (the
    committed, emptied file) is proven by the parametrized test above,
    which runs first in file order against the real, unmutated tree."""
    facade = "data"
    child = "batching"
    legacy_init = SRC / "entropic_substrate" / "__legacy" / facade / "__init__.py"
    old_init = SRC / "entropic_substrate" / facade / "__init__.py"
    emptied_snapshot = legacy_init.read_text()
    original_body = old_init.read_text()
    assert emptied_snapshot != original_body, (
        "fixture assumption violated: the __legacy copy is no longer the "
        "emptied docstring form, or the old-path copy no longer carries "
        "the original re-export body -- this test's precondition is stale"
    )
    try:
        legacy_init.write_text(original_body)
        result = _run_cold_import(facade, child)
        assert result.returncode != 0, (
            "restoring the facade's original bytes was supposed to break "
            f"the cold import, but it still exited 0: {result.stdout!r}"
        )
        assert "partially initialized module" in result.stderr, result.stderr
        assert "ImportError" in result.stderr, result.stderr
    finally:
        legacy_init.write_text(emptied_snapshot)
    # Sanity: the revert actually landed, so the suite's OWN tree is clean
    # for every test that runs after this one (including the parametrized
    # GREEN case if pytest ever reorders).
    assert legacy_init.read_text() == emptied_snapshot
    post_revert = _run_cold_import(facade, child)
    assert post_revert.returncode == 0, post_revert.stderr
    assert _MARKER in post_revert.stdout
