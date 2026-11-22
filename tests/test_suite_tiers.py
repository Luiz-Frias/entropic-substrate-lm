"""Suite tiers: the ``slow`` marker carves a fast tier out of the unit suite.

MEASURED 2026-11-21 at main 8c1f0c13 (``pytest -m unit -n auto --dist
loadfile``, 8 cores, one run): 10,961 unit tests, 2,335 summed
test-seconds, 392 s wall. ``tests/test_cold_import.py`` alone is 360 s (404
subprocess imports) and the 59 test functions with any case at or above 5 s
are another 633 s. Those carry ``slow``; ``just test-fast`` and the prek
pre-commit stage deselect them (fast tier: 10,489 tests, 194.5 s wall, same
regime, one run).

The hazard this file guards is the tier split eating the gate: a ``slow``
test that the FULL tier no longer selects is a test nothing runs. So the
pins are on both arms: the fast recipe must deselect ``slow``, and no full
recipe or pre-push hook may.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS = REPO_ROOT / "tests"
SLOW = "slow"
TIER_MARKS = frozenset({"unit", "integration"})

pytestmark = pytest.mark.unit


def _recipe_body(name: str) -> str:
    text = (REPO_ROOT / "justfile").read_text()
    match = re.search(rf"^{re.escape(name)}\b[^\n]*:[^\n]*\n((?:[ \t]+[^\n]*\n|\n)+)", text, re.M)
    assert match is not None, f"justfile has no recipe {name!r}"
    return match.group(1)


def _module_marks(tree: ast.Module) -> set[str]:
    marks: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue
        marks |= {n.attr for n in ast.walk(node.value) if isinstance(n, ast.Attribute)}
    return marks - {"mark"}


def _files_using_slow() -> list[Path]:
    return sorted(
        p
        for p in TESTS.glob("test_*.py")
        if p.name != Path(__file__).name and f"pytest.mark.{SLOW}" in p.read_text()
    )


def test_slow_marker_is_registered() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    names = {m.split(":", 1)[0].strip() for m in config["tool"]["pytest"]["ini_options"]["markers"]}
    assert SLOW in names


def test_fast_recipe_deselects_slow_and_stays_inside_unit() -> None:
    body = _recipe_body("test-fast")
    assert '-m "unit and not slow"' in body


@pytest.mark.parametrize("recipe", ["test", "test-all"])
def test_full_recipes_never_deselect_slow(recipe: str) -> None:
    assert SLOW not in _recipe_body(recipe)


def test_check_gate_runs_the_full_unit_recipe() -> None:
    text = (REPO_ROOT / "justfile").read_text()
    match = re.search(r"^check:([^\n]*)$", text, re.M)
    assert match is not None
    prerequisites = match.group(1).split()
    assert "test" in prerequisites
    assert "test-fast" not in prerequisites


def test_pre_push_hook_never_deselects_slow() -> None:
    text = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    pytest_arg_lines = [line for line in text.splitlines() if '"pytest"' in line]
    assert pytest_arg_lines, "no pytest hook found; the pin would be vacuous"
    unmarked = [line for line in pytest_arg_lines if '"-m"' not in line]
    assert unmarked, "no full-suite pytest hook left"
    assert all(SLOW not in line for line in unmarked)


def test_pre_commit_stage_runs_the_fast_tier_and_nothing_narrower() -> None:
    text = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    marked = [line for line in text.splitlines() if '"pytest"' in line and '"-m"' in line]
    assert len(marked) == 1, "exactly one marker-selected pytest hook: the commit stage"
    assert '"-m", "unit and not slow"' in marked[0]


def test_cold_import_module_is_slow() -> None:
    tree = ast.parse((TESTS / "test_cold_import.py").read_text())
    assert SLOW in _module_marks(tree)


def test_slow_is_actually_used() -> None:
    assert len(_files_using_slow()) >= 2, "an unused tier marker makes every pin above vacuous"


def test_every_slow_file_keeps_a_tier_marker_so_the_full_gate_selects_it() -> None:
    orphans = [
        p.name
        for p in _files_using_slow()
        if not (_module_marks(ast.parse(p.read_text())) & TIER_MARKS)
    ]
    assert orphans == []
