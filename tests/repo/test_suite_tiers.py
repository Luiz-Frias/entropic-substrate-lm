"""Suite tiers: the ``slow`` marker carves a fast tier out of the unit suite.

MEASURED 2026-11-21 at main 8c1f0c13 (``pytest -m unit -n auto --dist
loadfile``, 8 cores, one run): 10,961 unit tests, 2,335 summed
test-seconds, 392 s wall. ``tests/architecture/test_cold_import.py`` alone is 360 s (404
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
import yaml

import repo_paths

REPO_ROOT = repo_paths.REPO_ROOT
TESTS = REPO_ROOT / "tests"
SLOW = "slow"
TIER_MARKS = frozenset({"unit", "integration"})

pytestmark = pytest.mark.unit


def _recipe_body(name: str) -> str:
    text = (REPO_ROOT / "justfile").read_text()
    match = re.search(
        rf"^{re.escape(name)}(?![\w-])[^\n]*:[^\n]*\n((?:[ \t]+[^\n]*\n|\n)+)", text, re.M
    )
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
        for p in TESTS.rglob("test_*.py")
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


def _suite_hooks() -> dict[str, list[str]]:
    """stage -> args of the one hook that calls scripts/suite_stages.py there."""
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    found: dict[str, list[str]] = {}
    for repo in config["repos"]:
        for hook in repo["hooks"]:
            args = [str(arg) for arg in hook.get("args", [])]
            if "scripts/suite_stages.py" not in args:
                continue
            for stage in hook["stages"]:
                assert stage not in found, f"two suite hooks at {stage}"
                found[stage] = args[args.index("scripts/suite_stages.py") + 1 :]
    return found


def test_each_hook_stage_calls_its_own_suite_stage() -> None:
    assert _suite_hooks() == {
        "pre-commit": ["pre-commit"],
        "post-commit": ["post-commit"],
        "pre-push": ["pre-push"],
    }


def test_no_hook_calls_pytest_directly_any_more() -> None:
    text = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    assert [line for line in text.splitlines() if '"pytest"' in line] == [], (
        "a hook that calls pytest itself bypasses the stage table in scripts/suite_stages.py"
    )


def test_the_post_commit_hook_type_is_installed_by_default() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    assert {"pre-commit", "post-commit", "pre-push"} <= set(config["default_install_hook_types"])


SUBMIT = "gt submit --stack --no-edit --no-interactive --always"


def test_slim_ship_it_guards_scope_then_submits_then_runs_the_post_step() -> None:
    body = _recipe_body("ship-it")
    assert "container-test" not in body
    assert "tests/suite_lock.py ship-it --" in body
    order = [body.index("scripts/ship_scope.py"), body.index(SUBMIT), body.index("just post-ship")]
    assert order == sorted(order), "scope guard, then submit, then the post step"


def test_full_ship_it_runs_check_and_the_live_gate_before_submitting() -> None:
    body = _recipe_body("ship-it-full")
    assert "tests/suite_lock.py ship-it-full --" in body
    order = [
        body.index("just check"),
        body.index("just container-test"),
        body.index(SUBMIT),
        body.index("just post-ship"),
    ]
    assert order == sorted(order)
    assert "ship_scope" not in body, (
        "the full recipe is where the guard sends you; it must not refuse"
    )


def test_the_post_step_runs_the_post_ship_stage_and_the_dead_code_report() -> None:
    body = _recipe_body("post-ship")
    assert "scripts/suite_stages.py run post-ship" in body
    assert "scripts/dead_code_report.py" in body


def test_cold_import_module_is_slow() -> None:
    tree = ast.parse((TESTS / "architecture" / "test_cold_import.py").read_text())
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


def test_every_unit_module_carries_the_unit_marker() -> None:
    # `just test` and `just check` select with `-m unit`; a module under
    # tests/unit without the marker is collected by the hook stages and
    # silently skipped by the gate. Found 2026-11-21: two modules, 11 tests.
    unmarked = sorted(
        path.relative_to(repo_paths.REPO_ROOT).as_posix()
        for path in (repo_paths.TESTS_ROOT / "unit").rglob("test_*.py")
        if "mark.unit" not in path.read_text(encoding="utf-8")
    )
    assert unmarked == []
