"""Budget on test modules importing other test modules.

MEASURED 2026-11-21 at 13276219: test files use each other as fixture
libraries. A ``test_*`` module that another imports cannot move, be renamed
or be retired without touching its importers, and importing it under a second
module name runs its module-level setup twice. Shared material belongs in
``tests/support``.

This is a ratchet, not a ban. ``EDGE_BUDGET`` is the measured number of
distinct importer-to-provider edges and must EQUAL the live count: adding an
edge fails, and so does removing one without lowering the budget, so the
number can only walk down: 387 edges onto 94 provider modules, plus 6 sites
where a ``tests/support`` module imports a test module, which inverts the
intended direction. Two spellings are banned outright because the move to
folders removed every instance: relative imports between test modules and
bare ``import test_x``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import repo_paths

pytestmark = pytest.mark.unit

EDGE_BUDGET = 387
PROVIDER_BUDGET = 94
SUPPORT_SITE_BUDGET = 6


@dataclass
class Scan:
    edges: set[tuple[str, str]] = field(default_factory=set)
    relative: list[str] = field(default_factory=list)
    bare: list[str] = field(default_factory=list)
    support_importing_tests: list[str] = field(default_factory=list)

    @property
    def providers(self) -> set[str]:
        return {provider for _importer, provider in self.edges}


def _imported_modules(node: ast.AST) -> list[tuple[int, str]]:
    if isinstance(node, ast.ImportFrom):
        base = node.module or ""
        names = [f"{base}.{alias.name}" if base else alias.name for alias in node.names]
        return [(node.level, base), *[(node.level, name) for name in names]]
    if isinstance(node, ast.Import):
        return [(0, alias.name) for alias in node.names]
    return []


def scan(tests_root: Path) -> Scan:
    """Every test-to-test import under ``tests_root``, by spelling."""
    test_modules = {path.stem for path in tests_root.rglob("test_*.py")}
    result = Scan()
    for path in sorted(tests_root.rglob("*.py")):
        where = path.relative_to(tests_root).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            for level, module in _imported_modules(node):
                leaf = module.rsplit(".", 1)[-1]
                if leaf not in test_modules or leaf == path.stem:
                    continue
                site = f"{where}:{getattr(node, 'lineno', 0)}"
                if not path.name.startswith("test_"):
                    result.support_importing_tests.append(site)
                elif level:
                    result.relative.append(site)
                elif "." not in module:
                    result.bare.append(site)
                result.edges.add((path.stem, leaf))
    return result


def _write(root: Path, name: str, body: str) -> None:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)


def test_scan_counts_one_edge_per_importer_provider_pair(tmp_path: Path) -> None:
    _write(tmp_path, "unit/test_provider.py", "X = 1\n")
    _write(
        tmp_path,
        "unit/test_user.py",
        "from tests.unit.test_provider import X\nimport tests.unit.test_provider as again\n",
    )
    _write(tmp_path, "unit/test_alone.py", "import json\nfrom helpers import thing\n")
    found = scan(tmp_path)
    assert found.edges == {("test_user", "test_provider")}
    assert found.relative == found.bare == found.support_importing_tests == []


def test_scan_names_each_banned_spelling(tmp_path: Path) -> None:
    _write(tmp_path, "a/test_provider.py", "X = 1\n")
    _write(tmp_path, "a/test_relative.py", "from .test_provider import X\n")
    _write(tmp_path, "a/test_from_dot.py", "from . import test_provider\n")
    _write(tmp_path, "a/test_bare.py", "from test_provider import X\n")
    _write(tmp_path, "support/helpers.py", "from tests.a.test_provider import X\n")
    found = scan(tmp_path)
    assert found.relative == ["a/test_from_dot.py:1", "a/test_relative.py:1"]
    assert found.bare == ["a/test_bare.py:1"]
    assert found.support_importing_tests == ["support/helpers.py:1"]
    assert len(found.edges) == 4


def test_scan_ignores_a_module_importing_itself_and_lazy_imports_still_count(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "test_provider.py", "X = 1\n")
    _write(
        tmp_path,
        "test_lazy.py",
        "def test_x():\n    from tests.test_provider import X\n    assert X\n",
    )
    assert scan(tmp_path).edges == {("test_lazy", "test_provider")}


def test_banned_spellings_stay_at_zero() -> None:
    found = scan(repo_paths.TESTS_ROOT)
    assert found.relative == []
    assert found.bare == []


def test_support_modules_importing_tests_equals_the_budget() -> None:
    found = scan(repo_paths.TESTS_ROOT)
    assert len(found.support_importing_tests) == SUPPORT_SITE_BUDGET, found.support_importing_tests


def test_edge_count_equals_the_budget() -> None:
    found = scan(repo_paths.TESTS_ROOT)
    assert len(found.edges) == EDGE_BUDGET, (
        f"{len(found.edges)} test-to-test import edges, budget {EDGE_BUDGET}. More: move the shared "
        "material to tests/support instead. Fewer: lower EDGE_BUDGET in the same change."
    )


def test_provider_count_equals_the_budget() -> None:
    found = scan(repo_paths.TESTS_ROOT)
    assert len(found.providers) == PROVIDER_BUDGET, (
        f"{len(found.providers)} test modules are imported by other tests, budget {PROVIDER_BUDGET}."
    )
