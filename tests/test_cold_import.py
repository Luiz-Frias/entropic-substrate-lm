"""P13-COLD-IMPORT-CONSOLIDATION: the eight per-phase cold-import tests
(one per landing phase, each deriving its subject set from that phase's own
``.claude/workflows/landed-set/*.json`` fragment, ``gen_shims`` or
``tools/mapping.json``) are retired with the generator and consolidated
into ONE whole-tree cold-import test whose subject set is every module
under the five landed roots, read directly from the tree (never a fragment,
never ``gen_shims``, never ``tools/mapping.json``).

P13-SCC-ARM-REHOMED: the P12-SCC-TEETH arm
(``test_historical_composition_cold_import.py::test_every_landed_module_imports_only_legacy_free_resolvable_targets``)
is NOT retired with the eight files -- it is re-homed here, in-process, with
a tree-derived subject set over the five landed roots (both assertions
kept: no import target under ``entropic_substrate.__legacy``, and every
import target resolves to an existing module).
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "entropic_substrate"

LANDED_ROOTS = ("kernel", "core", "ladder", "historical", "composition")

pytestmark = pytest.mark.unit

_MARKER = "COLD_IMPORT_OK"

_COLD_IMPORT_TEMPLATE = """
import importlib
importlib.import_module({dotted!r})
print({marker!r})
"""


def _tree_derived_landed_modules() -> list[str]:
    """Every importable module dotted name under the five landed roots,
    read directly from the tracked tree -- never a fragment, gen_shims, or
    tools/mapping.json (both retire with the generator in this commit)."""
    out: list[str] = []
    for root_name in LANDED_ROOTS:
        root_dir = SRC_ROOT / root_name
        if root_dir.is_file():
            # a bare leaf module at src/entropic_substrate/<root>.py (none
            # expected among the five landed roots, but handled for parity
            # with the package case below)
            out.append(f"entropic_substrate.{root_name}")
            continue
        if not root_dir.is_dir():
            continue
        for p in sorted(root_dir.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(SRC_ROOT.parent).with_suffix("")
            parts = list(rel.parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            out.append(".".join(parts))
        # the root package's own __init__, if the loop above skipped it
        # because rglob does not include the directory itself
    return sorted(set(out))


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
    assert result.returncode == 0, f"cold import of {dotted} failed:\n{result.stderr}"
    assert _MARKER in result.stdout, (
        f"child process for {dotted} did not report success; "
        f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
    )


def test_landed_subject_set_is_non_empty() -> None:
    targets = _tree_derived_landed_modules()
    assert targets, (
        "zero modules found under the five landed roots -- the tree derivation is broken"
    )


@pytest.mark.parametrize("dotted", _tree_derived_landed_modules())
def test_cold_import_of_landed_module_succeeds_with_nothing_warming_it(dotted: str) -> None:
    _assert_cold_import_ok(_run_cold_import(dotted), dotted)


def test_cold_import_guard_accepts_a_real_import() -> None:
    result = _run_cold_import("json")
    assert result.returncode == 0
    assert _MARKER in result.stdout


def test_cold_import_guard_rejects_a_failed_import() -> None:
    """Negative control (P13-COLD-IMPORT-CONSOLIDATION addendum 2): the
    landed-prefixed control -- never a bare root-level name, which the
    retired-namespace rule would itself flag."""
    result = _run_cold_import("entropic_substrate.kernel.__no_such_module__")
    assert result.returncode != 0
    assert _MARKER not in result.stdout
    with pytest.raises(AssertionError):
        _assert_cold_import_ok(result, "entropic_substrate.kernel.__no_such_module__")


def test_cold_import_guard_is_not_fooled_by_a_process_that_silently_exits_0() -> None:
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


# ---------------------------------------------------------------------------
# P13-SCC-ARM-REHOMED: the P12-SCC-TEETH arm, tree-derived subject set.
# ---------------------------------------------------------------------------


def test_every_landed_module_imports_only_legacy_free_resolvable_targets() -> None:
    """Re-homed from tests/test_historical_composition_cold_import.py (that
    file, and the seven other per-phase cold-import files, retire in this
    commit). Both assertions kept: no import target under the retired
    quarantine namespace, and every import target resolves to an existing
    module file or package under src/entropic_substrate."""
    targets = _tree_derived_landed_modules()
    assert targets, "no landed modules yet -- nothing to walk"
    # Composed at runtime, never a literal chain (P13-NAMESPACE-RULE): a
    # literal "entropic_substrate.__legacy" here would itself be flagged by
    # tests/test_no_old_prefix.py's whole-tree scan.
    legacy_prefix = "entropic_substrate" + "." + "__legacy"

    def _is_package(mod: str) -> bool:
        spec = importlib.util.find_spec(mod)
        return spec is not None and bool(spec.submodule_search_locations)

    def _resolves(mod: str) -> bool:
        spec = importlib.util.find_spec(mod)
        return spec is not None and bool(spec.origin or spec.submodule_search_locations)

    def _import_targets(node: ast.AST, importer: str, importer_is_pkg: bool) -> list[str]:
        if isinstance(node, ast.Import):
            return [a.name for a in node.names if a.name.startswith("entropic_substrate.")]
        if not isinstance(node, ast.ImportFrom):
            return []
        if node.level:
            package = importer if importer_is_pkg else importer.rpartition(".")[0]
            for _ in range(node.level - 1):
                package = package.rpartition(".")[0]
            base = f"{package}.{node.module}" if node.module else package
        else:
            base = node.module or ""
        if not base.startswith("entropic_substrate"):
            return []
        if not _is_package(base):
            return [base]
        out = [base]
        pkg = None if base.startswith(legacy_prefix) else importlib.import_module(base)
        for alias in node.names:
            child = f"{base}.{alias.name}"
            if _resolves(child):
                out.append(child)
            elif pkg is not None and hasattr(pkg, alias.name):
                continue
            else:
                out.append(child)
        return out

    for dotted in targets:
        spec = importlib.util.find_spec(dotted)
        assert spec is not None and spec.origin, f"{dotted} has no resolvable origin"
        source = Path(spec.origin).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=spec.origin)
        importer_is_pkg = bool(spec.submodule_search_locations)
        for node in ast.walk(tree):
            modules = _import_targets(node, dotted, importer_is_pkg)
            for mod in modules:
                assert not mod.startswith(legacy_prefix), (
                    f"{dotted} (line {node.lineno}) imports {mod!r} -- a landed module must "
                    "never import the retired quarantine namespace directly"
                )
                assert _resolves(mod), (
                    f"{dotted} (line {node.lineno}) imports {mod!r}, which does not resolve to "
                    "an existing module file or package in the tree"
                )
