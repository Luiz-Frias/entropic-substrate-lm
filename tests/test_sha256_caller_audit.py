"""RED-first guard-teeth for the require_sha256 rename and its caller audit.

pre-for-1615 rung (FOR-1615, P14-SHA-SEQ). Three independent groups, each
failing for its own stated reason at RED (measured on the pre-rename tree,
`git rev-parse HEAD` = 9d90e4c421b96899c67ded8679ae8255f21526bd):

1. absence  -- ``require_sha256`` still resolves from
   ``kernel.identity.d1_primitives`` before the rename, so the ImportError
   this test expects does not happen: AssertionError on `hasattr`.
2. any-case -- ``require_sha256_hex_any_case`` does not exist yet: the
   deferred `from ... import require_sha256_hex_any_case` inside the test
   body raises ImportError.
3. audit    -- ``docs/plans/measurements/p14-sha-callers.txt`` does not
   exist yet: AssertionError on the file-existence check, before any other
   assertion in that test runs.

Neither name is imported at module (collection) level, precisely so a
single collection-time ImportError cannot red every arm at once (no guard
teeth would show for any one of them) -- see the brief's guard-teeth
recipe and house rule 5.

FOR-1626 end state (conductor ruling R1, 2026-10-08): the three retained
BOUNDARY sites were repointed to ``require_lowercase_sha256``, so
``require_sha256_hex_any_case`` had ZERO callers in ``src``. At that rung it
stayed defined and the live scan asserted it found no call site.

FOR-1636 end state (operator ruling 2026-10-10): the SHA sequence FOR-1615 ->
FOR-1626 -> FOR-1636 is closed. ``require_sha256_hex_any_case`` is REMOVED, so
the name no longer resolves from ``kernel.identity.d1_primitives``
(``test_any_case_name_no_longer_resolves``) and the live scan finds neither a
call site NOR a definition. ``docs/plans/measurements/p14-sha-callers.txt`` is
frozen history of the migration; its header and rows are still checked for
internal consistency below, and nothing is audited against the live tree any
more because there is nothing left to call.

Retracted by the FOR-1636 commit: (1) "stays defined (retirement is FOR-1636)"
is no longer true, the definition is gone; (2) "the definition-found control
still catches a misrooted scan" is no longer true, because with the definition
gone the expected scan result ``({}, False)`` is exactly what an empty or
misrooted scan returns. The misroot guard is now
``test_live_scan_root_contains_the_definition_module``, computed outside the
scan. RED for both new arms was measured with ``git diff --quiet
ee2c13ae421c70f1315315f8b862afc902af1b80 -- src`` exiting 0 (``src`` identical
to the parent tree; HEAD was 5289ca6e02d4ca89eb230bb8893ec832c45a174b, the
brief commit): the pin fails on `hasattr`, the end state fails on the
definition-present message.

``test_call_counter_counts_a_known_call`` and
``test_live_scan_aggregates_a_known_caller_end_to_end`` build their own
definition and caller on synthetic trees, so they keep proving the scan can
find a call and a definition; the end-state assertion alone could not (an
always-empty scan satisfies ``== {}``).
"""

from __future__ import annotations

import ast
import importlib
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_MODULE_DOTTED = "entropic_substrate.kernel.identity.d1_primitives"
_OLD_NAME = "require_sha256"
_NEW_NAME = "require_sha256_hex_any_case"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_AUDIT_PATH = _REPO_ROOT / "docs" / "plans" / "measurements" / "p14-sha-callers.txt"


def test_old_name_no_longer_resolves() -> None:
    """The loose kernel predicate no longer answers to its pre-rename name."""
    module = importlib.import_module(_MODULE_DOTTED)
    assert not hasattr(module, _OLD_NAME), (
        f"{_MODULE_DOTTED} must not carry an attribute named {_OLD_NAME} "
        "after the rename; no compatibility alias is permitted"
    )
    with pytest.raises(ImportError):
        from entropic_substrate.kernel.identity.d1_primitives import (  # noqa: F401
            require_sha256,
        )


def test_any_case_name_no_longer_resolves() -> None:
    """The any-case predicate is retired (FOR-1636); no alias, no stub."""
    module = importlib.import_module(_MODULE_DOTTED)
    assert not hasattr(module, _NEW_NAME), (
        f"{_MODULE_DOTTED} must not carry an attribute named {_NEW_NAME} "
        "after FOR-1636; boundary digests use require_lowercase_sha256 and "
        "no compatibility alias is permitted"
    )
    with pytest.raises(ImportError):
        from entropic_substrate.kernel.identity.d1_primitives import (  # noqa: F401
            require_sha256_hex_any_case,
        )


def _is_package(path: Path) -> bool:
    return path.name == "__init__.py"


def _module_dotted(path: Path, root: Path) -> str:
    """Dotted module name for a file under ``root`` (``root`` = the ``src`` dir).

    ``root``'s children already start with the ``entropic_substrate`` package
    segment, so it is not prepended again.
    """
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative_module(
    module_dotted: str, is_package: bool, node: ast.ImportFrom
) -> str | None:
    if node.level == 0:
        return node.module
    package = module_dotted if is_package else module_dotted.rsplit(".", 1)[0]
    parts = package.split(".")
    if node.level - 1 > len(parts):
        return None
    base_parts = parts[: len(parts) - (node.level - 1)]
    base = ".".join(base_parts)
    if node.module:
        return f"{base}.{node.module}" if base else node.module
    return base or None


def _bound_names_for_file(tree: ast.AST, module_dotted: str, is_package: bool) -> set[str]:
    """Local names this file has bound to the target symbol.

    Covers absolute and relative ImportFrom, aliased or not. The
    definition module binds its own name directly (no import needed).
    """
    bound: set[str] = set()
    if module_dotted == _MODULE_DOTTED:
        bound.add(_NEW_NAME)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target_mod = _resolve_relative_module(module_dotted, is_package, node)
            if target_mod == _MODULE_DOTTED:
                for alias in node.names:
                    if alias.name == _NEW_NAME:
                        bound.add(alias.asname or alias.name)
    return bound


def _count_calls(tree: ast.AST, bound_names: set[str]) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in bound_names:
                count += 1
            elif isinstance(func, ast.Attribute) and func.attr == _NEW_NAME:
                count += 1
    return count


def _scan_live_call_counts(
    src_root: Path, repo_root: Path = _REPO_ROOT
) -> tuple[Counter[str], bool]:
    """AST-scan src_root for every call-site reference to the new symbol.

    ``repo_root`` only relativizes the reported paths; it defaults to the real
    repository and is threaded so the end-to-end control can scan a fixture tree.

    Returns (path -> call count, definition_found). ``definition_found`` is
    True when d1_primitives.py defines the symbol. Since FOR-1636 the live
    expectation is ``({}, False)``, which an empty or misrooted scan also
    returns; the misroot guard is therefore
    ``test_live_scan_root_contains_the_definition_module``, not this flag.
    """
    counts: Counter[str] = Counter()
    definition_found = False
    py_files = sorted(src_root.rglob("*.py"))
    for path in py_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        module_dotted = _module_dotted(path, src_root)
        if module_dotted == _MODULE_DOTTED:
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == _NEW_NAME:
                    definition_found = True
        bound = _bound_names_for_file(tree, module_dotted, _is_package(path))
        call_count = _count_calls(tree, bound)
        if call_count:
            rel = path.relative_to(repo_root).as_posix()
            counts[rel] = call_count
    return counts, definition_found


def test_call_counter_counts_a_known_call() -> None:
    """Positive control of the live scan's own per-file composition.

    The end-state assertion (``live_counts == {}``) is satisfied by a counter
    that finds nothing, so the counter is proven here on a snippet with one live
    call of each shape the scan must count: an absolute import called by name, an
    aliased absolute import, a relative import (resolved against the snippet's
    module, which is NOT the definition module), and an attribute call. A counter
    that drops any one of those branches under-counts and fails the total.
    """
    snippet = (
        f"from {_MODULE_DOTTED} import {_NEW_NAME}\n"
        f"from {_MODULE_DOTTED} import {_NEW_NAME} as aliased\n"
        f"from ..identity.d1_primitives import {_NEW_NAME} as relative\n"
        "import entropic_substrate.kernel.identity.d1_primitives as d1\n"
        "\n"
        "\n"
        "def check(value):\n"
        f"    {_NEW_NAME}(value, label='control')\n"
        "    aliased(value, label='control')\n"
        "    relative(value, label='control')\n"
        f"    d1.{_NEW_NAME}(value, label='control')\n"
    )
    tree = ast.parse(snippet)
    bound = _bound_names_for_file(tree, "entropic_substrate.kernel.codec.strict_json", False)
    assert bound == {_NEW_NAME, "aliased", "relative"}
    assert _count_calls(tree, bound) == 4


def test_live_scan_aggregates_a_known_caller_end_to_end(tmp_path: Path) -> None:
    """Positive control of ``_scan_live_call_counts`` itself, on a filesystem.

    ``test_call_counter_counts_a_known_call`` drives only the per-file helpers, so
    an end-state ``live_counts == {}`` would stay green if the scan dropped its
    aggregation write or resolved relative imports against the wrong package
    (``_is_package`` always True). This builds a small ``entropic_substrate``
    tree: the definition module, one relative-aliased caller in a NON-package
    file, and a package ``__init__`` that must not be counted. The scan must
    report exactly that one caller with one call.
    """
    src = tmp_path / "src"
    identity = src / "entropic_substrate" / "kernel" / "identity"
    codec = src / "entropic_substrate" / "kernel" / "codec"
    identity.mkdir(parents=True)
    codec.mkdir(parents=True)
    (src / "entropic_substrate" / "__init__.py").write_text("", encoding="utf-8")
    (src / "entropic_substrate" / "kernel" / "__init__.py").write_text("", encoding="utf-8")
    (identity / "__init__.py").write_text("", encoding="utf-8")
    (codec / "__init__.py").write_text("", encoding="utf-8")
    (identity / "d1_primitives.py").write_text(
        f"def {_NEW_NAME}(value, *, label):\n    return value\n", encoding="utf-8"
    )
    (codec / "caller.py").write_text(
        f"from ..identity.d1_primitives import {_NEW_NAME} as relative\n"
        "\n"
        "\n"
        "def check(value):\n"
        "    relative(value, label='control')\n",
        encoding="utf-8",
    )

    counts, positive_control = _scan_live_call_counts(src, repo_root=tmp_path)

    assert positive_control
    assert counts == Counter({"src/entropic_substrate/kernel/codec/caller.py": 1})


def _parse_audit(text: str) -> tuple[dict[str, int], list[dict[str, str]]]:
    header: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if ":" in body:
                key, _, value = body.partition(":")
                key = key.strip()
                value = value.strip()
                if key in {"importer_files", "files", "rows", "boundary", "in-process"}:
                    try:
                        header[key] = int(value)
                    except ValueError:
                        pass
            continue
        fields = [f.strip() for f in line.split("|")]
        if len(fields) != 5:
            continue
        path_line, symbol, cls, evidence, owner = fields
        path, _, lineno = path_line.rpartition(":")
        rows.append(
            {
                "path": path,
                "line": lineno,
                "symbol": symbol,
                "class": cls,
                "evidence": evidence,
                "owner": owner,
            }
        )
    return header, rows


def test_live_scan_root_contains_the_definition_module() -> None:
    """Non-vacuous root control for the end-state scan.

    The end-state result ``({}, False)`` is also what an empty or misrooted scan
    returns, so absence alone proves nothing. This asserts the scan root really
    contains the module that used to define the symbol, under the scan's own
    ``_module_dotted`` mapping. It is computed outside ``_scan_live_call_counts``.
    """
    assert _SRC_ROOT.is_dir(), f"scan root is not a directory: {_SRC_ROOT}"
    mapped = {_module_dotted(path, _SRC_ROOT) for path in _SRC_ROOT.rglob("*.py")}
    assert _MODULE_DOTTED in mapped, (
        f"no *.py under {_SRC_ROOT} maps to {_MODULE_DOTTED}; the end-state scan "
        "would pass vacuously"
    )


def test_caller_audit_is_committed_and_covers_the_live_tree() -> None:
    """The frozen audit file is internally consistent and the live tree is clean.

    End state (FOR-1636): the live scan of ``src`` finds no call site and no
    definition. Fails loud (never skip/xfail) on a missing audit file, an
    inconsistent header, or any surviving reference. That the scan actually
    covered the definition module is pinned separately by
    ``test_live_scan_root_contains_the_definition_module``.
    """
    assert _AUDIT_PATH.is_file(), f"audit file missing: {_AUDIT_PATH}"

    header, rows = _parse_audit(_AUDIT_PATH.read_text(encoding="utf-8"))

    assert header, f"audit file has no parseable header: {_AUDIT_PATH}"
    assert header.get("rows") == len(rows), (
        f"audit header claims rows={header.get('rows')} but {len(rows)} data rows were parsed"
    )
    class_counts = Counter(row["class"] for row in rows)
    assert set(class_counts) <= {"boundary", "in-process"}, (
        f"audit rows use class words outside {{boundary, in-process}}: {sorted(class_counts)}"
    )
    assert header.get("boundary") == class_counts.get("boundary", 0), (
        f"audit header boundary={header.get('boundary')} does not match "
        f"{class_counts.get('boundary', 0)} boundary rows"
    )
    assert header.get("in-process") == class_counts.get("in-process", 0), (
        f"audit header in-process={header.get('in-process')} does not match "
        f"{class_counts.get('in-process', 0)} in-process rows"
    )

    audited_files = {row["path"] for row in rows}
    assert header.get("files") == len(audited_files), (
        f"audit header files={header.get('files')} does not match "
        f"{len(audited_files)} distinct audited files"
    )

    live_counts, definition_found = _scan_live_call_counts(_SRC_ROOT)
    assert not definition_found, (
        f"{_NEW_NAME} is defined again under {_SRC_ROOT}; FOR-1636 retired it and "
        "boundary digests use require_lowercase_sha256"
    )
    assert live_counts == {}, (
        f"{_NEW_NAME} has live call sites {dict(live_counts)}; FOR-1636 retired the "
        "name, so any reference is a defect, not an audit row"
    )
