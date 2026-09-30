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
_LABEL = "test digest"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_AUDIT_PATH = _REPO_ROOT / "docs" / "plans" / "measurements" / "p14-sha-callers.txt"

_EXPECTED_MESSAGE = f"{_LABEL} must be a 64-character hexadecimal SHA-256 digest"


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


def test_any_case_accepts_and_rejects() -> None:
    """require_sha256_hex_any_case accepts any-case hex, rejects the same way require_sha256 did."""
    from entropic_substrate.kernel.identity.d1_primitives import (
        require_sha256_hex_any_case,
    )

    upper = "ABCDEF0123456789" * 4
    mixed = "aBcDeF0123456789" * 4
    assert len(upper) == 64
    assert len(mixed) == 64

    # must not raise
    require_sha256_hex_any_case(upper, label=_LABEL)
    require_sha256_hex_any_case(mixed, label=_LABEL)

    too_short = upper[:-1]
    too_long = upper + "0"
    non_hex = "G" + upper[1:]
    assert len(too_short) == 63
    assert len(too_long) == 65
    assert len(non_hex) == 64

    for bad in (too_short, too_long, non_hex):
        with pytest.raises(ValueError) as excinfo:
            require_sha256_hex_any_case(bad, label=_LABEL)
        assert str(excinfo.value) == _EXPECTED_MESSAGE

    with pytest.raises(ValueError) as excinfo:
        require_sha256_hex_any_case(12345, label=_LABEL)  # type: ignore[arg-type]
    assert str(excinfo.value) == _EXPECTED_MESSAGE


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


def _scan_live_call_counts(src_root: Path) -> tuple[Counter[str], bool]:
    """AST-scan src_root for every call-site reference to the new symbol.

    Returns (path -> call count, positive_control_found). The positive
    control is the function definition itself, inside d1_primitives.py --
    an empty or misrooted scan must fail loud, never pass silently.
    """
    counts: Counter[str] = Counter()
    positive_control = False
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
                    positive_control = True
        bound = _bound_names_for_file(tree, module_dotted, _is_package(path))
        call_count = _count_calls(tree, bound)
        if call_count:
            rel = path.relative_to(_REPO_ROOT).as_posix()
            counts[rel] = call_count
    return counts, positive_control


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


def test_caller_audit_is_committed_and_covers_the_live_tree() -> None:
    """The audit file exists, is internally consistent, and audits >= live.

    Fails loud (never skip/xfail) on a missing audit file, an empty or
    misrooted live scan, or a missing positive control.
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

    assert _SRC_ROOT.is_dir(), f"scan root is not a directory: {_SRC_ROOT}"
    live_counts, positive_control = _scan_live_call_counts(_SRC_ROOT)
    assert positive_control, (
        f"scan positive control failed: no {_NEW_NAME} definition found under {_SRC_ROOT} "
        "-- the scan is misrooted or the rename has not landed"
    )
    assert live_counts, f"live scan of {_SRC_ROOT} found zero call sites -- misrooted scan"

    audited_counts: Counter[str] = Counter()
    for row in rows:
        audited_counts[row["path"]] += 1

    for path, live_count in live_counts.items():
        assert path in audited_counts, (
            f"{path} calls {_NEW_NAME} but has no row in the audit "
            f"({_AUDIT_PATH.relative_to(_REPO_ROOT)})"
        )
        assert live_count <= audited_counts[path], (
            f"{path}: live call count {live_count} exceeds audited count "
            f"{audited_counts[path]} -- the audit undercounts this file"
        )
