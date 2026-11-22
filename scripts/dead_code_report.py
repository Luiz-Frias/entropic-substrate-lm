#!/usr/bin/env python3
"""Warn about source modules that nothing reaches. Never fails.

    python scripts/dead_code_report.py

A module is ALIVE when it is reachable from a root by following imports and
dotted-name strings. Roots are every module with an ``if __name__ ==
"__main__"`` block, every module under an entrypoint directory, and every
module named in a file outside the package (the justfile, scripts, tools,
the image definition). Dotted names inside string literals count as edges
because this codebase launches workers by module name.

The output is a candidate list for a refactor, not a verdict: reachability
is per module, so a module imported for one constant counts as alive, and a
module reached only through a plugin table would be listed wrongly. It is
printed after `gt submit` by `just post-ship` and exits 0 on purpose.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_PACKAGE = Path("src/entropic_substrate")
DEFAULT_ROOT_FILES = (
    "justfile",
    "justfile.local",
    "Containerfile",
    "compose.yaml",
    "pyproject.toml",
)
DEFAULT_ROOT_DIRS = ("scripts", "tools", ".github")
ENTRYPOINT_DIRS = ("composition",)


def _modules(package: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package.parent).with_suffix("")
        parts = relative.parts[:-1] if path.name == "__init__.py" else relative.parts
        found[".".join(parts)] = path
    return found


def _resolve(name: str, modules: dict[str, Path]) -> str | None:
    while name:
        if name in modules:
            return name
        name = name.rpartition(".")[0]
    return None


def _has_main_block(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.If) and "__name__" in ast.unparse(node.test) for node in tree.body
    )


def _edges(
    name: str, path: Path, tree: ast.Module, modules: dict[str, Path], dotted: re.Pattern[str]
) -> set[str]:
    candidates: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = name.split(".")
                parts = parts if path.name == "__init__.py" else parts[:-1]
                parts = parts[: len(parts) - (node.level - 1)]
                base = ".".join([*parts, *([node.module] if node.module else [])])
            candidates += [base, *[f"{base}.{alias.name}" for alias in node.names]]
        elif isinstance(node, ast.Import):
            candidates += [alias.name for alias in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            candidates += dotted.findall(node.value)
    resolved = {_resolve(candidate, modules) for candidate in candidates}
    return {target for target in resolved if target and target != name}


def unreached(package: Path, *, extra_roots: Iterable[Path]) -> list[str]:
    """Dotted names of non-package modules no root reaches, sorted."""
    modules = _modules(package)
    top = package.name
    dotted = re.compile(rf"(?<![\w.]){re.escape(top)}(?:\.\w+)+")
    trees = {name: ast.parse(path.read_text(encoding="utf-8")) for name, path in modules.items()}
    edges = {
        name: _edges(name, modules[name], tree, modules, dotted) for name, tree in trees.items()
    }
    for name in modules:  # importing a module imports its parent packages
        parent = name.rpartition(".")[0]
        if parent in modules:
            edges[name].add(parent)

    roots = {name for name, tree in trees.items() if _has_main_block(tree)}
    roots |= {
        name for name in modules if any(f".{part}." in f".{name}." for part in ENTRYPOINT_DIRS)
    }
    for outside in extra_roots:
        files = (
            [outside] if outside.is_file() else sorted(p for p in outside.rglob("*") if p.is_file())
        )
        for file in files:
            try:
                text = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            roots |= {
                target for match in dotted.findall(text) if (target := _resolve(match, modules))
            }

    seen: set[str] = set()
    stack = list(roots)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(edges.get(current, ()))
    return sorted(
        name for name, path in modules.items() if name not in seen and path.name != "__init__.py"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args(argv)
    package: Path = args.package
    base = package.parent.parent if package == DEFAULT_PACKAGE else package.parent
    outside = [
        base / name for name in (*DEFAULT_ROOT_FILES, *DEFAULT_ROOT_DIRS) if (base / name).exists()
    ]
    try:
        dead = unreached(package, extra_roots=outside)
    except (OSError, SyntaxError, ValueError) as error:
        # Warning-only by contract: a file this report cannot read or parse
        # must not turn the post-ship step red.
        print(f"dead-code: WARNING report incomplete, inspection failed: {error}")
        return 0
    if not dead:
        print("dead-code: every source module is reachable from an entrypoint")
        return 0
    print(
        f"dead-code: WARNING {len(dead)} source module(s) no entrypoint reaches (candidates, not verdicts):"
    )
    for name in dead:
        print(f"    {(package.parent / Path(*name.split('.'))).as_posix()}.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
