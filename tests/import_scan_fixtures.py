"""One import scanner for the structural import-guard tests.

PR #376 review (CodeRabbit 4369372647): the strict-sink and observer guards each
carried their own AST scanner, and both read ``ImportFrom.module`` without its
``level``, so ``from ...run import checkpoint`` inside
``entropic_substrate.ladder.capture.adapters`` was recorded as ``run`` and slipped
past a ``.ladder.run`` predicate (measured: the guard stayed green). Relative
imports are resolved here against the module's own package, the way Python
resolves them, so every recorded name is absolute.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import ModuleType


def imported_names(module: ModuleType) -> set[str]:
    """Absolute names ``module`` imports at ANY scope, function bodies included.

    For ``from pkg import name`` both ``pkg`` and ``pkg.name`` are recorded, because
    the statement may bind the submodule ``pkg.name`` while naming only ``pkg``.
    """
    package = module.__package__ or ""
    tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            relative = "." * node.level + (node.module or "")
            base = importlib.util.resolve_name(relative, package) if node.level else relative
            imported.add(base)
            imported.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return imported
