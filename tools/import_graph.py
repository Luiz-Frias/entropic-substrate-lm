#!/usr/bin/env python3
"""Measures the internal (entropic_substrate.*) AST import graph of a source tree.

What this measures: for every ``*.py`` file under ``--root``, every top-level and
nested ``import``/``from ... import`` statement whose resolved target starts with
the package name, recorded as one (src module, dst module) edge. Relative imports
are resolved using the importing module's own dotted position. A ``from X import a``
edge is normalised to the submodule ``X.a`` when that submodule exists on disk,
else to the package ``X`` itself (so importing a name, not a submodule, attributes
the edge to the defining package). Self-edges are dropped.

This is promoted from the 2026-09-17 refactor survey unchanged in
algorithm from the original ``import_graph.py``; only the CLI (argparse, no
hardcoded paths) and the loud-failure behaviour on an unreadable/empty tree are
new. It is one of the plan's measurement-oracle tools (docs/plans/src-hexagonal-refactor.plan.md,
deliverable 7): every phase gate cites its output, so a change to the algorithm
here is a change to what "green" means for every phase and must be called out,
never silently drifted.

Exit codes: 0 success. 1 unreadable root / a *.py file that fails to parse / zero
modules discovered (a root that measures nothing is not a passing measurement,
it is a broken one -- house rule "measurement code fails LOUD").
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def modname(p: Path, root: Path, pkg: str) -> str:
    rel = p.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_of(mod: str) -> str:
    parts = mod.split(".")
    return parts[1] if len(parts) > 2 else "ROOT"


def build_graph(root: Path, pkg: str) -> tuple[dict[str, Path], list[tuple[str, str]]]:
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    if not files:
        raise SystemExit(
            f"FATAL: no *.py files found under {root} -- refusing to report an empty graph as success"
        )

    mods: dict[str, Path] = {}
    for p in files:
        mods[modname(p, root, pkg)] = p

    edges: list[tuple[str, str]] = []
    for mod, p in mods.items():
        try:
            source = p.read_text()
            tree = ast.parse(source, filename=str(p))
        except (OSError, SyntaxError) as exc:
            raise SystemExit(f"FATAL: cannot parse {p}: {exc}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith(pkg):
                        edges.append((mod, a.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = mod.split(".")
                    if p.name != "__init__.py":
                        base = base[:-1]
                    base = base[: len(base) - (node.level - 1)]
                    target = ".".join(base + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                if not target.startswith(pkg):
                    continue
                for a in node.names:
                    cand = f"{target}.{a.name}"
                    edges.append((mod, cand if cand in mods else target))

    def norm(d: str) -> str:
        while d not in mods and "." in d:
            d = d.rsplit(".", 1)[0]
        return d

    edges = sorted({(s, norm(d)) for s, d in edges if norm(d) != s})
    return mods, edges


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "root", type=Path, help="path to the package directory, e.g. src/entropic_substrate"
    )
    ap.add_argument("out_dir", type=Path, help="directory to write edges.json + summary.json into")
    ap.add_argument(
        "--package",
        default="entropic_substrate",
        help="top-level package name (default: entropic_substrate)",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"FATAL: root {root} is not a directory", file=sys.stderr)
        return 1

    try:
        mods, edges = build_graph(root, args.package)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fan_out = Counter(s for s, _ in edges)
    fan_in = Counter(d for _, d in edges)
    pkg_edges = Counter((package_of(s), package_of(d)) for s, d in edges)
    pkg_matrix: dict[str, dict[str, int]] = defaultdict(dict)
    for (a, b), n in sorted(pkg_edges.items()):
        pkg_matrix[a][b] = n
    cycles = sorted(
        {tuple(sorted((a, b))) for (a, b) in pkg_edges if a != b and (b, a) in pkg_edges}
    )

    (args.out_dir / "edges.json").write_text(
        json.dumps(
            {"modules": len(mods), "edges": [{"src": s, "dst": d} for s, d in edges]}, indent=1
        )
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "package_matrix": pkg_matrix,
                "package_cycles": cycles,
                "fan_in_top": fan_in.most_common(40),
                "fan_out_top": fan_out.most_common(40),
                "lines": {m: sum(1 for _ in open(p)) for m, p in mods.items()},
            },
            indent=1,
        )
    )
    print(f"modules={len(mods)} edges={len(edges)}")
    print("package matrix (src -> dst: count):")
    for a in sorted(pkg_matrix):
        print(
            f"  {a:20s} "
            + "  ".join(f"{b}:{n}" for b, n in sorted(pkg_matrix[a].items(), key=lambda kv: -kv[1]))
        )
    print("package-level cycles:", cycles)
    print("top fan-in (most depended-on modules):")
    for m, n in fan_in.most_common(20):
        print(f"  {n:4d}  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
