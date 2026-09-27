#!/usr/bin/env python3
"""Measures, per destination package in a mapping, how many modules move there and
how many OTHER modules import at least one of them -- those "external importers"
are the files whose import lines a phase that lands that package must rewrite
under the rewrite-on-arrival rule (docs/plans/src-hexagonal-refactor.plan.md: "a file's import
lines are rewritten exactly once, by the phase that moves it").

Promoted from the 2026-09-17 refactor survey; the original lived at
out/proposals/C/blast_radius.py with both inputs hardcoded relative to the
script's own location (``HERE.parents[2] / "graph" / "edges.json"`` and
``HERE / "mapping.json"``). Both are now required positional CLI arguments;
nothing here assumes the survey's directory layout.

Exit codes: 0 success. 1 unreadable input, or a mapping with an endpoint edges.json
references but the mapping does not define (the same fail-loud contract as
remap_check.py -- a blast radius computed over a partially-mapped graph
understates the true radius and must not be reported as if it were complete).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_json(path: Path, label: str) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError as exc:
        print(f"FATAL: cannot read {label} at {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except json.JSONDecodeError as exc:
        print(f"FATAL: {label} at {path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("edges_json", type=Path, help="edges.json from import_graph.py")
    ap.add_argument(
        "mapping_json", type=Path, help='{"layers": [...], "modules": {mod: {"dest":..., ...}}}'
    )
    args = ap.parse_args()

    edges_doc = load_json(args.edges_json, "edges.json")
    mapping_doc = load_json(args.mapping_json, "mapping")

    try:
        edges = edges_doc["edges"]
    except KeyError:
        print(f"FATAL: {args.edges_json} has no 'edges' key", file=sys.stderr)
        return 1
    try:
        mods = mapping_doc["modules"]
    except KeyError as exc:
        print(f"FATAL: {args.mapping_json} is missing required key {exc}", file=sys.stderr)
        return 1

    all_mods = {e[k] for e in edges for k in ("src", "dst")}
    missing = sorted(all_mods - set(mods))
    if missing:
        print(
            f"FATAL: {len(missing)} UNMAPPED module(s) referenced by edges.json but absent from the mapping:",
            file=sys.stderr,
        )
        for m in missing[:30]:
            print(f"    {m}", file=sys.stderr)
        print(
            "Blast radius computed over a partially-mapped graph understates the true radius; refusing to report it.",
            file=sys.stderr,
        )
        return 1

    inn: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        inn[e["dst"]].add(e["src"])

    by: dict[str, set[str]] = defaultdict(set)
    for m, v in mods.items():
        by[v["dest"]].add(m)

    rows = []
    for dest, dest_mods in by.items():
        importers: set[str] = set()
        for m in dest_mods:
            importers |= inn[m]
        rows.append((dest, len(dest_mods), len(importers - dest_mods), len(importers)))

    print(
        f"{'destination package':44s} {'moved':>6s} {'ext importers':>13s} {'all importers':>13s}"
    )
    for dest, n, ext, tot in sorted(rows, key=lambda r: -r[2]):
        print(f"{dest:44s} {n:6d} {ext:13d} {tot:13d}")
    print(f"\ntotal modules {len(mods)}; total edges {len(edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
