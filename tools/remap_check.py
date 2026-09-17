#!/usr/bin/env python3
"""Measures import-layer inversions and package-level cycles for a proposed
module -> {dest, layer, action} mapping, against a measured edges.json (see
import_graph.py).

An "inversion" is an edge whose source layer has a strictly LOWER rank than its
destination's layer, in the mapping's own declared ``layers`` order (a
"foundational" package importing from a package the mapping ranks above it). A
"cycle" is a pair of destinations with edges running both directions between
them, counted once per unordered pair. An edge with ``action: "delete"`` on
either endpoint is excluded from both counts (a module marked for deletion
cannot inform a layering verdict about the tree that remains).

Promoted from the 2026-09-17 refactor survey. The survey's
original printed "UNMAPPED modules (...)" and "UNKNOWN layer on ..." as
warnings and then KEPT GOING, silently dropping the offending edges from the
count -- so a mapping with a hole in it could still print a clean-looking
"inversions: N" and look like a passing measurement. That is exactly the
"reporting success by testing nothing" failure class this repo's house rules
name explicitly (CLAUDE.md house rule 1, re: skip-vs-fail). This promoted copy
never does that: an unmapped module or an unknown layer is now a hard FATAL
that exits 1 before any count is printed, per the plan's requirement that
"unmapped modules / unknown layers / unreadable input" never look like a
passing measurement.

Exit codes: 0 the counts were computed over every edge with both endpoints
mapped to a known layer. 1 unreadable input, an unmapped module, or a module
mapped to a layer absent from the mapping's own ``layers`` list.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
        "mapping_json",
        type=Path,
        help='{"layers": [...], "modules": {mod: {"dest":..., "layer":..., "action":...}}}',
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
        layers = mapping_doc["layers"]
        mods = mapping_doc["modules"]
    except KeyError as exc:
        print(f"FATAL: {args.mapping_json} is missing required key {exc}", file=sys.stderr)
        return 1

    rank = {layer: i for i, layer in enumerate(layers)}

    missing = sorted({e[k] for e in edges for k in ("src", "dst")} - set(mods))
    if missing:
        print(
            f"FATAL: {len(missing)} UNMAPPED module(s) referenced by edges.json but absent from the mapping:",
            file=sys.stderr,
        )
        for m in missing[:30]:
            print(f"    {m}", file=sys.stderr)
        if len(missing) > 30:
            print(f"    ... and {len(missing) - 30} more", file=sys.stderr)
        print(
            "A mapping with unmapped endpoints cannot certify a layering count; add them or mark them deleted.",
            file=sys.stderr,
        )
        return 1

    bad_layer = sorted((mo, v.get("layer")) for mo, v in mods.items() if v.get("layer") not in rank)
    if bad_layer:
        print(
            f"FATAL: {len(bad_layer)} module(s) mapped to a layer absent from mapping['layers']={layers}:",
            file=sys.stderr,
        )
        for mo, layer in bad_layer[:30]:
            print(f"    {mo} -> {layer!r}", file=sys.stderr)
        return 1

    inv: Counter[tuple[str, str]] = Counter()
    pkg: Counter[tuple[str, str]] = Counter()
    kept = 0
    for e in edges:
        s, d = mods[e["src"]], mods[e["dst"]]
        if s.get("action") == "delete" or d.get("action") == "delete":
            continue
        kept += 1
        if rank[s["layer"]] < rank[d["layer"]]:
            inv[(s["layer"], d["layer"])] += 1
        if s["dest"] != d["dest"]:
            pkg[(s["dest"], d["dest"])] += 1

    cycles = sorted({tuple(sorted(k)) for k in pkg if (k[1], k[0]) in pkg})

    print(f"edges considered: {kept} of {len(edges)}")
    print(f"layer inversions (lower layer importing higher): {sum(inv.values())}")
    for (a, b), n in inv.most_common():
        print(f"  {n:4d}  {a} -> {b}")
    print(f"package-level cycles: {len(cycles)}")
    for a, b in cycles:
        print(f"  {a} <-> {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
