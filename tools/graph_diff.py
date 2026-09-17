#!/usr/bin/env python3
"""Compares two import_graph.py edges.json snapshots and prints modules and edges
added/removed between them. This is the "verify graph at init" gate the plan
requires before every wave (docs/plans/src-hexagonal-refactor.plan.md preconditions: "graph
re-derived at verifiedAt ... modules/edges restated, diff vs 7ba2788 printed, any
new module mapped") -- run it between the epoch a mapping was built against and
the tree about to be operated on, so a moved epoch is caught rather than silently
reinterpreted (axiom: Epoch, "a moved epoch voids -- restate, never reinterpret").

Module set: edges.json only carries a module COUNT, not a module list, so the
module set here is derived from edge endpoints (src/dst), exactly as
remap_check.py and blast_radius.py do. A module with zero import edges in and out
is invisible to this diff -- it cannot appear in "modules added/removed" even if
the file was added or deleted between the two snapshots. This is a documented
limitation of the edges.json format, not a bug in this tool: out/proposals/B's
build_mapping.py hit the same gap and worked around it by reading summary.json's
line-count table instead, which lists every file regardless of edges. Callers who
need edge-free modules covered should diff summary.json's "lines" keys as well.

Delta definition: |edges added| + |edges removed| (module adds/removes are
reported but do not enter the delta -- an edge is the unit of coupling this
whole toolchain measures, so the gate is edge-shaped).

Exit codes: 0 delta <= --max-delta. 1 unreadable input. 2 delta > --max-delta
(the diff is fully printed either way; 2 means "STOP and look", not "computation
failed").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_edges(path: Path) -> tuple[set[tuple[str, str]], set[str]]:
    try:
        doc = json.loads(path.read_text())
    except OSError as exc:
        print(f"FATAL: cannot read {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except json.JSONDecodeError as exc:
        print(f"FATAL: {path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    try:
        raw_edges = doc["edges"]
    except KeyError:
        print(f"FATAL: {path} has no 'edges' key", file=sys.stderr)
        raise SystemExit(1) from None
    edges = {(e["src"], e["dst"]) for e in raw_edges}
    modules = {m for e in edges for m in e}
    return edges, modules


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("before_edges_json", type=Path)
    ap.add_argument("after_edges_json", type=Path)
    ap.add_argument(
        "--max-delta",
        type=int,
        required=True,
        help="exit 2 if |edges added| + |edges removed| exceeds this",
    )
    args = ap.parse_args()

    before_edges, before_mods = load_edges(args.before_edges_json)
    after_edges, after_mods = load_edges(args.after_edges_json)

    mods_added = sorted(after_mods - before_mods)
    mods_removed = sorted(before_mods - after_mods)
    edges_added = sorted(after_edges - before_edges)
    edges_removed = sorted(before_edges - after_edges)

    print(
        f"before: {len(before_mods)} modules (edge-endpoint-visible), {len(before_edges)} edges  ({args.before_edges_json})"
    )
    print(
        f"after:  {len(after_mods)} modules (edge-endpoint-visible), {len(after_edges)} edges  ({args.after_edges_json})"
    )
    print(f"modules added ({len(mods_added)}):")
    for m in mods_added:
        print(f"  + {m}")
    print(f"modules removed ({len(mods_removed)}):")
    for m in mods_removed:
        print(f"  - {m}")
    print(f"edges added ({len(edges_added)}):")
    for s, d in edges_added:
        print(f"  + {s} -> {d}")
    print(f"edges removed ({len(edges_removed)}):")
    for s, d in edges_removed:
        print(f"  - {s} -> {d}")

    delta = len(edges_added) + len(edges_removed)
    print(f"edge delta: {delta} (max allowed: {args.max_delta})")
    if delta > args.max_delta:
        print(f"FATAL: edge delta {delta} exceeds --max-delta {args.max_delta}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
