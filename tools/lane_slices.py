"""Derive per-lane file SLICES for a fan-out wave from the LANDED import graph, and fail
loud on any overlap (operator ruling D-WAVES r2, 2026-09-22).

A lane's slice is: the legacy modules it moves (its targets) ∪ every LANDED module that
imports one of those targets by its explicit ``entropic_substrate.__legacy.<x>`` address
(rewrite-on-arrival: the moving phase rewrites those importers, nobody else does) ∪ the typed
shim pair at each target's old path. Two lanes of one wave must be pairwise DISJOINT on
files; a shared file is a same-file conflict at the zip by construction (R-parallel).

Files declared ``zip_owned_regenerable`` (the checkpoint regime-row fixtures and
``.secrets.baseline``) are excluded from the overlap check: every lane that rewrites a
member of the checkpoint reader set re-pins the same rows, and the zip regenerates them on
the integrated tip in one rung (see docs/plans/src-hexagonal-refactor.plan.md, D-WAVES r2).

Input is the tree on disk (read-only) plus ``tools/mapping.json``; lanes are given as
``lane_id=dotted.target,dotted.target,...`` or derived from a fragment file. Exit code 1 on
overlap, 2 on a malformed lane. Nothing here writes.

    uv run python tools/lane_slices.py src tools/mapping.json \
        p6-core=entropic_substrate.analysis.predictive_probe,entropic_substrate.analysis.predictive_score_gain \
        p7-register-gate=@.claude/workflows/landed-set/p7-register-gate.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

LANDED_LAYERS: tuple[str, ...] = ("kernel", "core", "ladder", "historical", "composition")
_LEGACY_IMPORT = re.compile(
    r"^\s*(?:from\s+entropic_substrate\.__legacy\.([\w.]+)\s+import\b"
    r"|import\s+entropic_substrate\.__legacy\.([\w.]+))"
)


class LaneSlicesError(Exception):
    """Malformed lane spec or unreadable input; never a silent empty slice."""


def landed_legacy_edges(src: Path) -> dict[str, set[str]]:
    """target (old dotted name, e.g. ``entropic_substrate.paths``) -> set of landed
    importer files (paths relative to ``src``). Only explicit ``__legacy`` addresses count;
    old-prefix imports are forbidden in landed packages by tests/test_no_old_prefix.py."""
    pkg = src / "entropic_substrate"
    edges: dict[str, set[str]] = defaultdict(set)
    for layer in LANDED_LAYERS:
        for path in sorted((pkg / layer).rglob("*.py")):
            for line in path.read_text(encoding="utf-8").splitlines():
                m = _LEGACY_IMPORT.match(line)
                if m:
                    edges["entropic_substrate." + (m.group(1) or m.group(2))].add(
                        str(path.relative_to(src))
                    )
    return edges


def shim_pair(target: str) -> tuple[str, str]:
    rel = target.replace(".", "/")
    return f"{rel}.py", f"{rel}.pyi"


def legacy_source(target: str) -> str:
    rel = target.split("entropic_substrate.", 1)[1].replace(".", "/")
    return f"entropic_substrate/__legacy/{rel}.py"


def slice_for(targets: set[str], edges: dict[str, set[str]]) -> set[str]:
    files: set[str] = set()
    for t in targets:
        files.add(legacy_source(t))
        files.update(shim_pair(t))
        files.update(edges.get(t, ()))
    return files


def overlaps(
    slices: dict[str, set[str]], regenerable: set[str]
) -> list[tuple[str, str, list[str]]]:
    """Every pair of lanes sharing at least one non-regenerable file, with the files."""
    out: list[tuple[str, str, list[str]]] = []
    ids = sorted(slices)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            shared = sorted((slices[a] & slices[b]) - regenerable)
            if shared:
                out.append((a, b, shared))
    return out


def parse_lane(spec: str, repo: Path) -> tuple[str, set[str]]:
    if "=" not in spec:
        raise LaneSlicesError(f"lane spec must be <id>=<targets>, got {spec!r}")
    lane, rhs = spec.split("=", 1)
    if rhs.startswith("@"):
        frag = json.loads((repo / rhs[1:]).read_text(encoding="utf-8"))
        modules = frag.get("modules")
        if not isinstance(modules, dict) or not modules:
            raise LaneSlicesError(f"fragment {rhs[1:]} has no 'modules' object")
        targets = {k for k, v in modules.items() if isinstance(v, dict) and v.get("landed") is True}
    else:
        targets = {t for t in rhs.split(",") if t}
    if not targets:
        raise LaneSlicesError(f"lane {lane!r} has no targets")
    bad = [t for t in targets if not t.startswith("entropic_substrate.")]
    if bad:
        raise LaneSlicesError(f"lane {lane!r}: targets must be old dotted names: {bad}")
    return lane, targets


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src", type=Path)
    ap.add_argument("mapping", type=Path, help="tools/mapping.json (targets must be mapped rows)")
    ap.add_argument(
        "lanes", nargs="+", help="<lane-id>=<comma targets> or <lane-id>=@<fragment.json>"
    )
    ap.add_argument(
        "--regenerable",
        default="",
        help="comma-separated repo-relative files excluded from the overlap check",
    )
    ap.add_argument("--json", action="store_true", help="print the slices and overlaps as JSON")
    args = ap.parse_args(argv)
    repo = args.src.resolve().parent
    try:
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))["modules"]
        lanes = dict(parse_lane(s, repo) for s in args.lanes)
    except (OSError, KeyError, ValueError, LaneSlicesError) as exc:
        print(f"lane_slices: {exc}", file=sys.stderr)
        return 2
    unmapped = sorted(t for ts in lanes.values() for t in ts if t not in mapping)
    if unmapped:
        print(f"lane_slices: targets not in mapping.json: {unmapped}", file=sys.stderr)
        return 2
    edges = landed_legacy_edges(args.src)
    slices = {lane: slice_for(ts, edges) for lane, ts in lanes.items()}
    regen = {f for f in args.regenerable.split(",") if f}
    shared = overlaps(slices, regen)
    if args.json:
        print(
            json.dumps(
                {
                    "slices": {k: sorted(v) for k, v in slices.items()},
                    "overlaps": [{"a": a, "b": b, "files": f} for a, b, f in shared],
                },
                indent=2,
            )
        )
    else:
        for lane, files in slices.items():
            print(f"{lane}: {len(files)} files ({len(lanes[lane])} targets)")
        for a, b, f in shared:
            print(f"OVERLAP {a} x {b}: {len(f)} files")
            for x in f:
                print(f"  {x}")
        if not shared:
            print("no overlap: the wave is file-disjoint")
    return 1 if shared else 0


if __name__ == "__main__":
    raise SystemExit(main())
