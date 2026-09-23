"""Derive per-lane file SLICES for a fan-out wave from the LANDED import graph, and fail
loud on any overlap or any import dependency between lanes (operator ruling D-WAVES r2,
2026-09-22; dependency check added after the r3 measurement of 2026-09-23).

A lane's slice is: the legacy modules it moves (its targets) ∪ every LANDED module that
imports one of those targets by its explicit ``entropic_substrate.__legacy.<x>`` address
(rewrite-on-arrival: the moving phase rewrites those importers, nobody else does) ∪ the typed
shim pair at each target's old path. Two lanes of one wave must be pairwise DISJOINT on
files; a shared file is a same-file conflict at the zip by construction (R-parallel).

Files declared ``zip_owned_regenerable`` (the checkpoint regime-row fixtures and
``.secrets.baseline``) are excluded from the overlap check: every lane that rewrites a
member of the checkpoint reader set re-pins the same rows, and the zip regenerates them on
the integrated tip in one rung (see docs/plans/src-hexagonal-refactor.plan.md, D-WAVES r2).

DEPENDENCIES (D-WAVES r3, MEASURED 2026-09-23: w2's p9 and p11 each import 50 of p8's
targets from their own moved modules): a lane's moved module that imports a sibling lane's
target has NO valid address in the lane's own worktree (the old dotted name is forbidden in
landed packages, the explicit ``__legacy`` address is deleted by the sibling, the sibling's
new address does not exist until the zip), so file-disjoint lanes can still be illegal. The
check reads each target's legacy source, resolves every ``entropic_substrate.*`` import to
its mapping row (a facade package counts as every lane that moves one of its leaves) and
reports ``DEPENDS a -> b: n``; any dependency exits 1 unless ``--allow-dependencies``.

PACKAGE KEYS (``entropic_substrate.<pkg>`` rows whose legacy source is a package directory,
not a leaf ``.py``) are never lane targets: tools/gen_shims.py fails loud on a fragment row
naming one, and their moved facade ``__init__.py`` is handled by the p3 emptied-facade
ruling. Naming one in a lane spec is a malformed lane (exit 2).

Input is the tree on disk (read-only) plus ``tools/mapping.json``; lanes are given as
``lane_id=dotted.target,dotted.target,...`` or derived from a fragment file. Exit code 1 on
overlap or dependency, 2 on a malformed lane. Nothing here writes.

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
_ANY_IMPORT = re.compile(
    r"^\s*(?:from\s+(entropic_substrate\.[\w.]+)\s+import\b|import\s+(entropic_substrate\.[\w.]+))",
    re.M,
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


def _rel(target: str) -> str:
    return target.split("entropic_substrate.", 1)[1].replace(".", "/")


def shim_pair(target: str) -> tuple[str, str]:
    rel = target.replace(".", "/")
    return f"{rel}.py", f"{rel}.pyi"


def legacy_source(target: str) -> str:
    return f"entropic_substrate/__legacy/{_rel(target)}.py"


def is_package_key(src: Path, target: str) -> bool:
    """A mapping row whose legacy source is a package directory, never a leaf module."""
    return (
        not (src / legacy_source(target)).is_file()
        and (src / "entropic_substrate" / "__legacy" / _rel(target)).is_dir()
    )


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


def _resolve_import(name: str, mapping: dict) -> str:
    """Longest mapping-row prefix of an imported dotted name (``__legacy.`` stripped)."""
    name = name.replace(".__legacy.", ".")
    while name and name not in mapping and "." in name:
        name = name.rsplit(".", 1)[0]
    return name if name in mapping else ""


def dependencies(
    src: Path, lanes: dict[str, set[str]], mapping: dict
) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """``(a, b, [(moved module of a, target of b), ...])`` for every ordered lane pair where
    a moved module of ``a`` imports a target of ``b``, directly or through ``b``'s facade
    package (an import of ``entropic_substrate.<pkg>`` depends on every lane moving a leaf
    under ``<pkg>``)."""
    owner_of_leaf: dict[str, str] = {t: lane for lane, ts in lanes.items() for t in ts}
    owners_of_pkg: dict[str, set[str]] = defaultdict(set)
    for t, lane in owner_of_leaf.items():
        parts = t.split(".")
        for i in range(2, len(parts)):
            owners_of_pkg[".".join(parts[:i])].add(lane)
    out: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for lane, targets in lanes.items():
        for t in sorted(targets):
            path = src / legacy_source(t)
            if not path.is_file():
                continue
            for m in _ANY_IMPORT.finditer(path.read_text(encoding="utf-8")):
                row = _resolve_import(m.group(1) or m.group(2), mapping)
                if not row:
                    continue
                owners = (
                    {owner_of_leaf[row]} if row in owner_of_leaf else owners_of_pkg.get(row, set())
                )
                for other in owners:
                    if other != lane:
                        out[(lane, other)].add((t, row))
    return [(a, b, sorted(v)) for (a, b), v in sorted(out.items())]


def parse_lane(spec: str, repo: Path, src: Path) -> tuple[str, set[str]]:
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
    pkgs = sorted(t for t in targets if is_package_key(src, t))
    if pkgs:
        raise LaneSlicesError(
            f"lane {lane!r}: package keys are never lane targets (gen_shims fails loud on "
            f"them; their facade __init__.py follows the emptied-facade ruling): {pkgs}"
        )
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
    ap.add_argument(
        "--allow-dependencies",
        action="store_true",
        help="report import dependencies between lanes without failing on them",
    )
    ap.add_argument("--json", action="store_true", help="print the slices and overlaps as JSON")
    args = ap.parse_args(argv)
    repo = args.src.resolve().parent
    try:
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))["modules"]
        lanes = dict(parse_lane(s, repo, args.src) for s in args.lanes)
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
    deps = dependencies(args.src, lanes, mapping)
    failing = bool(shared) or (bool(deps) and not args.allow_dependencies)
    if args.json:
        print(
            json.dumps(
                {
                    "slices": {k: sorted(v) for k, v in slices.items()},
                    "overlaps": [{"a": a, "b": b, "files": f} for a, b, f in shared],
                    "dependencies": [
                        {"a": a, "b": b, "edges": [list(e) for e in es]} for a, b, es in deps
                    ],
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
        for a, b, es in deps:
            print(f"DEPENDS {a} -> {b}: {len(es)} import(s) of the sibling's targets")
            for mod, tgt in es[:10]:
                print(f"  {mod} -> {tgt}")
        if not shared and not deps:
            print("no overlap, no dependency: the wave is legal")
        elif not shared:
            print("no shared files, but lanes depend on each other's targets")
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
