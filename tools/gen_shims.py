#!/usr/bin/env python3
"""Generates/re-points one alias shim per module named across a directory of
disjoint per-phase "landed-set" fragments, against the authoritative module
list in ``mapping.json``.

Binding source: ``docs/plans/src-hexagonal-refactor.plan.md``, section "The mechanism rule that makes phases disjoint" (ruling DN-P1, 2026-09-17): no import-linter ignore list; each phase commits a disjoint fragment ``.claude/workflows/landed-set/<id>.json`` and this generator merges the fragments to emit shims.

Fragment schema (JSON), one file per phase, filename ``<phase-id>.json``
under a fragments directory (e.g. ``.claude/workflows/landed-set/``):

    {"phase": "<phase-id>", "modules": {"<old dotted>": {"target": "<new
    dotted, or entropic_substrate.__legacy.<x> while quarantined>", "landed":
    bool, "retired": bool (optional), "package": bool (optional)}},
    "legacy_edge_budget": <int>}

``mapping.json`` (the repo's authoritative old-module list, currently 334
entries, schema ``{"layers": [...], "modules": {"<old dotted>": {"dest":
..., "layer": ..., "action": ..., "note": ...}}}``) is the validation
source: every module name that appears in ANY fragment MUST be a key in
``mapping.json`` -- a fragment naming a module mapping.json does not know
about is a fail-loud error, not a module silently skipped. This is the
resolved reading of "fragments reference it by module name only": the
fragment is still the source of the literal shim ``target`` string (per the
schema above, verbatim as specified), but the module NAME it uses must be
drawn from the one authoritative list, so two fragments (or a fragment and a
typo) can never invent two different spellings of "the same" module.
Whether ``target`` should instead be DERIVED from ``mapping.json``'s
``dest`` field rather than restated per fragment is an open question this
generator does NOT resolve on its own initiative: ``mapping.json``'s
``dest`` is package-level only (e.g. ``entropic_substrate.ladder.analyze``,
not the fully-qualified ``entropic_substrate.ladder.analyze.charts``), so
deriving the exact final leaf module name would require inventing a
concatenation rule with no primary source pinning it. Restating ``target``
per fragment, verbatim as the schema specifies, is the choice that adds no
unverified mechanism.

Per module (semantics unchanged from the prior, already-corrected ruling --
see the retraction note in this generator's git history for the FIRST wrong
draft, which said `landed:true` deletes the shim):

* **A module keeps its shim for as long as it is named in some fragment**,
  regardless of ``landed`` -- deleting a shim the moment a module lands would
  break every test/importer not yet rewritten onto the new dotted path.
* ``landed: false`` -- module still physically quarantined under
  ``entropic_substrate.__legacy``; shim points at that legacy target.
* ``landed: true`` -- module has been ``git mv``'d to its structural home;
  shim is RE-POINTED at that new home.
* ``retired: true`` (p13-only; this generator's own explicit third state, not
  in any plan source -- "absent from every fragment" is not distinguishable
  from "not yet listed", so p13 needs a positive, auditable way to say "this
  old name is done, delete its shim") -- no shim emitted; overrides ``landed``.
* ``package: true`` (required whenever the old dotted name is a package, i.e.
  a dotted prefix of another module named across ANY fragment) -- shim is
  written as that package's ``__init__.py``. See the __path__ hazard note
  below the manifest-consistency section.

**Disjointness (fail loud, not a warning):** a module named in TWO OR MORE
fragments is a merge error -- exactly the "disjoint fragment" property the
plan's R-parallel rule depends on (two wave lanes both claiming the same
module would silently double-move it). ``--verify`` checks this without
generating anything.

**Manifest consistency (checked before any file is written, and by
``--verify``):** ``landed: true`` requires a target OUTSIDE
``entropic_substrate.__legacy``; ``landed: false`` requires a target INSIDE
it (the same invariant, both directions -- see ``find_landed_target_inconsistencies``).

**No import-linter ignore list is emitted anywhere by this generator** (DN-P1).
The per-phase ``legacy_edge_budget`` plus ``tests/test_legacy_coupling.py``
(owned by the phase, not this tool) are what replace it; this generator's
only budget-related duty is ``--print-budget``, which reports the
``legacy_edge_budget`` of the highest-ordered fragment PRESENT on disk, where
"ordering" is an explicit ``--sequence p0,p1,...`` list (or the ``sequence``
array read from a JSON file such as ``.claude/workflows/sprint-manifest.json``
via ``--sequence-file``). Every fragment present on disk must appear
somewhere in the given sequence, or the ordering is undefined and this FAILS
LOUD rather than guessing a position.

Shim form (fixed, house rule + plan requirement, docs/plans/src-hexagonal-refactor.plan.md
"Shim form (required)"):

    # Generated by gen_shims.py from <fragment paths> -- do not hand-edit.
    import <target> as _target
    import sys
    sys.modules[__name__] = _target

The ``import <target> as _target`` line MUST be a real static import (not a
string passed to ``importlib.import_module``), because import-linter's grimp
parses import statements structurally: a string-only shim makes every edge
routed through it invisible to the layers contract. ``--verify`` rejects any
shim whose body is not EXACTLY these three statements (comments aside).

The __path__ hazard (package shims only): once
``entropic_substrate.<pkg>``'s shim executes, ``sys.modules['entropic_substrate.<pkg>']``
is reassigned to the target package object. The import machinery, resolving a
later submodule import like ``entropic_substrate.<pkg>.child``, then fetches
the (now-aliased) parent from sys.modules and walks ITS ``__path__`` -- the
TARGET package's directory, not the old one -- so a physically separate shim
file for the child, still sitting in the OLD directory, is never consulted:
Python loads ``child.py`` fresh from the target directory under the OLD
dotted name, producing a SECOND module object backed by the same source as
``<target>.child``. This is exactly the failure the p0/p3 sentinel-singleton
test exists to catch; this generator documents the hazard, it does not
mitigate it.

Exit codes: 0 success. 1 unreadable fragment/mapping input, a module named in
more than one fragment, a module unknown to mapping.json, an unflagged
package prefix, a landed/target inconsistency, an out-of-sequence fragment
(``--print-budget``), or (``--verify``) a shim file not in the exact required
form.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

LEGACY_PREFIX = "entropic_substrate.__legacy"


class ManifestError(Exception):
    pass


def _load_json(path: Path, label: str) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError as exc:
        raise ManifestError(f"cannot read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{label} {path} is not valid JSON: {exc}") from exc


def load_mapping(path: Path) -> dict:
    doc = _load_json(path, "mapping")
    if "modules" not in doc:
        raise ManifestError(f"mapping {path} has no 'modules' key")
    return doc


def load_fragments(fragments_dir: Path) -> dict[str, dict]:
    """Returns {phase_id: fragment_dict} for every *.json directly under fragments_dir."""
    if not fragments_dir.is_dir():
        raise ManifestError(f"fragments directory {fragments_dir} does not exist")
    fragments: dict[str, dict] = {}
    for p in sorted(fragments_dir.glob("*.json")):
        doc = _load_json(p, "fragment")
        phase = doc.get("phase")
        if not phase:
            raise ManifestError(f"fragment {p} has no 'phase' field")
        if p.stem != phase:
            raise ManifestError(
                f"fragment {p} filename does not match its declared phase {phase!r}"
            )
        if "modules" not in doc:
            raise ManifestError(f"fragment {p} has no 'modules' key")
        fragments[phase] = doc
    if not fragments:
        raise ManifestError(f"no *.json fragments found under {fragments_dir}")
    return fragments


def merge_fragments(fragments: dict[str, dict], mapping: dict) -> dict[str, dict]:
    """Merges module entries across fragments. FAILS LOUD (never silently
    drops or overwrites) on: a module named in more than one fragment (the
    disjointness the plan's R-parallel rule requires), or a module name not
    present in mapping.json (the one authoritative module list)."""
    known = set(mapping["modules"])
    claimed_by: dict[str, str] = {}
    merged: dict[str, dict] = {}
    dupes: list[str] = []
    unknown: list[str] = []

    for phase in sorted(fragments):
        for mod, entry in fragments[phase].get("modules", {}).items():
            if mod not in known:
                unknown.append(f"{mod} (fragment {phase})")
                continue
            if mod in claimed_by:
                dupes.append(f"{mod} (fragments {claimed_by[mod]!r} and {phase!r})")
                continue
            claimed_by[mod] = phase
            merged[mod] = entry

    if unknown:
        raise ManifestError("module(s) not present in mapping.json: " + "; ".join(unknown))
    if dupes:
        raise ManifestError(
            "module(s) claimed by more than one fragment (disjointness violation): "
            + "; ".join(dupes)
        )
    return merged


def _is_legacy_target(target: str) -> bool:
    return target == LEGACY_PREFIX or target.startswith(LEGACY_PREFIX + ".")


def find_unflagged_packages(modules: dict) -> list[str]:
    """Dotted names that are a strict prefix of another key but lack package:true."""
    keys = sorted(modules)
    bad = []
    for old in keys:
        prefix = old + "."
        has_child = any(other.startswith(prefix) for other in keys if other != old)
        if has_child and not modules[old].get("package", False):
            bad.append(old)
    return bad


def find_landed_target_inconsistencies(modules: dict) -> list[str]:
    """landed:true + target still under __legacy, or landed:false + target
    NOT under __legacy -- both directions of the same invariant."""
    bad = []
    for old, entry in sorted(modules.items()):
        if entry.get("retired", False):
            continue
        landed = entry.get("landed", False)
        target = entry.get("target", "")
        is_legacy = _is_legacy_target(target)
        if landed and is_legacy:
            bad.append(f"{old}: landed=true but target {target!r} is still under {LEGACY_PREFIX}")
        elif not landed and not is_legacy:
            bad.append(f"{old}: landed=false but target {target!r} is not under {LEGACY_PREFIX}")
    return bad


def shim_source(old: str, target: str, source_label: str) -> str:
    return (
        f"# Generated by gen_shims.py from {source_label} -- do not hand-edit.\n"
        f"# Alias shim for {old} -> {target} (docs/plans/src-hexagonal-refactor.plan.md 'Shim form (required)').\n"
        f"import {target} as _target\n"
        f"import sys\n"
        f"sys.modules[__name__] = _target\n"
    )


def shim_output_path(out_dir: Path, old: str, is_package: bool) -> Path:
    parts = old.split(".")
    if is_package:
        return out_dir.joinpath(*parts, "__init__.py")
    *pkg_parts, leaf = parts
    return out_dir.joinpath(*pkg_parts, f"{leaf}.py")


def _check_merged(modules: dict) -> None:
    unflagged = find_unflagged_packages(modules)
    if unflagged:
        raise ManifestError(
            "the following module keys are a dotted prefix of another key "
            "(i.e. packages) but are not flagged 'package: true': " + ", ".join(unflagged)
        )
    inconsistent = find_landed_target_inconsistencies(modules)
    if inconsistent:
        raise ManifestError(
            "landed/target inconsistency (landed:true must target outside __legacy, "
            "landed:false must target inside __legacy): " + "; ".join(inconsistent)
        )


def generate(modules: dict, source_label: str, out_dir: Path) -> list[Path]:
    _check_merged(modules)
    written: list[Path] = []
    for old, entry in sorted(modules.items()):
        if entry.get("retired", False):
            continue  # p13 only: old name fully retired, no shim
        target = entry["target"]
        is_pkg = bool(entry.get("package", False))
        out_path = shim_output_path(out_dir, old, is_pkg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(shim_source(old, target, source_label))
        written.append(out_path)
    return written


def highest_ordered_fragment(fragments: dict[str, dict], sequence: list[str]) -> dict:
    present = set(fragments)
    out_of_sequence = present - set(sequence)
    if out_of_sequence:
        raise ManifestError(
            f"fragment(s) present on disk but absent from --sequence: {sorted(out_of_sequence)} "
            "-- ordering would be undefined, refusing to guess"
        )
    ordered_present = [p for p in sequence if p in present]
    if not ordered_present:
        raise ManifestError("no fragment on disk matches any phase in --sequence")
    return fragments[ordered_present[-1]]


# ---------------------------------------------------------------------------
# --verify: (1) fragment disjointness (via merge_fragments), (2) merged
# landed/target consistency, (3) AST-exact shim-form check on any
# already-generated *.py files.
# ---------------------------------------------------------------------------


def verify_shim_source(source: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"does not parse: {exc}"

    stmts = tree.body
    if len(stmts) != 3:
        return False, f"expected exactly 3 statements, found {len(stmts)}"

    s0, s1, s2 = stmts

    if not (isinstance(s0, ast.Import) and len(s0.names) == 1 and s0.names[0].asname == "_target"):
        return (
            False,
            "statement 1 must be `import <target> as _target` (a real static import, not from-import or importlib)",
        )

    if not (
        isinstance(s1, ast.Import)
        and len(s1.names) == 1
        and s1.names[0].name == "sys"
        and s1.names[0].asname is None
    ):
        return False, "statement 2 must be `import sys`"

    is_assign = isinstance(s2, ast.Assign)
    if is_assign:
        target_ok = (
            len(s2.targets) == 1
            and isinstance(s2.targets[0], ast.Subscript)
            and isinstance(s2.targets[0].value, ast.Attribute)
            and s2.targets[0].value.attr == "modules"
            and isinstance(s2.targets[0].value.value, ast.Name)
            and s2.targets[0].value.value.id == "sys"
        )
        value_ok = isinstance(s2.value, ast.Name) and s2.value.id == "_target"
    else:
        target_ok = value_ok = False
    if not (is_assign and target_ok and value_ok):
        return False, "statement 3 must be `sys.modules[__name__] = _target`"

    return True, "ok"


def verify_dir(out_dir: Path) -> list[tuple[Path, str]]:
    """Returns a list of (path, reason) for every *.py file under out_dir that
    is NOT in the exact required shim form. Empty list == everything verified."""
    failures = []
    for p in sorted(out_dir.rglob("*.py")):
        ok, reason = verify_shim_source(p.read_text())
        if not ok:
            failures.append((p, reason))
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "fragments_dir", type=Path, help="directory of <phase-id>.json landed-set fragments"
    )
    ap.add_argument("mapping_json", type=Path, help="the authoritative old-module mapping.json")
    ap.add_argument("out_dir", type=Path, help="directory to write/re-point shims into")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="verify fragment disjointness, merged landed/target consistency, and that every *.py already under out_dir is in the exact shim form, instead of generating",
    )
    ap.add_argument(
        "--print-budget",
        action="store_true",
        help="print the legacy_edge_budget of the highest-ordered fragment present, per --sequence / --sequence-file, instead of generating",
    )
    ap.add_argument(
        "--sequence", default=None, help="comma-separated phase-id ordering, e.g. p0,p1,p2"
    )
    ap.add_argument(
        "--sequence-file",
        type=Path,
        default=None,
        help='JSON file with a "sequence" array of phase ids (e.g. a sprint-manifest.json)',
    )
    args = ap.parse_args(argv)

    try:
        mapping = load_mapping(args.mapping_json)
        fragments = load_fragments(args.fragments_dir)
        merged = merge_fragments(fragments, mapping)
    except ManifestError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    if args.print_budget:
        if args.sequence and args.sequence_file:
            print("FATAL: pass only one of --sequence / --sequence-file", file=sys.stderr)
            return 1
        try:
            if args.sequence:
                sequence = [s.strip() for s in args.sequence.split(",") if s.strip()]
            elif args.sequence_file:
                seq_doc = _load_json(args.sequence_file, "sequence file")
                if "sequence" not in seq_doc:
                    raise ManifestError(f"{args.sequence_file} has no 'sequence' key")
                sequence = seq_doc["sequence"]
            else:
                raise ManifestError("--print-budget requires --sequence or --sequence-file")
            top = highest_ordered_fragment(fragments, sequence)
        except ManifestError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 1
        if "legacy_edge_budget" not in top:
            print(
                f"FATAL: fragment {top.get('phase')!r} has no 'legacy_edge_budget' field",
                file=sys.stderr,
            )
            return 1
        print(top["legacy_edge_budget"])
        return 0

    if args.verify:
        try:
            _check_merged(merged)
        except ManifestError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 1
        if not args.out_dir.is_dir():
            print(f"FATAL: --verify target {args.out_dir} is not a directory", file=sys.stderr)
            return 1
        failures = verify_dir(args.out_dir)
        if failures:
            print(
                f"FATAL: {len(failures)} shim file(s) are not in the exact required form:",
                file=sys.stderr,
            )
            for p, reason in failures:
                print(f"    {p}: {reason}", file=sys.stderr)
            return 1
        print(
            f"verified: fragments are disjoint, merged manifest is landed/target-consistent, and all *.py under {args.out_dir} are exact-form shims"
        )
        return 0

    try:
        written = generate(merged, str(args.fragments_dir), args.out_dir)
    except ManifestError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(written)} shim(s) under {args.out_dir} from {len(fragments)} fragment(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
