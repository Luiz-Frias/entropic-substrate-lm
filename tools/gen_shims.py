#!/usr/bin/env python3
"""Generates/re-points one alias shim per LEAF module named across a
directory of disjoint per-phase "landed-set" fragments, against the
authoritative module list in ``mapping.json``. A module a fragment does not
name is not an omission -- it is the DERIVED legacy default (see below).

Binding source: ``docs/plans/src-hexagonal-refactor.plan.md``, section "The mechanism rule that makes phases disjoint" (ruling DN-P1, 2026-09-17): no import-linter ignore list; each phase commits a disjoint fragment ``.claude/workflows/landed-set/<id>.json`` and this generator merges the fragments to emit shims.

Fragment schema (JSON), one file per phase, filename ``<phase-id>.json``
under a fragments directory (e.g. ``.claude/workflows/landed-set/``):

    {"phase": "<phase-id>", "modules": {"<old dotted>": {"target": "<new
    dotted, or entropic_substrate.__legacy.<x> while quarantined>", "landed":
    bool, "retired": bool (optional), "pending": "<phase-id>" (optional)}},
    "legacy_edge_budget": <int>}

``modules`` may be EMPTY (or absent entries for anything the phase doesn't
touch) -- "fragments only ever ADD" (sprint-preflight ruling, 2026-09-18):
p3's own fragment may carry nothing but its ``legacy_edge_budget``, because
every module p3 doesn't name still gets a shim, via the derived default
below, without any fragment naming it.

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
generator does NOT resolve on its own initiative for a NAMED module:
``mapping.json``'s ``dest`` is package-level only (e.g.
``entropic_substrate.ladder.analyze``, not the fully-qualified
``entropic_substrate.ladder.analyze.charts``), so deriving the exact final
leaf module name would require inventing a concatenation rule with no
primary source pinning it. Restating ``target`` per fragment, verbatim as
the schema specifies, is the choice that adds no unverified mechanism.
(The ONE case where this generator does derive a target on its own is the
legacy default below -- there is no fragment author to restate anything,
and the legacy destination is a pure, lossless rename of the old dotted
path, not a choice among destinations.)

Per module (semantics unchanged from the prior, already-corrected ruling --
see the retraction note in this generator's git history for the FIRST wrong
draft, which said `landed:true` deletes the shim):

* **A module keeps its shim for as long as it is not explicitly retired**,
  regardless of ``landed`` -- deleting a shim the moment a module lands would
  break every test/importer not yet rewritten onto the new dotted path.
* **DERIVED LEGACY DEFAULT (sprint-preflight DEFECT 1, ruled 2026-09-18):** a
  module present in ``mapping.json`` but named in NO fragment is implicitly
  ``landed: false``, target ``entropic_substrate.__legacy.<old suffix>``
  (``<old suffix>`` = the old dotted name with the ``entropic_substrate.``
  root prefix stripped). Its shim is emitted exactly as if some fragment had
  named it that way -- no fragment needs to enumerate the entire unmoved
  tree for this to happen; that is what "fragments only ever ADD" buys.
* ``landed: false`` (named explicitly, or via the derived default above) --
  module still physically quarantined under ``entropic_substrate.__legacy``;
  shim points at that legacy target.
* ``landed: true`` -- module has been ``git mv``'d to its structural home;
  shim is RE-POINTED at that new home.
* ``retired: true`` (p13-only; this generator's own explicit third state, not
  in any plan source -- "absent from every fragment" used to be ambiguous
  between "not yet listed" and "done", which is exactly what the derived
  default above resolves for the FIRST case; p13 still needs a positive,
  auditable way to say "this old name is done, delete its shim" for the
  SECOND) -- no shim emitted; overrides ``landed``.
  Also admissible before p13 for ONE case: a module deleted outright before
  its move (the fragment note names the deleting commit), because the guard
  below would otherwise refuse the whole emission on its behalf.
* ``pending: "<phase-id>"`` (FINDING 2026-09-20) -- the mapping row names a
  module that does not exist YET: measured, ``d1_training_step_records`` and
  ``d1_calibration_records`` are mapped today but only p5 creates them under
  ``__legacy/``. No shim is emitted while pending, and the guard FAILS LOUD if
  a pending module is in fact present (the creating phase then takes the row
  into its own fragment and drops the flag -- the one sanctioned migration
  between fragments, serial phases only; disjointness still holds at every
  tip). Overrides ``landed``; mutually exclusive with ``retired``.
* **Source-existence guard (FINDING 2026-09-20, p2 -> p3 gap).** Emission is
  keyed off mapping rows alone, so a row whose module was deleted outright
  before its move (measured: ``entropic_substrate.analysis
  .d1_reader_source_closure``, deleted at p2 commit 8cf5aae, ruling E-1
  extended; the mapping-coverage test tolerates the stale row because stale
  rows are swept at p13) would otherwise become a shim importing a missing
  ``__legacy`` module. generate() therefore refuses, BEFORE its first write,
  any non-retired row with no real module file at its target dotted path and
  none at its old one (a shim at the old path is not source), and refuses to
  overwrite real source still sitting at the old path (the generator was run
  before the ``git mv``). ``--verify`` re-checks the targets; ``--preflight``
  runs the same walk against a tree with no fragments yet, writing nothing,
  so p3 can name every stale row before it moves anything. The resolution of
  a stale row is a ruling (``retired: true`` in the phase's fragment, or the
  p13 sweep), never a silent skip.

**Package keys are never shimmable (sprint-preflight DEFECTS 2, 3 and 9,
ruled 2026-09-18).** A package key is any mapping key that is a dotted
prefix of another mapping key, plus the root ``entropic_substrate`` itself.
A fragment that names one FAILS LOUD, naming the key -- there is no
``package: true`` escape hatch any more, and the derived legacy default
above never manufactures an entry for one either. This generator therefore
never writes a package ``__init__.py``, under any code path.

This is correct, not merely simpler, because of how p3 restores a package
after the wholesale ``git mv`` that quarantines its children: the ORIGINAL
``__init__.py`` (real re-export content, 80 B to 2070 B across this repo's
packages -- MEASURED 2026-09-18, ``find src/entropic_substrate -name
__init__.py -exec wc -c {} +``, min ``sentence_prediction/__init__.py`` /
max ``predictive_state/__init__.py``) stays at the OLD location VERBATIM,
untouched by this generator. RETRACTION (house rule 4, constraint C9): this
sentence previously said its "relative imports (``from .charts import
...`` and similar)" resolve through the sibling shims -- that premise is
FALSE (MEASURED at de61455: no package ``__init__.py`` in the tree uses a
relative import; a leading-dot-import grep over every package
``__init__.py`` matches nothing). Its ABSOLUTE old-prefix re-export
imports resolve through the SIBLING LEAF shims this generator does write,
to the same module objects the new tree holds -- so a
RE-EXPORT, or a sentinel DEFINED IN A LEAF module and merely imported by the
package, stays single-object without this generator needing to reason about
it at all.

This does NOT cover a sentinel defined directly at the package
``__init__.py``'s own module level. RETRACTION (house rule 4, corrected by
this phase, constraint C9): this paragraph previously said that after p3
"TWO copies of that file execute under different dotted names ... a
module-level `_TOKEN = object()` or a `weakref` registry declared IN that
`__init__.py` would be two distinct objects, one per copy" -- that premise
is FALSE for any package `__init__.py` that also re-exports one of its own
children by an absolute old-prefix import (measured 2026-09-20, all 7
facades in this repo: ``data``, ``geometry``, ``init``, ``layers``,
``models``, ``observation``, ``predictive_state``). Python runs a package's
``__init__.py`` before any of its children, so the ``git mv``'d copy under
``entropic_substrate.__legacy.<pkg>`` does not quietly finish executing to
produce a second distinct object -- its own self-referencing import re-enters
the sibling leaf shim for that child while the shim is still mid-execution
(no attributes bound yet on the shim's own module object), raising
``ImportError: cannot import name ... from partially initialized module``.
The failure is LOAD-ORDER DEPENDENT: importing ``__legacy.<pkg>`` first warms
the chain, so a sweep that happens to touch ``__legacy`` before the old
prefix can stay green while a real consumer path raises. The two-distinct-
objects premise only holds for a package ``__init__.py`` with NO
self-referencing import of its own children -- this repo's ruling (option
(a) of design gate wf_43ef5138-4bc) closes the hazard for its own seven
facades by emptying the MOVED copy to a docstring-only file, present on disk
so packaging discovery still finds it; the restored copy at the old path
stays verbatim. Grep-verified 2026-09-18 (glob-independent, `grep -rn
--include='__init__.py' -e '= object()' -e weakref src/entropic_substrate`):
zero hits today, so no package `__init__.py` in this repo currently declares
a bare module-level sentinel of that shape either -- but that is a fact about
today's tree, not a guarantee this generator provides for a FUTURE package
`__init__.py` with neither a self-import nor a sentinel. The p0
sentinel-singleton test (docs/plans/src-hexagonal-refactor.plan.md, p0(c))
and this phase's cold-import subprocess guard are what would catch either
hazard shape if one were ever (re)introduced; this generator should not be
read as a substitute for either test.

Multi-destination packages (the 12 package keys in ``mapping.json`` whose
children split across up to 5 destinations) need no special case either:
nothing about this generator's behaviour depends on how many destinations a
package's children land in, because it never touches the package's own
file.

This closes the ``__path__`` hazard the previous draft of this generator
only documented: once a package shim executes,
``sys.modules['entropic_substrate.<pkg>']`` would be reassigned to the
target package object, and a later submodule import resolving through the
now-aliased parent would walk the TARGET's ``__path__`` instead of the OLD
one, silently producing a second module object for a physically separate
child shim under the old dotted name. **RETRACTION (house rule 4): the
prior text here said "this generator documents the hazard, it does not
mitigate it" -- that is no longer true and is retracted, not overwritten.
It now mitigates the hazard structurally, by refusing to ever emit the
package shim that could trigger it.**

**Disjointness (fail loud, not a warning):** a module named in TWO OR MORE
fragments is a merge error -- exactly the "disjoint fragment" property the
plan's R-parallel rule depends on (two wave lanes both claiming the same
module would silently double-move it). ``--verify`` checks this without
generating anything.

**Landed/target consistency (checked before any file is written, and by
``--verify``):** ``landed: true`` requires a target OUTSIDE
``entropic_substrate.__legacy``; ``landed: false`` requires a target INSIDE
it (the same invariant, both directions -- see ``find_landed_target_inconsistencies``).

**Emitted-files manifest (sprint-preflight DEFECT 8, ruled 2026-09-18).**
``generate()`` writes a JSON manifest of exactly the shim files it wrote
(path relative to ``out_dir``, plus each one's target), next to the
fragments directory by default, or at ``--manifest-out``. ``--verify`` reads
that manifest and checks EXACTLY those files for the required 3-statement
shim form -- it never ``rglob``s ``out_dir``. ``out_dir`` is the real
``src/entropic_substrate`` tree during structural phases, and it holds real
production modules (and, after p3, a restored real package ``__init__.py``)
alongside the shims this generator owns; scanning the whole tree scored
those as malformed shims, a false failure with nothing to do with what this
generator actually emitted.

The canonical invocation this repo pins (docs/plans/src-hexagonal-refactor.plan.md):

    uv run python tools/gen_shims.py .claude/workflows/landed-set tools/mapping.json src --manifest-out .claude/workflows/landed-set/.emitted.json
    uv run python tools/gen_shims.py .claude/workflows/landed-set tools/mapping.json src --manifest-out .claude/workflows/landed-set/.emitted.json --verify

Omitting ``--manifest-out`` defaults to exactly that path (``.emitted.json``
next to the fragments directory), so there is one truth for where it lives,
not two. The manifest is DERIVED state and UNTRACKED (gitignored, FOR-1539,
2026-09-22): it was tracked through p3b and regenerated whole by every
phase that re-points shims, which made it a same-file hub every wave lane
would edit. Each lane runs generate() then ``--verify`` in one step; the
tracked shim files are the durable evidence. That default sits INSIDE the fragments directory ``load_fragments``
globs -- ``pathlib.Path.glob("*.json")`` matches dotfiles, unlike a shell
glob (MEASURED 2026-09-18), so ``load_fragments`` explicitly skips
dot-prefixed files rather than trying, and failing, to parse the manifest
as a fragment.

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
    import sys
    import <target> as _target
    sys.modules[__name__] = _target

The ``import <target> as _target`` line MUST be a real static import (not a
string passed to ``importlib.import_module``), because import-linter's grimp
parses import statements structurally: a string-only shim makes every edge
routed through it invisible to the layers contract. ``--verify`` rejects any
shim whose body is not EXACTLY these three statements (comments aside),
UNLESS the module is flagged ``entrypoint: true`` in mapping.json (below),
in which case a second, longer form is also accepted.

**Entrypoint shims (mapping.json ``"entrypoint": true``).** A plain shim is
inert under ``python -m <old dotted>``: `-m`'s own machinery runs the
SHIM file as ``__main__``, but the shim's ``import <target> as _target``
line then imports the target under ITS OWN name, so the target's own ``if
__name__ == "__main__":`` guard never fires -- ``-m`` silently does nothing.
Seven modules in this repo are real CLI/worker entrypoints reached via
``-m`` (``entropic_substrate.__main__``, ``entropic_substrate.cli``,
``entropic_substrate.telemetry``,
``entropic_substrate.sentence_prediction.worker_command``,
``entropic_substrate.d1_trainer_mailbox_command``,
``entropic_substrate.d_ladder_worker_command``,
``entropic_substrate.analysis.d1_external_worker`` -- confirmed 2026-09-18,
each is either ``__main__.py`` itself or has its own ``if __name__ ==
"__main__":`` guard) and are flagged accordingly. For those, the shim is:

    if __name__ == "__main__":
        import runpy

        runpy.run_module("<target>", run_name="__main__", alter_sys=True)
        raise SystemExit(0)
    import sys
    import <target> as _target
    sys.modules[__name__] = _target

Run as ``python -m <old dotted>``, ``__name__ == "__main__"`` is true for
the shim itself, so the guard's body executes ``target`` via
``runpy.run_module(..., run_name="__main__", ...)`` -- which runs it with
ITS OWN ``__name__`` set to ``"__main__"``, so its guard fires exactly once
-- then ``raise SystemExit(0)`` stops the shim script before it ever
reaches the plain rebind below. Imported normally (not as ``__main__``),
the guard is skipped and the shim behaves exactly like every other one.
``entrypoint`` lives on the mapping.json entry, not the fragment schema --
it is a fact about the module, true regardless of which phase currently
names it or whether the derived legacy default filled it in.
``--verify`` accepts EXACTLY the plain 3-statement form or this 4-statement
(if-guard + rebind) form; anything else, including a differently-shaped
guard, is rejected.

**Stale-verify (manifest/resolution drift).** Passing the per-file shim-form
check is not enough: a fragment can be edited (e.g. ``landed`` flipped from
false to true, pointing a module at its new home) without re-running
``generate()``, leaving a manifest, and shim files, that both still reflect
the OLD resolution -- perfectly well-formed shims pointing at a target
nobody asked for any more. ``--verify`` therefore re-resolves fragments +
mapping.json fresh and compares the resulting ``{path: target}`` set
against the manifest's own, via ``find_manifest_drift``: an entry the fresh
resolution no longer expects (EXTRA -- e.g. retired since the last
generate), an entry the fresh resolution expects but the manifest lacks
(MISSING), or one present in both with a different target (RE-POINTED --
the landed-flip-without-regenerate case) are each named explicitly and fail
loud, before the per-file AST check ever runs.

Exit codes: 0 success. 1 unreadable fragment/mapping input, a module named in
more than one fragment, a module unknown to mapping.json, a fragment naming a
package key directly, a landed/target inconsistency, an out-of-sequence
fragment (``--print-budget``), (``--verify``) a missing emitted-files
manifest, the manifest disagreeing with a fresh fragment/mapping resolution,
or a manifest-listed shim file not in the exact required form.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess  # nosec B404 -- shim/stub ruff-canonicalization only, fixed argv below
import sys
import tempfile
from pathlib import Path

LEGACY_PREFIX = "entropic_substrate.__legacy"
ROOT_PACKAGE = "entropic_substrate"
DEFAULT_MANIFEST_NAME = ".emitted.json"
PRIVATE_IGNORE_COMMENT = "  # pyright: ignore[reportPrivateUsage]"


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


def _fragment_json_files(fragments_dir: Path) -> list[Path]:
    """The non-dotfile `*.json` paths directly under fragments_dir, sorted.
    `Path.glob("*.json")` matches dotfiles (unlike a shell glob, MEASURED
    2026-09-18), and this generator's own emitted-files manifest (default
    `.emitted.json`) lives in this same directory -- it must never be
    mistaken for a fragment. SINGLE SOURCE for both `load_fragments` and the
    CLI's `--preflight` empty-directory probe (root-cause fix, CodeRabbit
    2026-09-21: the probe used to test `any(glob("*.json"))` directly and so
    treated a directory holding only `.emitted.json` as non-empty, then
    handed it to `load_fragments`, which would fail loud with "no *.json
    fragments found" instead of the intended pre-p3 empty-directory
    derived-default path)."""
    return [p for p in sorted(fragments_dir.glob("*.json")) if not p.name.startswith(".")]


def load_fragments(fragments_dir: Path) -> dict[str, dict]:
    """Returns {phase_id: fragment_dict} for every non-dotfile *.json directly
    under fragments_dir (see `_fragment_json_files`)."""
    if not fragments_dir.is_dir():
        raise ManifestError(f"fragments directory {fragments_dir} does not exist")
    fragments: dict[str, dict] = {}
    for p in _fragment_json_files(fragments_dir):
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


def compute_package_keys(mapping: dict) -> set[str]:
    """A package key is any mapping key that is a dotted prefix of another
    mapping key, plus the root `entropic_substrate` itself (explicit, not
    just relying on the generic rule, per the sprint-preflight ruling's own
    wording -- the root is a package even in a minimal mapping that gives it
    no children)."""
    keys = set(mapping["modules"])
    packages = {k for k in keys if any(other != k and other.startswith(k + ".") for other in keys)}
    packages.add(ROOT_PACKAGE)
    return packages


def merge_fragments(fragments: dict[str, dict], mapping: dict) -> dict[str, dict]:
    """Merges module entries across fragments. FAILS LOUD (never silently
    drops or overwrites) on: a module named in more than one fragment (the
    disjointness the plan's R-parallel rule requires), a module name not
    present in mapping.json (the one authoritative module list), or a
    fragment naming a PACKAGE key directly (DEFECTS 2/3/9 -- packages are
    never shimmable, see the module docstring)."""
    known = set(mapping["modules"])
    package_keys = compute_package_keys(mapping)
    claimed_by: dict[str, str] = {}
    merged: dict[str, dict] = {}
    dupes: list[str] = []
    unknown: list[str] = []
    named_packages: list[str] = []

    for phase in sorted(fragments):
        for mod, entry in fragments[phase].get("modules", {}).items():
            if mod not in known:
                unknown.append(f"{mod} (fragment {phase})")
                continue
            if mod in package_keys:
                named_packages.append(f"{mod} (fragment {phase})")
                continue
            if mod in claimed_by:
                dupes.append(f"{mod} (fragments {claimed_by[mod]!r} and {phase!r})")
                continue
            claimed_by[mod] = phase
            merged[mod] = entry

    if unknown:
        raise ManifestError("module(s) not present in mapping.json: " + "; ".join(unknown))
    if named_packages:
        raise ManifestError(
            "package key(s) named directly in a fragment -- packages are never "
            "shimmable, no fragment may name one (sprint-preflight DEFECTS 2/3/9): "
            + "; ".join(named_packages)
        )
    if dupes:
        raise ManifestError(
            "module(s) claimed by more than one fragment (disjointness violation): "
            + "; ".join(dupes)
        )
    return merged


def resolve_all_modules(merged: dict[str, dict], mapping: dict) -> dict[str, dict]:
    """DEFECT 1 fix: a module present in mapping.json but named in NO
    fragment is not an omission to flag -- it is the derived legacy default
    (landed:false, target under __legacy), matching "fragments only ever
    ADD" (docs/plans/src-hexagonal-refactor.plan.md). Package keys are
    excluded on both sides of this: they are never shimmable (DEFECTS
    2/3/9) and so never get a synthesized entry either.

    Also stamps ``entrypoint`` onto every resolved entry from mapping.json's
    OWN per-module metadata (never from the fragment) -- the fragment schema
    only ever restates ``target``/``landed``/``retired``; whether a module
    is an entrypoint is a fact about the module, not a per-phase choice, so
    it lives once in mapping.json and is looked up here regardless of
    whether the fragment named the module or the derived default filled it
    in. Entries are copied, never mutated in place, so a caller's fragment
    dict is never modified by this lookup."""
    package_keys = compute_package_keys(mapping)
    resolved: dict[str, dict] = {}
    for old, meta in mapping["modules"].items():
        if old in package_keys:
            continue
        if old in merged:
            entry = dict(merged[old])
        else:
            # Every non-package mapping key is nested under the root package
            # (compute_package_keys always excludes the root itself), so
            # this strip is unconditional, not a guessed fallback.
            suffix = old.removeprefix(f"{ROOT_PACKAGE}.")
            entry = {"target": f"{LEGACY_PREFIX}.{suffix}", "landed": False}
        entry["entrypoint"] = bool(meta.get("entrypoint", False))
        resolved[old] = entry
    return resolved


def find_landed_target_inconsistencies(modules: dict) -> list[str]:
    """landed:true + target still under __legacy, or landed:false + target
    NOT under __legacy -- both directions of the same invariant."""
    bad = []
    for old, entry in sorted(modules.items()):
        if entry.get("retired", False):
            continue
        landed = entry.get("landed", False)
        target = entry.get("target", "")
        is_legacy = target == LEGACY_PREFIX or target.startswith(LEGACY_PREFIX + ".")
        if landed and is_legacy:
            bad.append(f"{old}: landed=true but target {target!r} is still under {LEGACY_PREFIX}")
        elif not landed and not is_legacy:
            bad.append(f"{old}: landed=false but target {target!r} is not under {LEGACY_PREFIX}")
    return bad


def shim_source(old: str, target: str, source_label: str, is_entrypoint: bool = False) -> str:
    header = (
        f"# Generated by gen_shims.py from {source_label} -- do not hand-edit.\n"
        f"# Alias shim for {old} -> {target} (docs/plans/src-hexagonal-refactor.plan.md 'Shim form (required)').\n"
    )
    # RULED 2026-09-20 (conductor, third design gate wf_43ef5138-4bc, "SECOND"
    # item): `import sys` sits AHEAD of the static target import. The ruling
    # pins the static import THEN the rebind assignment; `import sys`'s own
    # position between them was never load-bearing (grimp only cares about the
    # target import being a real static statement, and the rebind assignment
    # only needs `sys` bound by the time it runs) -- reordering it is an
    # in-envelope, ruff-fixed-point-preserving edit to this function alone.
    rebind = f"import sys\nimport {target} as _target\nsys.modules[__name__] = _target\n"
    if not is_entrypoint:
        return header + rebind
    # ENTRYPOINT FORM (mapping.json "entrypoint": true): `python -m <old>`
    # must execute `target` as __main__ exactly once, never fall through to
    # the plain rebind below -- `raise SystemExit(0)` inside the guard is
    # what stops execution there. Importing the shim normally (not as
    # __main__) skips the guard entirely and falls straight to the same
    # rebind every other shim has.
    entry_guard = (
        '# Entrypoint shim: "python -m <old dotted>" must run the target as\n'
        "# __main__ (its own if-guard is otherwise never reached through an\n"
        "# ordinary rebind -- see the module docstring, 'Entrypoint shims').\n"
        'if __name__ == "__main__":\n'
        "    import runpy\n"
        "\n"
        f'    runpy.run_module("{target}", run_name="__main__", alter_sys=True)\n'
        "    raise SystemExit(0)\n"
    )
    return header + entry_guard + rebind


def shim_output_path(out_dir: Path, old: str) -> Path:
    """Every module reaching this point is a LEAF (package keys are filtered
    out before this is ever called, in merge_fragments and
    resolve_all_modules) -- so this always writes a plain `<leaf>.py`, never
    an `__init__.py`."""
    *pkg_parts, leaf = old.split(".")
    return out_dir.joinpath(*pkg_parts, f"{leaf}.py")


# ---------------------------------------------------------------------------
# Typed shim pair (RULED 2026-09-20, operator, option (b) of design-gate
# wf_7f27b290-e61; stub text refined the same day by the conductor inside
# (b), resolving wf_7f55472a-499). basedpyright (strict, 0-error baseline)
# cannot see through the `sys.modules` rebind, so it would report every
# surviving old-prefix `from entropic_substrate.X import Name` at the
# consumer. The stub is type-level only (CPython never imports a `.pyi`) and
# is DERIVED from the TARGET module's own AST: one `from <target> import
# <name> as <name>` line per module-level binding, sorted, plus `__all__`
# when the target defines one as a literal list/tuple of strings. NEVER a
# star import: a star stub honours the target's `__all__` (omitting names
# real importers use) and trips ruff F403, which `just lint` cannot
# auto-fix (both measured 2026-09-20 with a control). This derivation
# function is the SINGLE SOURCE: generate() calls it to emit, `--verify`
# calls it again to re-derive and compares byte-for-byte.
# ---------------------------------------------------------------------------


def _is_private_name(name: str) -> bool:
    """basedpyright's reportPrivateUsage fires on a single- or double-
    leading-underscore name that is not a dunder (leading AND trailing
    double underscore, e.g. `__all__`, `__version__`, which are not
    'private' by that rule)."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _assign_target_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out: list[str] = []
        for elt in target.elts:
            out.extend(_assign_target_names(elt))
        return out
    if isinstance(target, ast.Starred):
        return _assign_target_names(target.value)
    return []


def _bound_names_in_stmts(stmts: list[ast.stmt]) -> list[str]:
    """Names bound by executing this statement list AT MODULE LEVEL: `def`/
    `class` (the name itself, not what's inside its body -- function and
    class bodies are NOT recursed into), `import`/`from import` (asname or
    the bound name; `*` is skipped -- it binds no single name statically),
    assignment targets (including tuple/list/starred unpacking), and an
    annotated assignment ONLY when it actually has a value (a bare
    `x: int` binds no runtime name). `if`/`try` blocks (including `if
    TYPE_CHECKING:`) are RECURSED INTO, because their bodies still execute
    (or are still visible to a type checker, for TYPE_CHECKING) at module
    level -- this is what the ruled derivation calls out explicitly."""
    names: list[str] = []
    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(stmt.name)
        elif isinstance(stmt, ast.TypeAlias):
            # PEP 695 `type X = ...` (Python 3.12+). MEASURED post-mass-
            # emission (constraint 5 firing after the fact): 9 real
            # quarantined modules define their public type this way
            # (FrozenJson, JsonValue, InterfaceMode, D1TrainerRequest,
            # ...); missing this branch produced 83 basedpyright errors,
            # all one root cause -- the name was simply never derived.
            if isinstance(stmt.name, ast.Name):
                names.append(stmt.name.id)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                names.append(alias.asname or alias.name.split(".")[0])
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                names.append(alias.asname or alias.name)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                names.extend(_assign_target_names(target))
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None and isinstance(stmt.target, ast.Name):
                names.append(stmt.target.id)
        elif isinstance(stmt, ast.If):
            names.extend(_bound_names_in_stmts(stmt.body))
            names.extend(_bound_names_in_stmts(stmt.orelse))
        elif isinstance(stmt, ast.Try):
            names.extend(_bound_names_in_stmts(stmt.body))
            for handler in stmt.handlers:
                names.extend(_bound_names_in_stmts(handler.body))
            names.extend(_bound_names_in_stmts(stmt.orelse))
            names.extend(_bound_names_in_stmts(stmt.finalbody))
    return names


def _module_all(tree: ast.Module) -> list[str] | None:
    """The target's own `__all__`, IFF it is a literal list/tuple of string
    constants assigned (plain or annotated) at module level. Anything else
    (computed, imported, augmented) is treated as absent: no `__all__` line
    is emitted, but every module-level binding is still re-exported, so
    nothing the target actually exposes becomes invisible to the stub."""
    for stmt in tree.body:
        value: ast.expr | None
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in stmt.targets
        ):
            value = stmt.value
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "__all__"
            and stmt.value is not None
        ):
            value = stmt.value
        else:
            continue
        if isinstance(value, (ast.List, ast.Tuple)) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts
        ):
            return [e.value for e in value.elts]
        return None
    return None


def derive_stub_body(target: str, target_source: str) -> str:
    """The single derivation function (SINGLE SOURCE for generate() and
    `--verify`): one `from <target> import <name> as <name>` line per
    module-level binding of `target_source`'s AST, sorted by name, an
    inline `# pyright: ignore[reportPrivateUsage]` on every underscore
    name, plus a literal `__all__` re-statement when the target defines
    one. Raises on unparseable source (fail loud, never a silent empty
    stub)."""
    tree = ast.parse(target_source)
    all_names = _module_all(tree)
    names = sorted(set(_bound_names_in_stmts(tree.body)))
    if all_names is not None:
        # `__all__` gets its own literal restatement below (SLICE-PROBE
        # FINDING, 2026-09-20: leaving it in the generic name list too
        # produced BOTH `from X import __all__ as __all__` and a literal
        # `__all__ = [...]` in the same stub -- redundant, the second
        # silently wins at runtime). Skipped only when a literal WAS
        # derived; when the target's `__all__` is computed/non-literal
        # (returns None), the generic line stays as the only exposure a
        # stub gives that attribute.
        names = [n for n in names if n != "__all__"]
    lines = [
        f"from {target} import {name} as {name}"
        + (PRIVATE_IGNORE_COMMENT if _is_private_name(name) else "")
        for name in names
    ]
    if all_names is not None:
        lines.append("__all__ = " + repr(list(all_names)))
    return "\n".join(lines) + ("\n" if lines else "")


def stub_source_raw(target: str, target_source: str, source_label: str) -> str:
    header = (
        f"# Generated by gen_shims.py from {source_label} -- do not hand-edit.\n"
        f"# Derived type stub for {target} (typed shim pair, RULED 2026-09-20; "
        "explicit derived re-exports, never a star import).\n"
    )
    return header + derive_stub_body(target, target_source)


def _find_pyproject(start: Path) -> Path | None:
    """Walks upward from `start` looking for the repo's own `pyproject.toml`,
    so the ruff pipeline below canonicalizes with the SAME isort/format
    settings `just lint` uses (constraint: never let the derivation drift
    from ruff's actual behaviour on this tree). Returns None when none is
    found (unit-test fixtures with no repo config of their own), in which
    case ruff runs with its own defaults -- correctness of the AST-derived
    NAMES never depends on this, only the canonical on-disk ORDER/spacing
    does, and those fixtures don't assert against real repo formatting."""
    for candidate in (start.resolve(), *start.resolve().parents):
        p = candidate / "pyproject.toml"
        if p.is_file():
            return p
    return None


def canonicalize_python_texts(raw_by_relpath: dict[str, str], out_dir: Path) -> dict[str, str]:
    """Pipes every raw derived `.py`/`.pyi` text through THIS repo's own
    `ruff check --fix --select I` (import sort only) then `ruff format`,
    batched into ONE subprocess pair regardless of how many files are given
    -- never reimplements ruff's own sort key (case-insensitive
    order-by-type; MEASURED `_TAN_LIMIT` before `_TANH_LIMIT`,
    SCREAMING_CASE constants after the underscore group). Originally
    stub-only ("canonicalize_stub_texts"); generate() found the SAME
    deadlock class on the shim `.py` side (constraint 3, design gate
    2026-09-20 -- MEASURED: `known-first-party = ["entropic_substrate"]`
    splits `import sys` from the first-party target import into two isort
    groups and `ruff format` inserts a blank line after the import block,
    so `ruff check . --fix` rewrites every emitted shim mid-gate), so this
    function now canonicalizes both file classes through the identical
    pipeline. Every file is what `just lint`'s `ruff check . --fix` would
    produce for it, by construction, so nothing rewrites it mid-gate and
    `--verify`'s byte-for-byte stub comparison (which re-derives through
    this SAME function) cannot deadlock against it. Files land in the temp
    dir at the SAME relative path they occupy on disk, so any
    path-pattern-scoped ruff config (e.g. per-file-ignores) still applies.
    Fails loud if `ruff` cannot run at all."""
    if not raw_by_relpath:
        return {}
    pyproject = _find_pyproject(out_dir)
    config_args = ["--config", str(pyproject)] if pyproject else []
    with tempfile.TemporaryDirectory(prefix="gen_shims_stub_fmt_") as td:
        tdp = Path(td)
        for relpath, text in raw_by_relpath.items():
            p = tdp / relpath
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        try:
            subprocess.run(  # nosec B603 B607 -- "ruff" is deliberately resolved from PATH
                # (the venv's own pinned ruff, same idiom as sinks.py's "git"),
                # argv is a fixed list, no shell.
                ["ruff", "check", "--fix", "--select", "I", *config_args, str(tdp)],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(  # nosec B603 B607 -- same fixed-argv, PATH-resolved idiom
                ["ruff", "format", *config_args, str(tdp)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            # CalledProcessError's own str() is just argv + returncode -- the
            # actual reason (a malformed derived stub, a ruff config error)
            # is in the captured streams this call already asked for
            # (capture_output=True). Root-cause fix, CodeRabbit 2026-09-21:
            # the prior text swallowed both, leaving "returned non-zero exit
            # status 1" as the only diagnostic for a fail-loud path.
            raise ManifestError(
                f"ruff stub-canonicalization pipeline failed: {exc}\nstdout:\n{exc.stdout}\n"
                f"stderr:\n{exc.stderr}"
            ) from exc
        except FileNotFoundError as exc:
            raise ManifestError(f"ruff stub-canonicalization pipeline failed: {exc}") from exc
        return {relpath: (tdp / relpath).read_text() for relpath in raw_by_relpath}


def _check_merged(modules: dict) -> None:
    inconsistent = find_landed_target_inconsistencies(modules)
    if inconsistent:
        raise ManifestError(
            "landed/target inconsistency (landed:true must target outside __legacy, "
            "landed:false must target inside __legacy): " + "; ".join(inconsistent)
        )


def module_file(out_dir: Path, dotted: str) -> Path | None:
    """The on-disk file a dotted module name resolves to under ``out_dir``:
    ``<dotted as dirs>.py`` first, then ``<dotted as dirs>/__init__.py`` (a
    leaf key whose module is a childless package); None when neither is a
    file. Mirrors the import system's own order for a module vs a package."""
    base = out_dir.joinpath(*dotted.split("."))
    for candidate in (base.parent / f"{base.name}.py", base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _is_shim_file(path: Path) -> bool:
    return verify_shim_source(path.read_text())[0]


def find_missing_sources(modules: dict, out_dir: Path) -> list[str]:
    """Rows (non-retired) whose module has NO real source anywhere on disk:
    nothing at the target dotted path, and at the old dotted path either
    nothing or only a shim. Accepting real source at the old path is what
    makes this valid BEFORE the move (``--preflight``); refusing to count a
    shim there is what keeps it load-bearing AFTER p3, when every old path
    holds one. Returns human-readable rows, one per stale entry."""
    missing: list[str] = []
    for old, entry in sorted(modules.items()):
        if entry.get("retired", False) or entry.get("pending"):
            continue
        target = entry["target"]
        if module_file(out_dir, target) is not None:
            continue
        old_file = module_file(out_dir, old)
        if old_file is not None and not _is_shim_file(old_file):
            continue
        missing.append(
            f"{old} -> {target}: no module file under {out_dir} at either dotted path "
            "(stale mapping row, or the move never happened)"
        )
    return missing


def find_unmoved_sources(modules: dict, out_dir: Path) -> list[str]:
    """Rows (non-retired) whose OLD dotted path still holds real, non-shim
    source. Emitting would overwrite that source with a three-line shim, so
    generate() refuses; the caller ran the generator before the ``git mv``."""
    unmoved: list[str] = []
    for old, entry in sorted(modules.items()):
        if entry.get("retired", False) or entry.get("pending"):
            continue
        old_file = module_file(out_dir, old)
        if old_file is not None and not _is_shim_file(old_file):
            unmoved.append(f"{old}: real source still at {old_file}")
    return unmoved


def find_pending_present(modules: dict, out_dir: Path) -> list[str]:
    """Rows declared ``pending`` whose module DOES exist (at the target, or as
    real source at the old path): the declaration is stale and the creating
    phase must take the row over. Also rejects a row that is both pending and
    retired -- the two say opposite things about the old name."""
    present: list[str] = []
    for old, entry in sorted(modules.items()):
        phase = entry.get("pending")
        if not phase:
            continue
        if entry.get("retired", False):
            present.append(f"{old}: both pending ({phase!r}) and retired")
            continue
        target_file = module_file(out_dir, entry["target"])
        old_file = module_file(out_dir, old)
        if target_file is not None or (old_file is not None and not _is_shim_file(old_file)):
            present.append(
                f"{old}: declared pending until {phase!r} but its module is present at "
                f"{target_file or old_file}"
            )
    return present


def _check_sources(modules: dict, out_dir: Path) -> None:
    stale_pending = find_pending_present(modules, out_dir)
    if stale_pending:
        raise ManifestError(
            f"{len(stale_pending)} pending declaration(s) are stale -- the creating phase "
            "takes the row into its own fragment and drops the flag: " + "; ".join(stale_pending)
        )
    missing = find_missing_sources(modules, out_dir)
    if missing:
        raise ManifestError(
            f"{len(missing)} mapping row(s) name a module with no source on disk -- "
            "refusing to emit ANY shim (a shim importing a missing module is an "
            "ImportError far from its cause; resolve by ruling, never by skipping): "
            + "; ".join(missing)
        )
    unmoved = find_unmoved_sources(modules, out_dir)
    if unmoved:
        raise ManifestError(
            f"{len(unmoved)} module(s) still hold real source at the old dotted path -- "
            "refusing to overwrite source with a shim (run the move first): " + "; ".join(unmoved)
        )


def generate(
    modules: dict,
    source_label: str,
    out_dir: Path,
    manifest_path: Path | None = None,
) -> list[Path]:
    _check_merged(modules)
    _check_sources(modules, out_dir)  # before the first write: all-or-nothing
    written: list[Path] = []
    manifest_entries: list[dict] = []
    stub_entries: list[dict] = []
    raw_texts: dict[str, str] = {}  # relpath -> raw text, BOTH shims and stubs
    out_paths: dict[str, Path] = {}
    for old, entry in sorted(modules.items()):
        if entry.get("retired", False):
            continue  # old name fully retired (p13, or deleted before its move): no shim
        if entry.get("pending"):
            continue  # module not created yet (doctrine above): no shim until it exists
        target = entry["target"]
        is_entrypoint = bool(entry.get("entrypoint", False))
        out_path = shim_output_path(out_dir, old)
        shim_relpath = str(out_path.relative_to(out_dir))
        # RAW text only here -- NOT written yet. Writing the un-canonicalized
        # form was the defect (constraint 3, design gate 2026-09-20): this
        # repo's own ruff config splits `import sys` from the first-party
        # target import into two isort groups and formats a blank line after
        # the import block, so `ruff check . --fix` (house rule: `just
        # lint`, run inside both `just check` and `just pc-ci`) rewrote every
        # emitted shim mid-gate -- the SAME deadlock class the stub pipeline
        # was built to avoid, just on the shim side of the typed pair.
        raw_texts[shim_relpath] = shim_source(
            old, target, source_label, is_entrypoint=is_entrypoint
        )
        out_paths[shim_relpath] = out_path
        written.append(out_path)
        manifest_entries.append(
            {
                "path": shim_relpath,
                "target": target,
                "entrypoint": is_entrypoint,
            }
        )
        # Typed shim PAIR (RULED 2026-09-20): every non-retired, non-pending
        # row also gets a `.pyi` stub, derived from the TARGET's own AST.
        # `_check_sources` above already guarantees the target resolves to a
        # real file by this point (a row can only reach this loop with real
        # source at the OLD path if `find_unmoved_sources` would have refused
        # the whole emission already -- see that function's docstring) --
        # this lookup failing is a defensive fail-loud, not a reachable path.
        target_file = module_file(out_dir, target)
        if target_file is None:
            raise ManifestError(
                f"{old}: target {target!r} has no module file on disk under {out_dir} -- "
                "cannot derive its type stub (run the move before generating)"
            )
        stub_path = out_path.with_suffix(".pyi")
        stub_relpath = str(stub_path.relative_to(out_dir))
        raw_texts[stub_relpath] = stub_source_raw(target, target_file.read_text(), source_label)
        out_paths[stub_relpath] = stub_path
        stub_entries.append({"path": stub_relpath, "target": target})
    # ONE canonicalization pass over BOTH the shim `.py` and stub `.pyi`
    # raw text, so the whole typed pair is a ruff fixed point by
    # construction and nothing mid-gate (`ruff check . --fix`, `ruff
    # format`) can rewrite anything under out_dir after generate() returns.
    canonical = canonicalize_python_texts(raw_texts, out_dir)
    for relpath, text in canonical.items():
        p = out_paths[relpath]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {"out_dir": str(out_dir), "shims": manifest_entries, "stubs": stub_entries},
                indent=2,
            )
        )
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
# --verify: (1) fragment disjointness/package-guard (via merge_fragments),
# (2) merged landed/target consistency, (3) AST-exact shim-form check on
# EXACTLY the files generate()'s emitted-files manifest lists (DEFECT 8 --
# never an rglob of out_dir, which may hold real production modules).
# ---------------------------------------------------------------------------


def _verify_standard_rebind(stmts: list[ast.stmt]) -> tuple[bool, str]:
    """The 3-statement rebind every shim ends with (entrypoint or not).
    Order RULED 2026-09-20 (wf_43ef5138-4bc, "SECOND"): `import sys` first,
    then the static target import, then the rebind assignment."""
    if len(stmts) != 3:
        return False, f"expected exactly 3 rebind statements, found {len(stmts)}"

    s0, s1, s2 = stmts

    if not (
        isinstance(s0, ast.Import)
        and len(s0.names) == 1
        and s0.names[0].name == "sys"
        and s0.names[0].asname is None
    ):
        return False, "statement 1 must be `import sys`"

    if not (isinstance(s1, ast.Import) and len(s1.names) == 1 and s1.names[0].asname == "_target"):
        return (
            False,
            "statement 2 must be `import <target> as _target` (a real static import, not from-import or importlib)",
        )

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


def _verify_entrypoint_guard(stmt: ast.stmt) -> tuple[bool, str]:
    """The `if __name__ == "__main__": ...` block an entrypoint shim leads
    with -- exactly `import runpy`, `runpy.run_module("<target>",
    run_name="__main__", alter_sys=True)`, `raise SystemExit(0)`, no
    else/elif. The literal string argument to run_module is not compared
    against the shim's own restated target here (the caller may do that);
    this only checks the AST SHAPE is the required one."""
    if not isinstance(stmt, ast.If):
        return False, 'entrypoint shim must open with `if __name__ == "__main__":`'
    test = stmt.test
    guard_ok = (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )
    if not guard_ok:
        return False, 'entrypoint if-guard must be exactly `if __name__ == "__main__":`'
    if stmt.orelse:
        return False, "entrypoint if-guard must have no else/elif"

    body = stmt.body
    if len(body) != 3:
        return False, f"entrypoint if-guard body must have exactly 3 statements, found {len(body)}"
    b0, b1, b2 = body

    if not (
        isinstance(b0, ast.Import)
        and len(b0.names) == 1
        and b0.names[0].name == "runpy"
        and b0.names[0].asname is None
    ):
        return False, "entrypoint if-guard statement 1 must be `import runpy`"

    call_ok = (
        isinstance(b1, ast.Expr)
        and isinstance(b1.value, ast.Call)
        and isinstance(b1.value.func, ast.Attribute)
        and b1.value.func.attr == "run_module"
        and isinstance(b1.value.func.value, ast.Name)
        and b1.value.func.value.id == "runpy"
        and len(b1.value.args) == 1
        and isinstance(b1.value.args[0], ast.Constant)
        and isinstance(b1.value.args[0].value, str)
    )
    if call_ok:
        kwargs = {kw.arg: kw.value for kw in b1.value.keywords}
        call_ok = (
            set(kwargs) == {"run_name", "alter_sys"}
            and isinstance(kwargs["run_name"], ast.Constant)
            and kwargs["run_name"].value == "__main__"
            and isinstance(kwargs["alter_sys"], ast.Constant)
            and kwargs["alter_sys"].value is True
        )
    if not call_ok:
        return (
            False,
            'entrypoint if-guard statement 2 must be `runpy.run_module("<target>", '
            'run_name="__main__", alter_sys=True)`',
        )

    raise_ok = (
        isinstance(b2, ast.Raise)
        and isinstance(b2.exc, ast.Call)
        and isinstance(b2.exc.func, ast.Name)
        and b2.exc.func.id == "SystemExit"
        and len(b2.exc.args) == 1
        and isinstance(b2.exc.args[0], ast.Constant)
        and b2.exc.args[0].value == 0
    )
    if not raise_ok:
        return False, "entrypoint if-guard statement 3 must be `raise SystemExit(0)`"

    return True, "ok"


def verify_shim_source(source: str) -> tuple[bool, str]:
    """Accepts exactly two shapes: the standard 3-statement rebind, or the
    ENTRYPOINT form (mapping.json ``entrypoint: true``) -- the same rebind
    preceded by the `if __name__ == "__main__":` guard `shim_source` emits
    when ``is_entrypoint=True``. Anything else (a different statement
    count, a differently-shaped guard, a from-import, an importlib string
    shim) is rejected."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"does not parse: {exc}"

    stmts = tree.body
    if len(stmts) == 3:
        return _verify_standard_rebind(stmts)
    if len(stmts) == 4:
        guard_ok, guard_reason = _verify_entrypoint_guard(stmts[0])
        if not guard_ok:
            return False, guard_reason
        return _verify_standard_rebind(stmts[1:])
    return (
        False,
        "expected exactly 3 statements (standard shim) or 4 (entrypoint shim: "
        f'`if __name__ == "__main__":` guard + rebind), found {len(stmts)}',
    )


def load_emitted_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        raise ManifestError(
            f"no emitted-files manifest at {manifest_path} -- run generate() (or this "
            "CLI without --verify) at least once before verifying"
        )
    return _load_json(manifest_path, "emitted-files manifest")


def verify_shims_from_manifest(doc: dict, out_dir: Path) -> list[tuple[Path, str]]:
    """Returns a list of (path, reason) for every shim the emitted-files
    manifest lists that is missing or not in the exact required form. Empty
    list == everything verified. Deliberately never walks out_dir -- a real
    production module or a restored package __init__.py living in the same
    tree, but never emitted by this generator, is not this function's
    concern (DEFECT 8)."""
    failures: list[tuple[Path, str]] = []
    for shim in doc.get("shims", []):
        p = out_dir / shim["path"]
        if not p.is_file():
            failures.append((p, "listed in emitted-files manifest but file is missing"))
            continue
        ok, reason = verify_shim_source(p.read_text())
        if not ok:
            failures.append((p, reason))
    return failures


def compute_expected_shim_map(modules: dict, out_dir: Path) -> dict[str, str]:
    """The {path -> target} set generate() WOULD produce right now from the
    freshly resolved module set -- used by find_manifest_drift to catch a
    manifest that no longer reflects the fragments/mapping on disk."""
    expected: dict[str, str] = {}
    for old, entry in modules.items():
        if entry.get("retired", False) or entry.get("pending"):
            continue
        out_path = shim_output_path(out_dir, old)
        expected[str(out_path.relative_to(out_dir))] = entry["target"]
    return expected


def find_manifest_drift(manifest_shims: list[dict], expected: dict[str, str]) -> list[str]:
    """STALE-VERIFY fix: a phase that edits its fragment (e.g. flips
    landed:true) without re-running generate() must not verify green with a
    manifest, and shim files, that still reflect the OLD resolution. Fails
    loud naming each: a manifest entry the fresh resolution no longer
    expects (EXTRA -- e.g. the module was since retired), an expected entry
    missing from the manifest (MISSING), or one present in both whose target
    disagrees (RE-POINTED, the landed-flip-without-regenerate case)."""
    manifest_map = {s["path"]: s["target"] for s in manifest_shims}
    extra = sorted(set(manifest_map) - set(expected))
    missing = sorted(set(expected) - set(manifest_map))
    repointed = sorted(
        p for p in (set(manifest_map) & set(expected)) if manifest_map[p] != expected[p]
    )

    problems: list[str] = []
    if extra:
        problems.append(
            "manifest lists shim(s) no longer in the resolved module set (stale, or "
            "since retired -- regenerate): " + ", ".join(extra)
        )
    if missing:
        problems.append(
            "resolved module set expects shim(s) missing from the manifest (regenerate): "
            + ", ".join(missing)
        )
    if repointed:
        detail = "; ".join(
            f"{p} (manifest target {manifest_map[p]!r}, freshly resolved target {expected[p]!r})"
            for p in repointed
        )
        problems.append(
            "manifest shim(s) point at a STALE target -- a fragment changed (e.g. "
            "landed flipped) without regenerating: " + detail
        )
    return problems


def compute_expected_stub_map(modules: dict, out_dir: Path) -> dict[str, str]:
    """The {path -> target} set generate() WOULD produce for stubs right now
    -- same shape as compute_expected_shim_map, reused directly by
    find_manifest_drift (both lists share the {"path", "target"} schema)."""
    expected: dict[str, str] = {}
    for old, entry in modules.items():
        if entry.get("retired", False) or entry.get("pending"):
            continue
        out_path = shim_output_path(out_dir, old)
        stub_path = out_path.with_suffix(".pyi")
        expected[str(stub_path.relative_to(out_dir))] = entry["target"]
    return expected


def verify_stubs_from_manifest(
    doc: dict, out_dir: Path, source_label: str
) -> list[tuple[Path, str]]:
    """Stub verification is RE-DERIVATION from the target's current on-disk
    AST, run through the same ruff canonicalization pipeline, compared
    byte-for-byte -- never the shim exact-form AST check (`verify_shim_
    source`), and never applied to a `.py` shim. A missing stub file, a
    target with no source to re-derive from, or a content mismatch (a
    hand-edited stub, one whose target's public names changed since it was
    generated, or one a mutation pointed at a different target) is reported
    by path, one entry per failure."""
    failures: list[tuple[Path, str]] = []
    raw_by_relpath: dict[str, str] = {}
    order: list[str] = []
    for stub in doc.get("stubs", []):
        p = out_dir / stub["path"]
        if not p.is_file():
            failures.append((p, "listed in emitted-files manifest but stub file is missing"))
            continue
        target_file = module_file(out_dir, stub["target"])
        if target_file is None:
            failures.append(
                (p, f"stub target {stub['target']!r} has no module file on disk to re-derive from")
            )
            continue
        raw_by_relpath[stub["path"]] = stub_source_raw(
            stub["target"], target_file.read_text(), source_label
        )
        order.append(stub["path"])
    canonical = canonicalize_python_texts(raw_by_relpath, out_dir) if raw_by_relpath else {}
    for relpath in order:
        p = out_dir / relpath
        expected_text = canonical[relpath]
        actual_text = p.read_text()
        if actual_text != expected_text:
            failures.append(
                (p, "stub content does not match a fresh re-derivation from its target's AST")
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "fragments_dir", type=Path, help="directory of <phase-id>.json landed-set fragments"
    )
    ap.add_argument("mapping_json", type=Path, help="the authoritative old-module mapping.json")
    ap.add_argument(
        "out_dir",
        type=Path,
        help="the src root, i.e. the directory that CONTAINS the top-level entropic_substrate package (shims land at <out_dir>/<dotted path as dirs>.py); pass `src`, not `src/entropic_substrate`",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="verify fragment disjointness/package-guard, merged landed/target consistency, "
        "and that every file the emitted-files manifest lists is in the exact shim form, "
        "instead of generating",
    )
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="write nothing; resolve every mapping row (a missing or empty fragments "
        "directory is the legitimate pre-p3 state and resolves to the __legacy defaults) "
        "and fail loud naming every row whose module has no source on disk at either its "
        "old or its target dotted path",
    )
    ap.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="where generate() writes (and --verify reads) the emitted-files manifest; "
        f"defaults to <fragments_dir>/{DEFAULT_MANIFEST_NAME}",
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

    manifest_path = args.manifest_out or (args.fragments_dir / DEFAULT_MANIFEST_NAME)

    try:
        mapping = load_mapping(args.mapping_json)
        if args.preflight and not _fragment_json_files(args.fragments_dir):
            fragments = {}  # pre-p3: no fragment exists yet; every row takes the derived default
        else:
            fragments = load_fragments(args.fragments_dir)
        merged = merge_fragments(fragments, mapping)
        resolved = resolve_all_modules(merged, mapping)
    except ManifestError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    if args.preflight:
        stale_pending = find_pending_present(resolved, args.out_dir)
        if stale_pending:
            print(f"FATAL: {len(stale_pending)} stale pending declaration(s):", file=sys.stderr)
            for row in stale_pending:
                print(f"    {row}", file=sys.stderr)
            return 1
        missing = find_missing_sources(resolved, args.out_dir)
        if missing:
            print(
                f"FATAL: {len(missing)} mapping row(s) name a module with no source on disk "
                f"under {args.out_dir}:",
                file=sys.stderr,
            )
            for row in missing:
                print(f"    {row}", file=sys.stderr)
            return 1
        # find_missing_sources/find_unmoved_sources both skip retired/pending
        # rows internally (neither is a row this preflight pass actually
        # resolves against on-disk source), so the summary below counts only
        # the CHECKED subset -- not len(resolved), which still includes them
        # (root-cause fix, CodeRabbit 2026-09-21: the prior text used
        # len(resolved) as the "resolves to source on disk" / "at target"
        # total, silently crediting retired/pending rows as checked-and-
        # landed when they were never looked at).
        excluded = sum(1 for e in resolved.values() if e.get("retired", False) or e.get("pending"))
        checked = len(resolved) - excluded
        unmoved = len(find_unmoved_sources(resolved, args.out_dir))
        print(
            f"preflight ok: {checked} row(s) resolve to source on disk under "
            f"{args.out_dir} ({unmoved} still at the old path, "
            f"{checked - unmoved} at the target; {excluded} retired/pending row(s) "
            f"not checked) from {len(fragments)} fragment(s)"
        )
        return 0

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
            _check_merged(resolved)
            doc = load_emitted_manifest(manifest_path)
            expected = compute_expected_shim_map(resolved, args.out_dir)
            drift = find_manifest_drift(doc.get("shims", []), expected)
            expected_stubs = compute_expected_stub_map(resolved, args.out_dir)
            drift += find_manifest_drift(doc.get("stubs", []), expected_stubs)
            if drift:
                raise ManifestError("; ".join(drift))
            failures = verify_shims_from_manifest(doc, args.out_dir)
            failures += verify_stubs_from_manifest(doc, args.out_dir, str(args.fragments_dir))
            stale_pending = find_pending_present(resolved, args.out_dir)
            if stale_pending:
                raise ManifestError("stale pending declaration(s): " + "; ".join(stale_pending))
            missing = find_missing_sources(resolved, args.out_dir)
            if missing:
                raise ManifestError(
                    f"{len(missing)} shim(s) point at a module with no source on disk: "
                    + "; ".join(missing)
                )
        except ManifestError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 1
        if failures:
            print(
                f"FATAL: {len(failures)} shim/stub file(s) are not in the exact required form:",
                file=sys.stderr,
            )
            for p, reason in failures:
                print(f"    {p}: {reason}", file=sys.stderr)
            return 1
        print(
            f"verified: fragments are disjoint, no package key was named, merged manifest is "
            f"landed/target-consistent, the manifest matches the freshly resolved module set, "
            f"every shim in {manifest_path} is exact-form, every stub re-derives byte-for-byte "
            f"from its target's AST, and every target has source on disk"
        )
        return 0

    try:
        written = generate(
            resolved, str(args.fragments_dir), args.out_dir, manifest_path=manifest_path
        )
    except ManifestError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(written)} shim(s) under {args.out_dir} from {len(fragments)} fragment(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
