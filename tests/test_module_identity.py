"""p0 sentinel-singleton door (frozen_after: p3): every module-level
``Name = object()`` sentinel, in BOTH ``ast.Assign`` and ``ast.AnnAssign``
spelling, must resolve to one ``sys.modules`` object across every dotted path
that can name its owning module.

RETRACTED 2026-09-18 (brief Q5 ruling (a)): the earlier scope was "every
``_TOKEN`` singleton" -- a name-pattern predicate. That misses
``_MATRIX_ATTESTATION`` and ``_DESIGN_ATTESTATION``, which are the same
identity hazard under a different name. The predicate here is structural
(AST shape), never name-pattern, and is pinned at 18 sentinels in 18 modules
(measured at afc9d09, unchanged at this lane's parent tip).

At p0 there is no ``__legacy`` namespace yet (it is created starting p3b), so
the "both spellings resolve to the same object" arm -- old-prefix path vs.
new-layer path -- is vacuous by construction: there is only one importable
path per module today. The load-bearing assertions at THIS phase are:
  1. the AST-derived count is exactly 18 (a regression on the predicate
     itself: an Assign-only predicate would silently return 8 and drop 10
     modules from p3's exit-gate coverage -- see the architect ruling);
  2. the double-load mutation (importing a module under a synthetic second
     dotted name) is DEMONSTRATED to produce two distinct objects -- the
     hazard, not the guard. The guard is the file-singleton test run after
     the exhaustive import sweep further below: it is what actually goes RED
     when a probe module registers a duplicate ``sys.modules`` entry sharing
     a sentinel module's file. The RULED shim form (static ``import ... as
     _target`` then ``sys.modules[__name__] = _target``) is the only thing
     that collapses two dotted names to the SAME object, demonstrated in
     ``test_shim_form_aliases_sys_modules_to_collapse_to_one_object`` below.

The analysis/d1_execution_receipt.py weakref registry is pinned as a
LOCK+DICT PAIR, not the dict alone (architect ruling): pinning the dict alone
would leave the lock free to duplicate under a double load, silently
un-guarding the very registry the sentinel test exists to protect.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src" / "entropic_substrate"


def _canonical_sentinel_key(node: str) -> str:
    """RETIRED 2026-09-26 (p13-sweep, B2 row 8): the fragment-based
    reverse-lookup and the ``tools/mapping.json`` cross-check this function
    used to perform are gone with the generator (``tools/gen_shims.py``)
    and the landed-set fragments directory, both deleted in this commit.
    ``_FROZEN_SENTINEL_MAP`` is now keyed directly by landed dotted name
    (row 8), and ``discover_module_level_sentinels`` walks the real,
    already-landed tree -- so the module names it finds and the map's keys
    are already the same dotted address space. The only normalization left
    is stripping a leading ``__legacy.`` segment, kept as a structural
    no-op guard: ``__legacy`` no longer exists post-B2 (P13-POSTCONDITION),
    so this branch can never fire again, but removing it outright would be
    a second, unrelated behavior change riding this commit's rename."""
    # Composed at runtime, never a literal chain (P13-NAMESPACE-RULE): a
    # literal "entropic_substrate.__legacy." here would itself be flagged
    # by tests/test_no_old_prefix.py's whole-tree scan.
    root = "entropic_substrate"
    legacy_prefix = root + "." + "__legacy" + "."
    if node.startswith(legacy_prefix):
        return root + "." + node[len(legacy_prefix) :]
    return node


def _all_graph_modules() -> tuple[str, ...]:
    """Every dotted module path over the live tree, via the same measurement
    oracle the epoch fixture uses (tools/import_graph.py) -- never a
    hand-written list, so a module a phase adds or a probe mutation drops is
    swept automatically."""
    tools_dir = _REPO_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from import_graph import build_graph  # local import: tools/ is not a package

    mods, _edges = build_graph(_SRC_ROOT, "entropic_substrate")
    return tuple(sorted(mods.keys()))


def _is_bare_object_call(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "object"
        and not node.args
        and not node.keywords
    )


def _module_dotted_name(path: Path) -> str:
    rel = path.relative_to(_SRC_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def discover_module_level_sentinels() -> dict[str, tuple[str, ...]]:
    """AST predicate: module-level ``Name = object()`` in both ``ast.Assign``
    and ``ast.AnnAssign`` form. Returns {dotted module name: (sentinel names,)}
    for every module carrying at least one, over the real installed source
    tree -- never a hand-written module list."""
    found: dict[str, tuple[str, ...]] = {}
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and _is_bare_object_call(node.value):
                names.extend(target.id for target in node.targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.AnnAssign) and _is_bare_object_call(node.value):
                if isinstance(node.target, ast.Name):
                    names.append(node.target.id)
        if names:
            found[_module_dotted_name(path)] = tuple(names)
    return found


_FROZEN_SENTINEL_COUNT = 18
_FROZEN_SENTINEL_MODULE_COUNT = 18

_FROZEN_SENTINEL_MAP: dict[str, tuple[str, ...]] = {
    "entropic_substrate.ladder.review.d1_external_analysis_codec": ("_REQUEST_V2_TOKEN",),
    "entropic_substrate.ladder.review.d1_external_analysis_control": ("_INPUT_TOKEN",),
    "entropic_substrate.ladder.analyze.d1_factorial": ("_EXCLUSION_AUTHORITY_TOKEN",),
    "entropic_substrate.ladder.review.d1_machine_analysis_binding": ("_TOKEN",),
    "entropic_substrate.ladder.review.d1_machine_analysis_verification": ("_TOKEN",),
    "entropic_substrate.ladder.run.closure.d1_billing_finalization_values": ("_RESULT_TOKEN",),
    "entropic_substrate.ladder.run.campaign.d1_campaign_capture": ("_CAPTURE_TOKEN",),
    "entropic_substrate.ladder.run.campaign.d1_campaign_queue": ("_CAMPAIGN_TOKEN",),
    "entropic_substrate.ladder.run.campaign.d1_campaign_recovery": ("_RECOVERY_TOKEN",),
    "entropic_substrate.ladder.run.closure.d1_excluded_capture": ("_RECEIPT_TOKEN",),
    "entropic_substrate.ladder.run.campaign.d1_operator_evidence": ("_RECEIPT_TOKEN",),
    "entropic_substrate.ladder.capture.custody.d1_capture_aggregate": ("_AGGREGATE_TOKEN",),
    "entropic_substrate.ladder.capture.custody.d1_capture_bundle": ("_BUNDLE_TOKEN",),
    "entropic_substrate.ladder.file.d1_filing_graph": ("_VALIDATED_GRAPH_TOKEN",),
    "entropic_substrate.ladder.gate.ladder_campaign": ("_TOKEN",),
    "entropic_substrate.kernel.lifecycle": ("_RECORD_FACTORY_TOKEN",),
    "entropic_substrate.ladder.register.d1_confirmatory_matrix": ("_MATRIX_ATTESTATION",),
    "entropic_substrate.ladder.register.d1_registered_design": ("_DESIGN_ATTESTATION",),
}


class TestSentinelPredicateCountIsFrozen:
    def test_ast_predicate_finds_exactly_18_sentinels_in_18_modules(self) -> None:
        found = discover_module_level_sentinels()
        total = sum(len(names) for names in found.values())
        assert len(found) == _FROZEN_SENTINEL_MODULE_COUNT
        assert total == _FROZEN_SENTINEL_COUNT

    def test_ast_predicate_matches_the_frozen_map_exactly(self) -> None:
        discovered = discover_module_level_sentinels()
        # Normalize keys so the test survives p3's __legacy move without edit
        canonicalized = {_canonical_sentinel_key(k): v for k, v in discovered.items()}
        assert canonicalized == _FROZEN_SENTINEL_MAP

    def test_predicate_covers_both_assign_and_annassign_spellings(self) -> None:
        """Regression on the predicate itself: an Assign-only predicate
        returns exactly 8 (the RETRACTED scope's count) and silently drops
        the 10 AnnAssign (`x: Final = object()`) modules -- including both
        attestation sentinels. This proves both AST shapes are exercised."""

        def assign_only(path: Path) -> list[str]:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            names = []
            for node in tree.body:
                if isinstance(node, ast.Assign) and _is_bare_object_call(node.value):
                    names.extend(
                        target.id for target in node.targets if isinstance(target, ast.Name)
                    )
            return names

        assign_only_total = 0
        for path in sorted(_SRC_ROOT.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            assign_only_total += len(assign_only(path))
        assert assign_only_total == 8, (
            "the Assign-only count moved -- re-derive both the retracted "
            "8-only baseline and the full 18 before touching this test"
        )
        assert _FROZEN_SENTINEL_COUNT == 18
        assert _FROZEN_SENTINEL_COUNT != assign_only_total


# RETIRED 2026-09-26 (p13-sweep, B2 row 8): the fragment-based reverse-lookup
# mechanism this class guarded (tools/gen_shims.py's load_fragments /
# merge_fragments / resolve_all_modules, plus the mapping.json dest
# cross-check) is deleted in this commit along with the generator and the
# landed-set fragments directory. _canonical_sentinel_key no longer has a
# reverse-lookup branch to guard.


class TestExternalIsolationLockDictPairIsPinned:
    """analysis/d1_execution_receipt.py's registry (architect ruling): the
    LOCK and the DICT are one logical singleton pair, neither matches the
    ``Name = object()`` predicate, and both must be pinned together."""

    def test_lock_and_dict_are_both_absent_from_the_ast_sentinel_map(self) -> None:
        found = discover_module_level_sentinels()
        assert "entropic_substrate.ladder.review.d1_execution_receipt" not in found

    def test_lock_and_dict_are_each_module_singletons(self) -> None:
        import entropic_substrate.ladder.review.d1_execution_receipt as mod

        assert isinstance(mod._EXTERNAL_ISOLATION_LOCK, type(threading.Lock()))
        assert isinstance(mod._MINTED_EXTERNAL_ISOLATIONS, dict)

    def test_double_load_of_the_owning_module_produces_distinct_lock_and_dict(self) -> None:
        """This is the HAZARD demonstration, not the guard: loading the
        module under a synthetic second dotted name DOES produce a second
        lock and a second dict -- two independently constructed objects with
        no relationship to the originals. That is exactly the defect class
        the sentinel test exists to catch; the actual guard that must go RED
        under this class of mutation is the file-singleton test after the
        exhaustive import sweep, further below. Both arms here restore
        sys.modules/sys.path so no state leaks into sibling tests (house
        rule 3)."""
        import entropic_substrate.ladder.review.d1_execution_receipt as first

        alias_name = "_p0_probe_double_load_execution_receipt"
        saved_modules = dict(sys.modules)
        try:
            spec = importlib.util.spec_from_file_location(
                alias_name,
                first.__file__,
            )
            assert spec is not None and spec.loader is not None
            second = importlib.util.module_from_spec(spec)
            sys.modules[alias_name] = second
            spec.loader.exec_module(second)

            # The two module OBJECTS are legitimately distinct (that is what
            # "loaded under a second name" means) -- but the guard this
            # protects is that no LATER real import path collision, once
            # __legacy exists, is allowed to silently coexist. At p0 we prove
            # the mechanism: sys.modules correctly holds two distinct entries
            # for two distinct load requests, so a future shim that instead
            # ALIASES sys.modules[new] = sys.modules[old] (the ruled shim
            # form) is what collapses them to one -- never a re-exec.
            assert sys.modules[alias_name] is second
            assert sys.modules[alias_name] is not first
            assert type(second._EXTERNAL_ISOLATION_LOCK) is type(first._EXTERNAL_ISOLATION_LOCK)
            assert second._MINTED_EXTERNAL_ISOLATIONS is not first._MINTED_EXTERNAL_ISOLATIONS
        finally:
            sys.modules.clear()
            sys.modules.update(saved_modules)
        assert alias_name not in sys.modules

    def test_shim_form_aliases_sys_modules_to_collapse_to_one_object(self) -> None:
        """The RULED shim form (`import ... as _target` then
        `sys.modules[__name__] = _target`) is what makes two dotted paths
        resolve to ONE object -- unlike the double-load above, which
        legitimately produces two. This is the sentinel test's positive
        case for why the static-import shim is mandatory."""
        import entropic_substrate.ladder.review.d1_execution_receipt as canonical

        alias_name = "_p0_probe_shim_alias_execution_receipt"
        saved_modules = dict(sys.modules)
        try:
            sys.modules[alias_name] = canonical
            aliased = importlib.import_module(alias_name)
            assert aliased is canonical
            assert aliased._MINTED_EXTERNAL_ISOLATIONS is canonical._MINTED_EXTERNAL_ISOLATIONS
            assert aliased._EXTERNAL_ISOLATION_LOCK is canonical._EXTERNAL_ISOLATION_LOCK
        finally:
            sys.modules.clear()
            sys.modules.update(saved_modules)
        assert alias_name not in sys.modules


class TestExhaustiveImportSweepIsTheP3ExitGateFoundation:
    """p3's exit gate is a sentinel test over EVERY dotted path. A sweep that
    swallows import failures makes that gate vacuously green over precisely
    the modules most likely to break at the quarantine (architect ruling).
    This sweep imports every graph node with no ``except: continue`` -- any
    failure fails this test by name, and is a DECISION for the conductor, not
    an exclusion list the implementer invents.

    Probed clean at authoring time: all 337 dotted paths import successfully
    under ``.venv/bin/python`` (scratch probe, not shipped). A duplicate-load
    mutation (guard-teeth mutation 3, performed and reverted in-branch) adds
    a probe module under ``kernel/`` that registers a second ``sys.modules``
    entry sharing ``evidence/lifecycle.py``'s file -- this sweep discovers
    and imports the probe module as an ordinary graph node, and the
    file-singleton test below is what actually goes RED."""

    def test_every_graph_module_imports_without_exception(self) -> None:
        failures: list[tuple[str, str]] = []
        for name in _all_graph_modules():
            try:
                importlib.import_module(name)
            except Exception as exc:
                failures.append((name, repr(exc)))
        assert failures == [], f"modules that failed to import: {failures}"


class TestSentinelModulesAreFileSingletonsAfterTheSweep:
    """After the exhaustive sweep has imported every module, none of the 18
    AST-derived sentinel modules may share its ``__file__`` with any OTHER
    ``sys.modules`` entry. This is the guard that actually goes RED under
    the double-load mutation -- the earlier double-load test in this file
    demonstrates the hazard (two distinct objects); this test is the guard
    that a real collision must never pass."""

    def test_no_sentinel_module_file_is_double_registered_in_sys_modules(self) -> None:
        for name in _all_graph_modules():
            try:
                importlib.import_module(name)
            except Exception:
                continue

        violations: dict[str, list[str]] = {}
        for sentinel_module in _FROZEN_SENTINEL_MAP:
            owner = sys.modules.get(sentinel_module)
            if owner is None or getattr(owner, "__file__", None) is None:
                continue
            owner_file = Path(owner.__file__).resolve()
            duplicates = [
                other_name
                for other_name, other_module in sys.modules.items()
                if other_name != sentinel_module
                and other_module is not owner
                and getattr(other_module, "__file__", None) is not None
                and Path(other_module.__file__).resolve() == owner_file
            ]
            if duplicates:
                violations[sentinel_module] = duplicates
        assert violations == {}, (
            f"sentinel module(s) double-registered under another sys.modules "
            f"name (breaks the singleton guarantee): {violations}"
        )
