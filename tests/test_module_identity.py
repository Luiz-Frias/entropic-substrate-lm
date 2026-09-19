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
import json
import sys
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src" / "entropic_substrate"


def _mapping() -> dict:
    """Load tools/mapping.json to support sentinel key normalization."""
    with (_REPO_ROOT / "tools" / "mapping.json").open() as f:
        return json.load(f)


def _canonical_sentinel_key(node: str, *, fragments_dir: Path | None = None) -> str:
    """Normalize a module name for sentinel test comparison by:
    1. Stripping exactly one leading __legacy. segment if present.
    2. If the result is a LANDED module's exact new dotted name, return its
       mapping.json source key instead.
    3. Else return the result as-is.

    This allows test_module_identity to stay green when a later phase moves
    modules under __legacy or on into a landed layer, without needing to
    edit this p0-owned file.

    RETRACTED (gate-check finding, 2026-09-19): step 2 previously compared
    the FULL normalized name against mapping.json's `dest` with `==`.
    `dest` is a destination PACKAGE, shared by up to 100 source modules
    (measured: `entropic_substrate.ladder.run` alone), never the module's
    own final leaf path -- concatenating `dest + "." + basename` to guess
    that path is exactly the invented rule tools/gen_shims.py's own
    docstring refuses to add, because no primary source pins it. That made
    the `==` match succeed only for the one sentinel whose dest happens to
    be a singleton (`evidence.lifecycle` -> `kernel.lifecycle`, shared_by
    1) and silently fail for the other 17.

    The authoritative source for a landed module's exact new dotted name is
    the landed-set fragments' `target` field
    (`.claude/workflows/landed-set/<phase>.json`, merged via
    tools/gen_shims.py's own `load_fragments`/`merge_fragments`/
    `resolve_all_modules`) -- the same oracle the shim generator itself
    reads to re-point a shim. Reverse-lookup goes through THAT, with
    mapping.json's `dest` kept only as a fail-loud cross-check (a landed
    target that disagrees with its mapping row's dest package is a defect
    to raise, never to silently prefer one source over the other)."""
    # Step 1: strip __legacy. prefix
    legacy_prefix = "entropic_substrate.__legacy."
    if node.startswith(legacy_prefix):
        normalized = "entropic_substrate." + node[len(legacy_prefix) :]
    else:
        normalized = node

    if normalized in _FROZEN_SENTINEL_MAP:
        return normalized

    # Step 2: reverse-lookup a LANDED module through the fragments' `target`
    # (never through a `dest`-equality guess). At p0 (and any phase before
    # the fragments directory exists) this is vacuous by construction.
    if fragments_dir is None:
        fragments_dir = _REPO_ROOT / ".claude" / "workflows" / "landed-set"
    if not fragments_dir.is_dir():
        return normalized

    tools_dir = _REPO_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import gen_shims  # local import: tools/ is not a package

    mapping = _mapping()
    fragments = gen_shims.load_fragments(fragments_dir)
    merged = gen_shims.merge_fragments(fragments, mapping)
    resolved = gen_shims.resolve_all_modules(merged, mapping)

    for source_key in _FROZEN_SENTINEL_MAP:
        entry = resolved.get(source_key)
        if entry is None or not entry.get("landed", False):
            continue
        if entry["target"] != normalized:
            continue
        dest = mapping["modules"].get(source_key, {}).get("dest", "")
        if not (normalized == dest or normalized.startswith(dest + ".")):
            raise AssertionError(
                f"{source_key}: landed-set target {normalized!r} disagrees "
                f"with mapping.json dest {dest!r} for this sentinel module "
                "-- the fragment and mapping.json sources have drifted "
                "apart, this is a defect to raise, not to paper over"
            )
        return source_key

    # Step 3: return normalized
    return normalized


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
    "entropic_substrate.analysis.d1_external_analysis_codec": ("_REQUEST_V2_TOKEN",),
    "entropic_substrate.analysis.d1_external_analysis_control": ("_INPUT_TOKEN",),
    "entropic_substrate.analysis.d1_factorial": ("_EXCLUSION_AUTHORITY_TOKEN",),
    "entropic_substrate.analysis.d1_machine_analysis_binding": ("_TOKEN",),
    "entropic_substrate.analysis.d1_machine_analysis_verification": ("_TOKEN",),
    "entropic_substrate.d1_billing_finalization_values": ("_RESULT_TOKEN",),
    "entropic_substrate.d1_campaign_capture": ("_CAPTURE_TOKEN",),
    "entropic_substrate.d1_campaign_queue": ("_CAMPAIGN_TOKEN",),
    "entropic_substrate.d1_campaign_recovery": ("_RECOVERY_TOKEN",),
    "entropic_substrate.d1_excluded_capture": ("_RECEIPT_TOKEN",),
    "entropic_substrate.d1_operator_evidence": ("_RECEIPT_TOKEN",),
    "entropic_substrate.evidence.d1_capture_aggregate": ("_AGGREGATE_TOKEN",),
    "entropic_substrate.evidence.d1_capture_bundle": ("_BUNDLE_TOKEN",),
    "entropic_substrate.evidence.d1_filing_graph": ("_VALIDATED_GRAPH_TOKEN",),
    "entropic_substrate.evidence.ladder_campaign": ("_TOKEN",),
    "entropic_substrate.evidence.lifecycle": ("_RECORD_FACTORY_TOKEN",),
    "entropic_substrate.experiments.d1_confirmatory_matrix": ("_MATRIX_ATTESTATION",),
    "entropic_substrate.experiments.d1_registered_design": ("_DESIGN_ATTESTATION",),
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


class TestCanonicalSentinelKeyReverseLookupSurvivesSharedDestPackages:
    """Guard teeth for the reverse-lookup mechanism (gate-check finding,
    2026-09-19): mapping.json's `dest` is a destination PACKAGE, shared by
    up to 100 source modules (measured), so a bare `dest == normalized`
    match is structurally broken for every sentinel except the one
    (`evidence.lifecycle` -> `kernel.lifecycle`) whose dest happens to be a
    singleton. These mutations exercise the fragment-based reverse lookup
    directly, via a throwaway fragments dir under ``tmp_path`` -- no real
    source file ever moves (p0's Non-goals: zero modules move)."""

    _SHARED_DEST_SOURCE = "entropic_substrate.d1_campaign_capture"
    _SHARED_DEST_TARGET = "entropic_substrate.ladder.run.d1_campaign_capture"
    _SINGLETON_SOURCE = "entropic_substrate.evidence.lifecycle"
    _SINGLETON_TARGET = "entropic_substrate.kernel.lifecycle"

    @staticmethod
    def _write_fragment(tmp_path: Path, modules: dict) -> Path:
        frag_dir = tmp_path / "landed-set"
        frag_dir.mkdir(exist_ok=True)
        (frag_dir / "p9-probe.json").write_text(
            json.dumps({"phase": "p9-probe", "modules": modules})
        )
        return frag_dir

    def test_shared_dest_package_resolves_via_fragment_target(self, tmp_path: Path) -> None:
        """d1_campaign_capture's dest (`ladder.run`) is shared by 100 source
        modules (measured) -- a `dest == normalized` guess can never match
        its landed leaf name. GUARD-TEETH GREEN: with the landed-set
        fragment naming the exact target, the reverse lookup resolves to
        the original mapping.json source key."""
        frag_dir = self._write_fragment(
            tmp_path,
            {self._SHARED_DEST_SOURCE: {"target": self._SHARED_DEST_TARGET, "landed": True}},
        )
        assert (
            _canonical_sentinel_key(self._SHARED_DEST_TARGET, fragments_dir=frag_dir)
            == self._SHARED_DEST_SOURCE
        )

    def test_landed_looking_name_with_no_claiming_fragment_does_not_falsely_resolve(
        self, tmp_path: Path
    ) -> None:
        """GUARD-TEETH RED (discriminating arm): the identical dotted name
        the previous test resolves, but with NO fragment claiming it, must
        NOT resolve to the sentinel's old key. This is what proves the
        fragment lookup -- not the shape of the dotted name -- is
        load-bearing; a predicate that pattern-matched on name shape alone
        would pass this test wrongly."""
        frag_dir = self._write_fragment(tmp_path, {})
        assert (
            _canonical_sentinel_key(self._SHARED_DEST_TARGET, fragments_dir=frag_dir)
            == self._SHARED_DEST_TARGET
        )

    def test_singleton_dest_module_still_resolves(self, tmp_path: Path) -> None:
        """The one sentinel the OLD `dest == normalized` code happened to
        get right (`evidence.lifecycle`, dest shared_by 1) must keep
        resolving under the fragment-based lookup too -- the fix must not
        regress the one case that used to work."""
        frag_dir = self._write_fragment(
            tmp_path,
            {self._SINGLETON_SOURCE: {"target": self._SINGLETON_TARGET, "landed": True}},
        )
        assert (
            _canonical_sentinel_key(self._SINGLETON_TARGET, fragments_dir=frag_dir)
            == self._SINGLETON_SOURCE
        )

    def test_target_disagreeing_with_mapping_dest_fails_loud(self, tmp_path: Path) -> None:
        """A landed-set target that disagrees with mapping.json's own dest
        package for that source module is a drift between the two
        authoritative sources -- raised, never silently resolved either
        way (epistemic hygiene: never fill the gap)."""
        wrong_target = "entropic_substrate.core.somewhere_else"
        frag_dir = self._write_fragment(
            tmp_path,
            {self._SHARED_DEST_SOURCE: {"target": wrong_target, "landed": True}},
        )
        with pytest.raises(AssertionError, match=r"disagrees with mapping\.json dest"):
            _canonical_sentinel_key(wrong_target, fragments_dir=frag_dir)

    def test_absent_fragments_directory_is_vacuous(self) -> None:
        """No landed-set directory yet (true at p0) -- the reverse lookup
        must not attempt to import tools/gen_shims.py or touch the
        filesystem beyond the existence check."""
        missing = Path("/nonexistent-p0-landed-set-probe")
        assert not missing.exists()
        assert (
            _canonical_sentinel_key(self._SHARED_DEST_TARGET, fragments_dir=missing)
            == self._SHARED_DEST_TARGET
        )


class TestExternalIsolationLockDictPairIsPinned:
    """analysis/d1_execution_receipt.py's registry (architect ruling): the
    LOCK and the DICT are one logical singleton pair, neither matches the
    ``Name = object()`` predicate, and both must be pinned together."""

    def test_lock_and_dict_are_both_absent_from_the_ast_sentinel_map(self) -> None:
        found = discover_module_level_sentinels()
        assert "entropic_substrate.analysis.d1_execution_receipt" not in found

    def test_lock_and_dict_are_each_module_singletons(self) -> None:
        import entropic_substrate.analysis.d1_execution_receipt as mod

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
        import entropic_substrate.analysis.d1_execution_receipt as first

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
        import entropic_substrate.analysis.d1_execution_receipt as canonical

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
