"""Pytest for the landed<->__legacy coupling bound (p3-quarantine contract
row: "asserts from the AST graph (tools/import_graph.py) that edges between
landed packages and __legacy are <= the merged budget, and that EVERY shim
carries the static-import form").

CLASSIFICATION, not `package_of()` (architect constraint, design gate
2026-09-20): `tools/import_graph.py`'s own `package_of()` collapses a
dotted name to its SECOND component or `"ROOT"` -- `package_of(
"entropic_substrate.kernel")` returns `"ROOT"`, the same bucket as every
top-level module still quarantined under `entropic_substrate.<leaf>.py`.
A classifier built on `package_of()` would therefore lump the (still
empty) landed layer packages in with the entire unmoved tree and measure
the wrong set. This file classifies by dotted PREFIX instead, over the
`(mods, edges)` `tools/import_graph.build_graph()` already returns.

EQUALITY, not `<=` (architect constraint, MEASURED trap): at p3 the five
landed layers are empty `__init__.py` files, so the measured landed ->
__legacy edge count is 0. A budget of 0 checked with `edges <= budget` is
green whether the classifier works or classifies nothing at all, and is
insensitive to a budget raised ABOVE the true measured count (the brief's
own required guard-teeth arm). `measured == committed_budget` catches
both directions: an uncounted real edge (measured drifts above budget)
and an inflated budget (budget drifts above measured) are the same
failure under `==`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:  # tools/ is not a package; import the measurement oracles directly
    sys.path.insert(0, str(TOOLS))
import gen_shims  # noqa: E402
import import_graph  # noqa: E402

pytestmark = pytest.mark.unit

LANDED_LAYER_PREFIXES = (
    "entropic_substrate.kernel",
    "entropic_substrate.core",
    "entropic_substrate.ladder",
    "entropic_substrate.historical",
    "entropic_substrate.composition",
)
LEGACY_PREFIX = "entropic_substrate.__legacy"


def _is_landed(mod: str) -> bool:
    """A module is LANDED iff its dotted name is exactly one of the five
    layer package roots, or a descendant of one -- prefix-matched on a
    dot boundary so ``entropic_substrate.kernelling`` (a hypothetical
    sibling) is never mistaken for a child of ``entropic_substrate.kernel``."""
    return any(mod == p or mod.startswith(p + ".") for p in LANDED_LAYER_PREFIXES)


def _is_legacy(mod: str) -> bool:
    return mod == LEGACY_PREFIX or mod.startswith(LEGACY_PREFIX + ".")


def count_landed_to_legacy_edges(edges: list[tuple[str, str]]) -> int:
    """Edges whose SOURCE is a landed-layer module and whose DESTINATION is
    under the quarantine -- the one direction DN-P1's bound is about (a
    landed module reaching back into __legacy). Both classifiers are dotted
    -PREFIX based, never `import_graph.package_of()` (see module docstring)."""
    return sum(1 for src, dst in edges if _is_landed(src) and _is_legacy(dst))


def load_committed_budget(fragments_dir: Path, sequence_file: Path) -> int:
    """The SAME lookup `gen_shims.py --print-budget --sequence-file` performs
    (constraint 9, design gate 2026-09-20): highest-ordered fragment present,
    per the real sprint-manifest sequence -- never a hardcoded number here,
    so this test tracks the committed fragment file, not a copy of it."""
    fragments = gen_shims.load_fragments(fragments_dir)
    sequence = json.loads(sequence_file.read_text())["sequence"]
    top = gen_shims.highest_ordered_fragment(fragments, sequence)
    return int(top["legacy_edge_budget"])


# ---------------------------------------------------------------------------
# The MEASURED trap, encoded directly: package_of() is provably the wrong
# classifier for this bound.
# ---------------------------------------------------------------------------


def test_package_of_collapses_landed_layer_into_root_MEASURED_trap() -> None:
    """Documents exactly why this file does not use `package_of()`: it
    returns `"ROOT"` for a landed-layer top-level package, the same bucket
    every still-quarantined top-level module (`entropic_substrate.cli`,
    etc.) falls into. A budget check keyed on this would not distinguish
    'kernel imports __legacy' from 'an ordinary top-level module does'."""
    assert import_graph.package_of("entropic_substrate.kernel") == "ROOT"
    assert import_graph.package_of("entropic_substrate.cli") == "ROOT"
    # The trap: package_of() cannot tell these two apart, but _is_landed can.
    assert _is_landed("entropic_substrate.kernel") is True
    assert _is_landed("entropic_substrate.cli") is False


def test_is_landed_matches_only_the_five_layer_roots_and_descendants() -> None:
    for prefix in LANDED_LAYER_PREFIXES:
        assert _is_landed(prefix) is True
        assert _is_landed(f"{prefix}.sub.module") is True
    assert _is_landed("entropic_substrate.kernelling") is False  # dot-boundary, not startswith
    assert _is_landed("entropic_substrate.__legacy") is False
    assert _is_landed("entropic_substrate") is False


def test_is_legacy_matches_the_quarantine_root_and_descendants() -> None:
    assert _is_legacy("entropic_substrate.__legacy") is True
    assert _is_legacy("entropic_substrate.__legacy.kernel_stuff") is True
    assert _is_legacy("entropic_substrate.__legacyish") is False  # dot-boundary
    assert _is_legacy("entropic_substrate.kernel") is False


# ---------------------------------------------------------------------------
# The real measurement, against the committed fragment and the real tree.
# ---------------------------------------------------------------------------


def test_measured_landed_to_legacy_edges_equals_committed_budget() -> None:
    """The ACCEPTANCE-level assertion: build the AST graph over the real
    `src/entropic_substrate`, count landed->__legacy edges by dotted
    prefix, and require EXACT equality with the fragment's own
    `legacy_edge_budget` (both read live -- see module docstring for why
    `==`, not `<=`)."""
    _mods, edges = import_graph.build_graph(
        REPO_ROOT / "src" / "entropic_substrate", "entropic_substrate"
    )
    measured = count_landed_to_legacy_edges(edges)
    budget = load_committed_budget(
        REPO_ROOT / ".claude" / "workflows" / "landed-set",
        REPO_ROOT / ".claude" / "workflows" / "sprint-manifest.json",
    )
    assert measured == budget, (
        f"measured landed->__legacy edges ({measured}) != committed "
        f"legacy_edge_budget ({budget}) -- either an uncounted coupling "
        "edge exists, or the committed budget no longer matches reality"
    )


# ---------------------------------------------------------------------------
# Guard teeth (house rule 5, both arms): the counting function must be
# LOAD-BEARING, not merely coincidentally correct at budget 0. Both
# mutations run against a synthetic, self-contained module/edge set (no
# real file touched, no subprocess) so they are hermetic and fast, while
# still exercising the SAME classify-and-count function the real
# measurement above calls.
# ---------------------------------------------------------------------------


def test_injected_landed_to_legacy_edge_is_counted_RED_at_zero_budget() -> None:
    """RED-first: fabricated landed->__legacy edges (the shape a real
    `import entropic_substrate.__legacy.x` inside `kernel/__init__.py`
    would add to the graph) must be COUNTED, and checked against the real
    committed budget it must fail equality -- proving the guard is
    load-bearing, not vacuously green because nothing was ever reachable
    (house rule 5). Injects `budget + 1` edges (CodeRabbit 2026-09-21),
    never a hardcoded `1`: that guarantees `measured != budget` by
    construction, so the mutation stays RED even if a later phase's fragment
    ever raises the committed budget above 0, instead of accidentally
    depending on today's value."""
    budget = load_committed_budget(
        REPO_ROOT / ".claude" / "workflows" / "landed-set",
        REPO_ROOT / ".claude" / "workflows" / "sprint-manifest.json",
    )
    edges = [
        ("entropic_substrate.kernel", f"entropic_substrate.__legacy.some_leaf_{i}")
        for i in range(budget + 1)
    ]
    measured = count_landed_to_legacy_edges(edges)
    assert measured == budget + 1
    assert measured != budget, (
        "this mutation is supposed to break equality against the committed "
        f"budget ({budget}) -- if it did not, the counting function is not "
        "load-bearing"
    )


def test_no_injected_edge_measures_zero_GREEN_control() -> None:
    """Control for the mutation above: with NO fabricated edge, the
    classify-and-count function itself returns 0 -- proving the RED case
    above is caused by the injection, not by the counting function being
    broken in general.

    RETRACTION (house rule 4, p3b): this control previously also asserted
    ``measured == load_committed_budget(...)``, coupling it to the REAL
    committed fragment's budget rather than to the synthetic input this test
    actually constructs. That held only by coincidence, while every fragment
    on disk still carried ``legacy_edge_budget: 0`` (p3-quarantine, before
    any phase had landed real modules). p3b legitimately measures and commits
    a non-zero budget (2 -- see
    ``test_measured_landed_to_legacy_edges_equals_committed_budget``, the
    ACCEPTANCE-level assertion that already covers the real tree against the
    real committed value), so re-asserting it here against a synthetic
    zero-edge input would fail for a reason that has nothing to do with this
    test's own subject."""
    edges: list[tuple[str, str]] = []
    measured = count_landed_to_legacy_edges(edges)
    assert measured == 0


def test_edge_in_the_other_direction_is_not_counted_GREEN_control() -> None:
    """A __legacy -> landed edge (the opposite direction) is NOT what this
    bound measures -- DN-P1 bounds landed reaching back INTO the
    quarantine, not the quarantine being imported by (not-yet-existing)
    landed code. Confirms the classifier is direction-sensitive."""
    edges = [("entropic_substrate.__legacy.some_leaf", "entropic_substrate.kernel")]
    assert count_landed_to_legacy_edges(edges) == 0


def test_edge_between_two_ordinary_quarantined_modules_is_not_counted_GREEN_control() -> None:
    """An ordinary __legacy -> __legacy edge (the overwhelming majority of
    the real graph today) is neither landed nor a landed->legacy edge."""
    edges = [
        ("entropic_substrate.__legacy.a", "entropic_substrate.__legacy.b"),
        ("entropic_substrate.cli", "entropic_substrate.__legacy.telemetry"),  # ROOT, not landed
    ]
    assert count_landed_to_legacy_edges(edges) == 0


def test_raising_the_budget_above_the_real_measured_count_breaks_equality(tmp_path: Path) -> None:
    """The brief's OWN required arm, restated through the SAME loader chain
    `test_measured_landed_to_legacy_edges_equals_committed_budget` calls --
    not a hardcoded `measured != measured + 5` (root-cause fix, CodeRabbit
    2026-09-21: any two distinct local ints satisfy that trivially; it
    never touches `load_committed_budget`, `load_fragments` or
    `highest_ordered_fragment`, so it could not actually catch a
    regression in any of them). A real, isolated fragments dir + sequence
    file with `legacy_edge_budget` inflated ABOVE the real measured count
    is pushed through `load_committed_budget`, proving a `<=` form would
    have stayed green here (measured <= inflated) -- exactly why `==` was
    chosen (module docstring)."""
    _mods, edges = import_graph.build_graph(
        REPO_ROOT / "src" / "entropic_substrate", "entropic_substrate"
    )
    measured = count_landed_to_legacy_edges(edges)
    inflated_budget = measured + 5

    frags_dir = tmp_path / "landed-set"
    frags_dir.mkdir()
    (frags_dir / "p3-quarantine.json").write_text(
        json.dumps({"phase": "p3-quarantine", "modules": {}, "legacy_edge_budget": inflated_budget})
    )
    sequence_file = tmp_path / "sprint-manifest.json"
    sequence_file.write_text(json.dumps({"sequence": ["p3-quarantine"]}))

    loaded = load_committed_budget(frags_dir, sequence_file)
    assert loaded == inflated_budget
    assert measured <= loaded  # the `<=` form's blind spot, made explicit
    assert measured != loaded, (
        "raising a phase's budget above its measured count was supposed to "
        f"break equality (measured={measured}, inflated budget={loaded}) -- "
        "if it did not, the equality check is not load-bearing"
    )


# ---------------------------------------------------------------------------
# Contract row, second half: "EVERY shim carries the static-import form."
# ---------------------------------------------------------------------------


def test_every_emitted_shim_is_the_static_import_form_not_string_only() -> None:
    """Every `.py` shim the generator resolves from the real fragments parses to
    the exact-form AST rebind (`verify_shim_source`) -- never a
    string-only `importlib.import_module` shim, which is invisible to
    grimp and makes the landed-layers contract vacuously green for any
    edge routed through it (see `.importlinter`'s own recorded probe)."""
    # FOR-1539 (2026-09-22): the emitted-files manifest is untracked derived state, so a
    # fresh checkout has none. Derive the shim set the generator WOULD emit from the
    # fragments + mapping on disk instead of reading a file that only exists after a
    # local generate() run (measured RED in a fresh worktree at 1c10969).
    fragments_dir = REPO_ROOT / ".claude" / "workflows" / "landed-set"
    mapping = gen_shims.load_mapping(REPO_ROOT / "tools" / "mapping.json")
    merged = gen_shims.merge_fragments(gen_shims.load_fragments(fragments_dir), mapping)
    out_dir = REPO_ROOT / "src"
    # FOR-1547: the emitted static-import set now also includes the FORWARDER
    # each landed row gets at the `__legacy` address its move vacated. Same
    # exact form, so it belongs in this arm; forwarders come only from landed
    # rows, and a landed row is always fragment-named (the derived default is
    # landed:false), so `merged` is the complete input for both maps.
    expected = gen_shims.compute_expected_shim_map(merged, out_dir)
    expected |= gen_shims.compute_expected_forwarder_map(merged, out_dir)
    assert expected, "the fragments resolve to zero shims -- nothing is quarantined?"
    bad: list[tuple[str, str]] = []
    for rel in sorted(expected):
        if not rel.endswith(".py"):
            continue
        p = out_dir / rel
        if not p.is_file():
            bad.append((rel, "expected shim file is missing"))
            continue
        ok, reason = gen_shims.verify_shim_source(p.read_text())
        if not ok:
            bad.append((rel, reason))
    assert not bad, f"{len(bad)} shim(s) are not in the required static-import form: {bad}"


def test_a_hand_built_string_only_shim_is_rejected_by_the_same_check_GREEN_mirror() -> None:
    """Mirror of the check above, against a deliberately illegal string-only
    shim -- proves the pass above is discriminating, not vacuous."""
    illegal = 'import importlib\nsys.modules[__name__] = importlib.import_module("entropic_substrate.__legacy.x")\n'
    ok, _ = gen_shims.verify_shim_source(illegal)
    assert ok is False
