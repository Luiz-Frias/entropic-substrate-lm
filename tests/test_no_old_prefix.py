"""Pytest for zero old-prefix (entropic_substrate.<leaf>) imports in landed
packages. This arm was extracted from test_legacy_coupling.py during p3b
(observability rail) to isolate the guard-teeth for the no-old-prefix
invariant (CLAUDE.md, Sacred Invariants).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:  # tools/ is not a package; import the measurement oracles directly
    sys.path.insert(0, str(TOOLS))
import import_graph  # noqa: E402

# Sourced from test_legacy_coupling (sealed this phase, GATE-CHECK RULING
# 2026-09-21: exactly one change already made, nothing else in that file
# moves here) rather than hand-copied a third time, per the architect
# constraint that the five landed prefixes have ONE source. That module has
# no package __init__ (pytest --import-mode=importlib, tests/ un-packaged),
# so this is the same cross-test import pattern test_sinks.py's importers
# already use (e.g. test_tracking_sinks.py: "from test_sinks import
# FakeMlflowClient").
from test_legacy_coupling import LANDED_LAYER_PREFIXES  # noqa: E402

pytestmark = pytest.mark.unit


def _is_landed(mod: str) -> bool:
    """A module is LANDED iff its dotted name is exactly one of the five
    layer package roots, or a descendant of one -- prefix-matched on a dot
    boundary so ``entropic_substrate.kernelling`` (a hypothetical sibling)
    is never mistaken for a child of ``entropic_substrate.kernel`` (same
    dot-boundary form as test_legacy_coupling.py's own ``_is_landed``,
    architect constraint 2026-09-21: the earlier ``mod.startswith(prefix)``
    form here was boundary-less and regressed the classifier fix that
    module's docstring records)."""
    return any(mod == p or mod.startswith(p + ".") for p in LANDED_LAYER_PREFIXES)


_LEGACY_PREFIX = "entropic_substrate.__legacy"


def _is_legacy(mod: str) -> bool:
    """A module is in __legacy iff it IS the quarantine root or a descendant
    of it, dot-boundary matched (same form as test_legacy_coupling.py's own
    ``_is_legacy``): a plain ``startswith`` would mistake a hypothetical
    ``entropic_substrate.__legacyish`` sibling for quarantined code, the
    same boundary hazard ``_is_landed`` above was fixed for."""
    return mod == _LEGACY_PREFIX or mod.startswith(_LEGACY_PREFIX + ".")


# SACRED INVARIANT (p3b): zero old-prefix (entropic_substrate.<legacy-dotted-
# name>) imports in any landed package. count_landed_to_legacy_edges above is
# BLIND to this hazard by construction -- a landed module importing the
# ROOT-LEVEL compat shim (e.g. `entropic_substrate.logging`, still present
# for out-of-envelope importers) is neither a landed->legacy edge (the shim's
# dotted name is not under `entropic_substrate.__legacy`) nor caught by the
# layers contract (the shim sits outside all five layer packages, so
# `.importlinter`'s landed-layers contract never sees it either -- MEASURED,
# `.importlinter`'s own header comment records the same blindness class for
# a string-only shim). This phase is the first with real landed modules, so
# this is the phase where the gap stops being vacuous and needs its own
# arm, landed here (not a new file) per the SAME classify-by-dotted-prefix
# discipline as the rest of this module.
# ---------------------------------------------------------------------------


_ROOT_PACKAGE = "entropic_substrate"


def _is_old_prefix(mod: str) -> bool:
    """A module is an OLD-PREFIX compat shim destination iff it is neither
    under a landed layer prefix nor under `__legacy`, and is not the root
    package itself -- i.e. one of the pre-refactor dotted names gen_shims.py
    still emits a shim for (`entropic_substrate.logging`,
    `entropic_substrate.sinks`, `entropic_substrate.cli`, ...)."""
    return mod != _ROOT_PACKAGE and not _is_landed(mod) and not _is_legacy(mod)


def count_landed_to_old_prefix_edges(edges: list[tuple[str, str]]) -> int:
    """Edges whose SOURCE is landed and whose DESTINATION is an old-prefix
    compat shim -- the hazard neither `count_landed_to_legacy_edges` nor
    `.importlinter`'s landed-layers contract can see (module docstring)."""
    return sum(1 for src, dst in edges if _is_landed(src) and _is_old_prefix(dst))


def test_is_landed_is_dot_boundary_not_prefix_matched() -> None:
    """Guard-tooth for the boundary fix (architect constraint 2026-09-21):
    a hypothetical sibling package sharing a plain string prefix with a
    landed layer root must NOT be classified landed. Before this fix,
    ``_is_landed`` here used bare ``mod.startswith(prefix)`` and this
    assertion was FALSE."""
    assert _is_landed("entropic_substrate.kernelling") is False
    assert _is_landed("entropic_substrate.kernel") is True
    assert _is_landed("entropic_substrate.kernel.runtime.logging") is True


def test_is_legacy_is_dot_boundary_not_prefix_matched() -> None:
    """Same boundary hazard, for the quarantine-root classifier."""
    assert _is_legacy("entropic_substrate.__legacyish") is False
    assert _is_legacy("entropic_substrate.__legacy") is True
    assert _is_legacy("entropic_substrate.__legacy.sinks") is True


def test_is_old_prefix_matches_root_level_shims_not_landed_or_legacy() -> None:
    assert _is_old_prefix("entropic_substrate.logging") is True
    assert _is_old_prefix("entropic_substrate.sinks") is True
    assert _is_old_prefix("entropic_substrate.cli") is True
    assert _is_old_prefix("entropic_substrate") is False  # the root package itself
    assert _is_old_prefix("entropic_substrate.kernel.runtime.logging") is False  # landed
    assert _is_old_prefix("entropic_substrate.__legacy.sinks") is False  # legacy


def test_measured_landed_to_old_prefix_edges_is_zero() -> None:
    """ACCEPTANCE-level assertion, hard zero (not a budget, unlike the
    landed->__legacy bound above) -- every landed module in this phase
    reaches the pre-refactor names it still needs (`MlflowStartupError`,
    the `MLFLOW_*_ENV` residue) through `entropic_substrate.__legacy.<x>`
    directly, never through the root-level compat shim."""
    _mods, edges = import_graph.build_graph(
        REPO_ROOT / "src" / "entropic_substrate", "entropic_substrate"
    )
    measured = count_landed_to_old_prefix_edges(edges)
    assert measured == 0, (
        f"{measured} landed->old-prefix edge(s) found -- a landed module must import "
        "entropic_substrate.__legacy.<x> directly, never the root-level compat shim "
        "entropic_substrate.<x>"
    )


def test_injected_landed_to_old_prefix_edge_is_counted_RED() -> None:
    """Guard-tooth (house rule 5): fabricated landed->old-prefix edges (the
    shape a real `import entropic_substrate.logging` inside a landed module
    would add to the graph) must be COUNTED and must break the hard-zero
    invariant above -- proving the classifier is load-bearing. A real-file
    mutation-and-revert (one landed module importing
    `entropic_substrate.logging` by its old prefix) was additionally
    performed in-branch against this phase's own tip and reverted; both
    arms are quoted in this phase's PR body (house rule 5)."""
    edges = [("entropic_substrate.kernel.runtime.logging", "entropic_substrate.logging")]
    measured = count_landed_to_old_prefix_edges(edges)
    assert measured == 1
    assert measured != 0


def test_old_prefix_edge_from_a_non_landed_module_is_not_counted_GREEN_control() -> None:
    """A quarantined or root-level module importing another old-prefix shim
    is not what this bound measures -- only a LANDED source counts."""
    edges = [("entropic_substrate.cli", "entropic_substrate.logging")]
    assert count_landed_to_old_prefix_edges(edges) == 0
