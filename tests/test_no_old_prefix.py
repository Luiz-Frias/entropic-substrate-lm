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

pytestmark = pytest.mark.unit

LANDED_LAYER_PREFIXES = (
    "entropic_substrate.kernel",
    "entropic_substrate.core",
    "entropic_substrate.ladder",
    "entropic_substrate.historical",
    "entropic_substrate.composition",
)


def _is_landed(mod: str) -> bool:
    """A module is LANDED iff it starts with one of the five layer prefixes."""
    return any(mod.startswith(prefix) for prefix in LANDED_LAYER_PREFIXES)


def _is_legacy(mod: str) -> bool:
    """A module is in __legacy iff it is under entropic_substrate.__legacy."""
    return mod.startswith("entropic_substrate.__legacy")


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
