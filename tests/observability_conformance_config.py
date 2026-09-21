"""Named constants for :mod:`tests.test_observability_conformance`'s three arms.

Lives BESIDE the frozen test module (``frozen_after: p13``), not inside it
-- ENTRY-POINT ARM RULING (conductor 2026-09-21, resolving wf_3b1acf4d-36f):
the entry-point partition must be editable by each phase that moves one of
the seven ``entrypoint: true`` rows, without touching the frozen contract
test itself. This module is p3b's mechanical companion to that ruling.

RETRACTION (house rule 4, conductor 2026-09-21): an earlier revision of
this docstring claimed to be "the ONE file added beyond the brief's
Allowed-files list". That was wrong on its face -- the brief's Allowed
files section (``.claude/workflows/briefs/p3b-observability-rail.md``)
lists ``tests/observability_conformance_config.py`` explicitly. Zero files
were added beyond the brief's own envelope; this module is declared IN it,
not beside it.

CONFORMANCE-PROMOTION RULING (conductor 2026-09-21, resolving
wf_8a5c8008-50e review:fixer): This module DERIVES its three per-phase sets
(``CONFORMING_ENTRYPOINTS``, ``PRINT_ALLOWLIST``, ``ALLOWED_SINK_CONSTRUCTION_MODULES``)
by merging the optional fragment keys ``conforming_entrypoints``, ``print_allowlist``
and ``sink_construction_modules`` of every ``.claude/workflows/landed-set/*.json``
present. A phase promotes an entry point, re-keys an allowlist entry, or adds a
construction site by editing its OWN fragment in the same commit that moves the
module. The fragment is already in every phase's touch set and fragments are
disjoint by construction, so later phases never share a file with p3b.
"""

from __future__ import annotations

import json
from pathlib import Path

LANDED_LAYER_PREFIXES = (
    "entropic_substrate.kernel",
    "entropic_substrate.core",
    "entropic_substrate.ladder",
    "entropic_substrate.historical",
    "entropic_substrate.composition",
)


def _load_merged_fragment_sets(
    fragments_dir: Path,
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Load and merge conforming_entrypoints, print_allowlist, and
    sink_construction_modules from every .claude/workflows/landed-set/*.json
    fragment present. Returns (conforming, allowlist, sinks) as frozensets."""
    conforming = set()
    allowlist = set()
    sinks = set()

    if fragments_dir.exists():
        for fragment_file in sorted(fragments_dir.glob("*.json")):
            try:
                data = json.loads(fragment_file.read_text())
                if "conforming_entrypoints" in data:
                    conforming.update(data["conforming_entrypoints"])
                if "print_allowlist" in data:
                    allowlist.update(data["print_allowlist"])
                if "sink_construction_modules" in data:
                    sinks.update(data["sink_construction_modules"])
            except (json.JSONDecodeError, OSError):
                pass

    return frozenset(conforming), frozenset(allowlist), frozenset(sinks)


def _load_all_entrypoints(mapping_file: Path) -> frozenset[str]:
    """Load all entrypoint: true rows from tools/mapping.json."""
    if not mapping_file.exists():
        return frozenset()

    try:
        data = json.loads(mapping_file.read_text())
        entrypoints = {
            name for name, entry in data.get("modules", {}).items() if entry.get("entrypoint")
        }
        return frozenset(entrypoints)
    except (json.JSONDecodeError, OSError):
        return frozenset()


# Derive all three constants from fragment merging and tools/mapping.json
_repo_root = Path(__file__).resolve().parents[1]
_fragments_dir = _repo_root / ".claude" / "workflows" / "landed-set"
_mapping_file = _repo_root / "tools" / "mapping.json"

_conforming_from_fragments, _allowlist_from_fragments, _sinks_from_fragments = (
    _load_merged_fragment_sets(_fragments_dir)
)
_all_entrypoints = _load_all_entrypoints(_mapping_file)

# ---------------------------------------------------------------------------
# Print-ban arm: the two machine-consumed stdout sites pinned by p0's CLI
# snapshot (entropic_substrate.telemetry's `final-eval` subcommand; cli's
# `eval --json`) stay raw `print()` forever -- a loguru prefix on either is
# a hard break, not a formatting nit (see
# entropic_substrate.kernel.runtime.logging's module docstring). Entries
# here are keyed on the LANDED dotted name the print-ban arm derives from a
# file's path under a landed layer root (test_observability_conformance.py
# scans only `_landed_py_files()`) -- the same dotted-name space the
# entry-point and sink-construction arms use. The phase that lands each
# module must re-key its entry to the new landed dotted name in the SAME
# commit that moves it by editing its own fragment.
# ---------------------------------------------------------------------------
PRINT_ALLOWLIST: frozenset[str] = _allowlist_from_fragments

# ---------------------------------------------------------------------------
# Entry-point arm (RULING 2026-09-21, wf_3b1acf4d-36f): the SEVEN
# `entrypoint: true` rows in tools/mapping.json, partitioned into CONFORMING
# (calls configure_logging() at its entry function -- MEASURED 2026-09-21:
# exactly entropic_substrate.cli:1057 and entropic_substrate.telemetry:494,
# both the first statement of `main()`) and NOT-YET-CONFORMED (the other
# five, zero configure_logging calls today). len(CONFORMING) +
# len(NOT_YET_CONFORMED) must equal 7 and len(CONFORMING) must be >= 2 --
# asserted by the test, not here, so a future edit to this file that drops
# an entrypoint by accident is still caught.
# ---------------------------------------------------------------------------
CONFORMING_ENTRYPOINTS: frozenset[str] = _conforming_from_fragments
NOT_YET_CONFORMED_ENTRYPOINTS: frozenset[str] = _all_entrypoints - _conforming_from_fragments

# ---------------------------------------------------------------------------
# Sink-construction arm: the run-descriptor door's single construction seam
# (brief, Contract surface owned: "TrackingRunDescriptor ... + build_sink_for_run
# ... as the single construction seam"). Every MetricSink-implementing class
# construction site inside a LANDED package must resolve to a module in this
# set. Container-typed and membership-tested (architect constraint
# 2026-09-21), not a scalar compared with `!=`: every OTHER per-phase-
# editable constant in this file is a frozenset, and when the real
# composition root acquires a second sanctioned construction site the
# promoting phase adds a member here -- it never has to change a scalar's
# TYPE inside a contract test frozen_after p13 to do it.
# ---------------------------------------------------------------------------
ALLOWED_SINK_CONSTRUCTION_MODULES: frozenset[str] = _sinks_from_fragments
SINK_IMPLEMENTATION_CLASS_NAMES: frozenset[str] = frozenset(
    {"JsonlSink", "MlflowSink", "CompositeSink"}
)
