"""Named constants for :mod:`tests.test_observability_conformance`'s three arms.

Lives BESIDE the frozen test module (``frozen_after: p13``), not inside it --
ENTRY-POINT ARM RULING (conductor 2026-09-21, resolving wf_3b1acf4d-36f): the
entry-point partition must be editable by each phase that moves one of the
seven ``entrypoint: true`` rows, without touching the frozen contract test
itself. This module is p3b's mechanical companion to that ruling.

INLINED 2026-09-26 (p13-sweep, B2 row 10, P13-OBS as amended): all four sets
are now literal constants. The fragment-merging convention
(``.claude/workflows/landed-set/*.json``) and the ``tools/mapping.json``
entry-point reader retire with the generator in this same commit -- both
``import gen_shims`` and the JSON readers are gone, so a future phase that
wants to promote an entry point, re-key an allowlist entry, or add a
construction site edits this module's literals directly. This module still
carries no fragment reader, per B2 row 10 and the ACCEPTANCE bullet: "keeps
no fragment reader and no gen_shims import".

The three sets below were measured against the last fragment-merged state at
the B1 tip before this inlining (12 merged print-allowlist entries minus the
2 orphaned old-name duplicates `entropic_substrate.telemetry` and
`entropic_substrate.cli`, canonical count 10; 7 `entrypoint: true` rows from
`tools/mapping.json` partitioned 3 CONFORMING / 4 NOT_YET_CONFORMED; 3 sink
construction modules), each old name mapped to its landed dotted address
one-for-one via the shim each old name resolved through at that tip.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
# ONE source of the five landed prefixes (P13-NAMESPACE-RULE, B2 row 10): this
# module's own second copy is retired, importing test_no_old_prefix's instead.
from test_no_old_prefix import LANDED_LAYER_PREFIXES  # noqa: E402

__all__ = [
    "ALLOWED_SINK_CONSTRUCTION_MODULES",
    "CONFORMING_ENTRYPOINTS",
    "LANDED_LAYER_PREFIXES",
    "NOT_YET_CONFORMED_ENTRYPOINTS",
    "NOT_YET_CONFORMED_REASONS",
    "PRINT_ALLOWLIST",
    "SINK_IMPLEMENTATION_CLASS_NAMES",
]

# ---------------------------------------------------------------------------
# Print-ban arm: the two machine-consumed stdout sites pinned by p0's CLI
# snapshot (entropic_substrate.telemetry's `final-eval` subcommand; cli's
# `eval --json`) stay raw `print()` forever -- a loguru prefix on either is
# a hard break, not a formatting nit (see
# entropic_substrate.kernel.runtime.logging's module docstring). Entries here
# are keyed on the LANDED dotted name the print-ban arm derives from a file's
# path under a landed layer root (test_observability_conformance.py scans
# only `_landed_py_files()`) -- the same dotted-name space the entry-point and
# sink-construction arms use. EXACT set with orphan detection (ACCEPTANCE):
# every allowlisted module must contain a `print(` site.
# ---------------------------------------------------------------------------
PRINT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "entropic_substrate.composition.capture_cli",
        "entropic_substrate.composition.cli",
        "entropic_substrate.composition.entrypoints.cli",
        "entropic_substrate.composition.source_cli",
        "entropic_substrate.composition.span_cli",
        "entropic_substrate.composition.wiring.d1_probe_cli",
        "entropic_substrate.composition.wiring.d_ladder_cli",
        "entropic_substrate.composition.wiring.d_ladder_probe_cli",
        "entropic_substrate.ladder.capture.reader.telemetry",
        "entropic_substrate.ladder.review.d1_external_worker",
    }
)

# ---------------------------------------------------------------------------
# Entry-point arm (RULING 2026-09-21, wf_3b1acf4d-36f; AMENDED 2026-09-26,
# P13-OBS addendum 2): the SEVEN `entrypoint: true` rows of the retired
# `tools/mapping.json`, at their LANDED names, partitioned into CONFORMING
# (calls `configure_logging()` first in `main()`) and NOT_YET_CONFORMED (the
# other four, pinned by name with the reason beside each). len(CONFORMING) +
# len(NOT_YET_CONFORMED) must equal 11 and len(CONFORMING) must be >= 2 --
# asserted by the test, not here, so a future edit to this file that drops an
# entrypoint by accident is still caught.
#
# AMENDED 2026-10-16 (FOR-1645, DN-OBS-CONF): the universe is ELEVEN. Four H host
# entry points join as CONFORMING, so len(CONFORMING) is 7 and len(NOT_YET) stays 4.
# ---------------------------------------------------------------------------
CONFORMING_ENTRYPOINTS: frozenset[str] = frozenset(
    {
        "entropic_substrate.composition.entrypoints.cli",
        "entropic_substrate.ladder.run.worker_command",
        "entropic_substrate.ladder.capture.reader.telemetry",
        # DN-OBS-CONF (operator 2026-10-13; FOR-1645, conductor DN-1645-CONTROLLER-ENTRY = B,
        # 2026-10-16): the four H host ENTRY POINTS join the universe as CONFORMING. The
        # controller (ladder.run.worker.h_probe_controller) is a library the command process
        # runs; it defines no main(), so it is deliberately absent here and has no mapping row.
        "entropic_substrate.ladder.run.worker.h_probe_command",
        "entropic_substrate.ladder.run.worker.h_probe_stop",
        "entropic_substrate.composition.wiring.h_tiny_tracer",
        "entropic_substrate.composition.entrypoints.h_wave_monitor",
    }
)

# Reasons, one per NOT_YET_CONFORMED member (ACCEPTANCE: "each pinned with
# its reason"). FOR-1563 files the conformance DEBT for the two Bug
# Squashing rows in Backlog (P13-OBS, P13-DOCS-ISSUE); the other two have
# their own structural reasons.
NOT_YET_CONFORMED_REASONS: dict[str, str] = {
    "entropic_substrate.composition.entrypoints.__main__": (
        "defines no main() of its own; delegates to composition.entrypoints.cli, "
        "which is already CONFORMING"
    ),
    "entropic_substrate.ladder.review.d1_external_worker": "P11-WORKER-PINNED",
    "entropic_substrate.composition.entrypoints.d1_trainer_mailbox_command": (
        "conformance DEBT, FOR-1563 (Bug Squashing, Backlog)"
    ),
    "entropic_substrate.historical.d_ladder.d_ladder_worker_command": (
        "conformance DEBT, FOR-1563 (Bug Squashing, Backlog)"
    ),
}
NOT_YET_CONFORMED_ENTRYPOINTS: frozenset[str] = frozenset(NOT_YET_CONFORMED_REASONS)

# ---------------------------------------------------------------------------
# Sink-construction arm: the run-descriptor door's single construction seam
# (brief, Contract surface owned: "TrackingRunDescriptor ... + build_sink_for_run
# ... as the single construction seam"). Every MetricSink-implementing class
# construction site inside a LANDED package must resolve to a module in this
# set. Container-typed and membership-tested (architect constraint 2026-09-21),
# not a scalar compared with `!=`: every OTHER per-phase-editable constant in
# this file is a frozenset, and when the real composition root acquires a
# second sanctioned construction site the promoting phase adds a member here
# -- it never has to change a scalar's TYPE inside a contract test frozen_after
# p13 to do it.
# ---------------------------------------------------------------------------
ALLOWED_SINK_CONSTRUCTION_MODULES: frozenset[str] = frozenset(
    {
        "entropic_substrate.historical.e_series.field_pretrain",
        "entropic_substrate.historical.e_series.train",
        "entropic_substrate.ladder.capture.adapters.composite",
    }
)
SINK_IMPLEMENTATION_CLASS_NAMES: frozenset[str] = frozenset(
    {"JsonlSink", "MlflowSink", "CompositeSink"}
)
