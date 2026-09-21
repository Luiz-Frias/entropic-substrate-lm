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

Every later phase that lands one of ``NOT_YET_CONFORMED_ENTRYPOINTS`` moves
its dotted name into ``CONFORMING_ENTRYPOINTS`` in the SAME commit that
lands it and adds a ``configure_logging()`` call at its entry function --
this file is where that promotion happens, never inside the test.
"""

from __future__ import annotations

LANDED_LAYER_PREFIXES = (
    "entropic_substrate.kernel",
    "entropic_substrate.core",
    "entropic_substrate.ladder",
    "entropic_substrate.historical",
    "entropic_substrate.composition",
)

# ---------------------------------------------------------------------------
# Print-ban arm: the two machine-consumed stdout sites pinned by p0's CLI
# snapshot (entropic_substrate.telemetry's `final-eval` subcommand; cli's
# `eval --json`) stay raw `print()` forever -- a loguru prefix on either is
# a hard break, not a formatting nit (see
# entropic_substrate.kernel.runtime.logging's module docstring). Entries
# here are keyed on the LANDED dotted name the print-ban arm derives from a
# file's path under a landed layer root (test_observability_conformance.py
# scans only `_landed_py_files()`) -- the same dotted-name space the
# entry-point and sink-construction arms use. Neither `telemetry` nor `cli`
# moves this phase (p9/p10 own them), so the two entries below are PRE-MOVE
# PLACEHOLDERS at their pre-refactor dotted names: they match nothing today
# (the allowlist is currently never consulted against a landed file) and
# the phase that lands each module must re-key its entry to the new landed
# dotted name in the SAME commit that moves it, never edit the frozen
# contract test to do so.
# ---------------------------------------------------------------------------
PRINT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "entropic_substrate.telemetry",
        "entropic_substrate.cli",
    }
)

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
CONFORMING_ENTRYPOINTS: frozenset[str] = frozenset(
    {
        "entropic_substrate.cli",
        "entropic_substrate.telemetry",
    }
)
NOT_YET_CONFORMED_ENTRYPOINTS: frozenset[str] = frozenset(
    {
        "entropic_substrate.__main__",
        "entropic_substrate.analysis.d1_external_worker",
        "entropic_substrate.d1_trainer_mailbox_command",
        "entropic_substrate.d_ladder_worker_command",
        "entropic_substrate.sentence_prediction.worker_command",
    }
)

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
ALLOWED_SINK_CONSTRUCTION_MODULES: frozenset[str] = frozenset(
    {"entropic_substrate.ladder.capture.adapters.composite"}
)
SINK_IMPLEMENTATION_CLASS_NAMES: frozenset[str] = frozenset(
    {"JsonlSink", "MlflowSink", "CompositeSink"}
)
