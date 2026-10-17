"""The ``MetricSink`` port -- where ``Trainer.log()``/``log_artifact()`` records go.

Landed at p3b (docs/plans/src-hexagonal-refactor.plan.md #p3b) from
``entropic_substrate.sinks:217`` (``entropic_substrate.__legacy.sinks`` under
quarantine). The ONLY metric port in the tree, ``kind: door``, ``frozen_after:
p3b`` -- D1's second port does not survive (RULED 2026-09-17, reaffirmed R1
2026-09-21, wf_12821961-8cd: "one port" means one METRIC port). p8 moved the D1
observers RENAME-ONLY (ruling P8-OBSERVERS-RENAME-ONLY, FOR-1552); re-basing
``D1MlflowObserver`` onto this Protocol is deferred past p13.

Adapters behind this port: :class:`entropic_substrate.ladder.capture.adapters.jsonl.JsonlSink`
(authoritative), :class:`entropic_substrate.ladder.capture.adapters.mlflow.MlflowSink`
(best-effort), fanned out by
:class:`entropic_substrate.ladder.capture.adapters.composite.CompositeSink`.

**Amendment 2026-10-15 (FOR-1642/1643/1644, DN-1643-PORT, operator ruling 2026-10-14,
option A; same shape as DN-1637-DOOR):** the door gains two ADDITIVE members and
nothing else. ``annotate(tags)`` attaches ``str -> str`` run tags (the H worker's
per-checkpoint evidence): :class:`~entropic_substrate.ladder.capture.adapters.jsonl.JsonlSink`
appends ``{"event": "annotate", **tags}`` (no ``schema`` key, so no schema-keyed reader
counts it), :class:`~entropic_substrate.ladder.capture.adapters.mlflow.MlflowSink` sets one
run tag per key, best-effort and counted like every other write after startup.
``finish(*, status="FINISHED")`` takes the closed vocabulary ``{"FINISHED", "FAILED"}``:
:class:`~entropic_substrate.ladder.capture.adapters.mlflow.MlflowSink` terminates the run
with it, :class:`~entropic_substrate.ladder.capture.adapters.jsonl.JsonlSink` ignores it,
and :class:`~entropic_substrate.ladder.capture.adapters.composite.CompositeSink` fans both
out. A no-argument ``finish()`` is byte-identical to the pre-amendment one on every adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = ["MetricSink"]


@runtime_checkable
class MetricSink(Protocol):
    """Port for one run's metric stream. See module docstring for the contract."""

    def log(self, record: Mapping[str, Any]) -> None:
        """Record one JSONL-shaped event (``run_start`` / ``train`` / ``eval`` / ``run_end``)."""
        ...

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        """Attach a file (a checkpoint) to this run."""
        ...

    def annotate(self, tags: Mapping[str, str]) -> None:
        """Attach ``str -> str`` run tags (DN-1643-PORT). The keys ``event`` and ``schema``
        are reserved for the record stream and refused."""
        ...

    def finish(self, *, status: str = "FINISHED") -> None:
        """Close out the run. Idempotent — safe to call more than once."""
        # DN-1643-PORT: ``status`` is "FINISHED" (default) or "FAILED"; any other value is
        # refused with a ValueError before any side effect.
        ...
