"""The ``MetricSink`` port -- where ``Trainer.log()``/``log_artifact()`` records go.

Landed at p3b (docs/plans/src-hexagonal-refactor.plan.md #p3b) from
``entropic_substrate.sinks:217`` (``entropic_substrate.__legacy.sinks`` under
quarantine). The ONLY metric port in the tree, ``kind: door``, ``frozen_after:
p3b`` -- D1's second port does not survive (RULED 2026-09-17, reaffirmed R1
2026-09-21, wf_12821961-8cd: "one port" means one METRIC port; p8 re-bases
``D1MlflowObserver`` as a strict adapter of this Protocol).

Adapters behind this port: :class:`entropic_substrate.ladder.capture.adapters.jsonl.JsonlSink`
(authoritative), :class:`entropic_substrate.ladder.capture.adapters.mlflow.MlflowSink`
(best-effort), fanned out by
:class:`entropic_substrate.ladder.capture.adapters.composite.CompositeSink`.
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

    def finish(self) -> None:
        """Close out the run. Idempotent — safe to call more than once."""
        ...
