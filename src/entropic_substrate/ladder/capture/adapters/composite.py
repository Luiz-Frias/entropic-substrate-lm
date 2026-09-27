"""``CompositeSink`` -- fans one sink call out to N adapters, plus the
run-descriptor door's single construction seam, :func:`build_sink_for_run`.

Landed at p3b from ``entropic_substrate.sinks`` (quarantined
``entropic_substrate.ladder.capture.reader.sinks``): ``CompositeSink`` was ``sinks.py:640``,
``build_sink_for_run`` was ``sinks.py:737``.

``build_sink_for_run`` physically lives HERE, not in
``entropic_substrate.ladder.capture.ports.run_descriptor`` -- the Allowed-files
door forbids ``ports/`` importing ``adapters/``, and this function directly
constructs :class:`~entropic_substrate.ladder.capture.adapters.jsonl.JsonlSink`
and :class:`~entropic_substrate.ladder.capture.adapters.mlflow.MlflowSink`.
Together with :class:`~entropic_substrate.ladder.capture.ports.run_descriptor.TrackingRunDescriptor`
it is the run-descriptor door (``kind: door``, ``frozen_after: p3b``) -- the
single construction seam every later phase (p8's ``d1_tracking_descriptor``
included) consumes.

**Cycle note:** ``build_sink_for_run`` reads ``mlflow_enabled``,
``MLFLOW_TRACKING_ENV``, ``MLFLOW_EXPERIMENT_ENV`` and the default experiment
name from ``entropic_substrate.ladder.capture.reader.sinks`` (residue, p8) -- imported
inside the function body, never at module scope, for the same reason
``adapters/mlflow.py`` does: ``ladder.capture.reader.sinks`` imports this
module at module level to re-export both names (the ``__legacy/sinks.py``
facade that once did so was deleted in 0c94e25), so a module-level import
back would be a real cycle.

**Frozen door vs. body residue (design-gate risk, architect constraint
2026-09-21):** the run-descriptor door's freeze covers ``TrackingRunDescriptor``'s
and ``build_sink_for_run``'s SIGNATURE and documented raise/abort semantics --
never the residue names their bodies still reach into
``entropic_substrate.ladder.capture.reader.sinks`` for. The full p8-owned relocation set,
across both files this door's construction path touches:

* HERE (``build_sink_for_run``, function-scope import above): ``_DEFAULT_EXPERIMENT``,
  ``MLFLOW_TRACKING_ENV``, ``MLFLOW_EXPERIMENT_ENV``, ``mlflow_enabled``.
* ``adapters/mlflow.py`` (function-scope imports, that module's own body):
  ``MLFLOW_SYSTEM_METRICS_ENV`` (``mlflow.py:144``) and ``MlflowStartupError``
  (``mlflow.py:262``, raised at ``mlflow.py:292`` and ``mlflow.py:328``).

p8 re-pointing any of these six names to their eventual landed home is a BODY
edit to this function and to ``adapters/mlflow.py``, not a door violation --
the door's signature and abort contract are unchanged by where the residue
lives. The one exception is ``MlflowStartupError``'s class IDENTITY: it is
part of ``build_sink_for_run``'s documented startup-abort contract (see that
function's docstring), and out-of-envelope callers catch it by the old
prefix (``tests/test_caddy_auth.py``, ``tests/test_sinks.py``,
``__legacy/cli.py``) -- p8 MUST re-export the SAME class object under its new
home, never redefine it, or ``except MlflowStartupError`` silently stops
catching and a bad ``MLFLOW_TRACKING_URI`` burns GPU hours instead of
aborting before ``Trainer`` construction.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from entropic_substrate.ladder.capture.adapters.jsonl import JsonlSink
from entropic_substrate.ladder.capture.adapters.mlflow import MlflowSink
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.capture.ports.run_descriptor import (
    MetricExtractor,
    TrackingRunDescriptor,
)

__all__ = ["CompositeSink", "build_sink_for_run"]


class CompositeSink:
    """Fans one ``Trainer.log()``/``log_artifact()`` call out to N sinks.

    ``authoritative`` is never guarded here — a genuine defect in it must
    propagate. Every sink in ``best_effort`` is expected to already
    self-guard per :class:`MetricSink`'s contract (:class:`MlflowSink` does);
    this class does not add a second guard on top, so a sink that is NOT
    actually best-effort still surfaces its own failures honestly.
    """

    def __init__(self, authoritative: MetricSink, *best_effort: MetricSink) -> None:
        self._authoritative = authoritative
        self._best_effort = best_effort

    def log(self, record: Mapping[str, Any]) -> None:
        self._authoritative.log(record)
        for sink in self._best_effort:
            sink.log(record)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        self._authoritative.log_artifact(path, artifact_path=artifact_path)
        for sink in self._best_effort:
            sink.log_artifact(path, artifact_path=artifact_path)

    def finish(self) -> None:
        self._authoritative.finish()
        for sink in self._best_effort:
            sink.finish()

    @property
    def best_effort(self) -> tuple[MetricSink, ...]:
        """The non-authoritative sinks — for tests/diagnostics."""
        return self._best_effort


def build_sink_for_run(
    descriptor: TrackingRunDescriptor,
    *,
    results_path: str | Path,
    client: Any | None = None,
    metric_extractor: MetricExtractor | None = None,
) -> MetricSink:
    """Compose one authoritative JSONL sink plus an optional tracking mirror.

    Raises :class:`entropic_substrate.ladder.capture.reader.sinks.MlflowStartupError` when
    MLflow is enabled but the tracking server is unreachable/misconfigured —
    callers must let this abort BEFORE constructing a ``Trainer`` (see the
    CLI's ``cmd_train``), so a bad ``MLFLOW_TRACKING_URI`` never burns GPU
    hours.
    """
    # entropic_substrate.ladder.capture.reader.sinks residue (MLFLOW_*_ENV, mlflow_enabled,
    # the default experiment name) -- function-scope import, see module
    # docstring's cycle note.
    from entropic_substrate.ladder.capture.reader.sinks import (
        _DEFAULT_EXPERIMENT,  # pyright: ignore[reportPrivateUsage]
        MLFLOW_EXPERIMENT_ENV,
        MLFLOW_TRACKING_ENV,
        mlflow_enabled,
    )

    if type(descriptor) is not TrackingRunDescriptor:
        raise TypeError("descriptor must be a TrackingRunDescriptor")
    jsonl = JsonlSink(results_path)
    if not mlflow_enabled(tiny=descriptor.tiny) and client is None:
        return jsonl

    tracking_uri = os.environ.get(MLFLOW_TRACKING_ENV, "")
    experiment_name = os.environ.get(MLFLOW_EXPERIMENT_ENV, _DEFAULT_EXPERIMENT)
    mlflow_sink = MlflowSink(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=descriptor.run_name,
        params=descriptor.params,
        tags=descriptor.tags,
        results_path=Path(results_path),
        config_hash=descriptor.config_hash,
        dataset_pointer_path=descriptor.dataset_pointer_path,
        dataset_split_spec=descriptor.dataset_split_spec,
        client=client,
        metric_extractor=metric_extractor,
    )
    return CompositeSink(jsonl, mlflow_sink)
