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

**Amendment 2026-10-15 (DN-1643-PORT):** :class:`CompositeSink` gained ``annotate(tags)`` and
``finish(*, status)`` with the :class:`MetricSink` door; ``build_sink_for_run``'s signature is
untouched.

**Amendment 2026-10-12 (FOR-1637, ASSUMED pending operator ratification, DN-1637-DOOR):**
``build_sink_for_run`` gained ONE additive keyword-only parameter,
``log_path: str | Path | None = None``, the worker's structured log file. It is
threaded to :class:`~entropic_substrate.ladder.capture.adapters.mlflow.MlflowSink`,
which uploads it beside ``results_path`` at ``finish()``. ``None`` (the default)
keeps every existing caller byte-identical: no extra client call, no extra
warning. ``TrackingRunDescriptor`` is unchanged (the file is a property of the
process, not of the run descriptor). A ``log_path`` resolving to the same file
as ``results_path`` is refused loudly (two writers in the authoritative
telemetry file). Same shape as the ruled DN-1632-SEAM.

**Amendment 2026-10-19 (FOR-1669):** ``build_sink_for_run`` installs ONE loguru handler beside the
composite when it is given a ``metric_extractor``, mirroring the worker's ``stage.end`` and
``admission.end`` log lines into the sink as ``h-worker-stage-v1`` records (see
:class:`CompositeSink`); ``finish()`` removes it first. Signature and abort semantics are
untouched. This module may not import ``ladder.run`` (the capture layer's edge, pinned by
``tests/test_h_worker_log_lines.py::test_ladder_capture_never_imports_the_run_layer``), so
the event names, the schema and the field list below are literals; the closed vocabulary is
enforced where the record is PROJECTED (``worker_log_events.project_worker_stage_record``,
called by ``HTrainingStepExtractor``), and ``tests/test_h_stage_metrics_bridge.py`` pins the
literals against it.

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
import weakref
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.capture.adapters.jsonl import (
    DEFAULT_FINISH_STATUS,
    JsonlSink,
    require_annotation,
    require_finish_status,
)
from entropic_substrate.ladder.capture.adapters.mlflow import MlflowSink
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.capture.ports.run_descriptor import (
    MetricExtractor,
    TrackingRunDescriptor,
)

__all__ = ["CompositeSink", "build_sink_for_run"]

# FOR-1669: the log-substrate bridge. Literals, never imports from ``ladder.run`` (layer edge).
_BRIDGE_EVENTS = frozenset({"stage.end", "admission.end"})
_BRIDGE_SCHEMA = "h-worker-stage-v1"
_BRIDGE_FIELDS = (
    "event",
    "stage",
    "caller",
    "root",
    "memo",
    "seals_walked",
    "evaluations",
    "journals_read",
    "seconds",
)
_BRIDGE_MEMOS = frozenset({"fresh", "shared"})


def _is_bridge_event(record: Mapping[str, Any]) -> bool:
    return record["extra"].get("event") in _BRIDGE_EVENTS


def _remove_handler(handler_id: int) -> None:
    try:
        logger.remove(handler_id)
    except ValueError:
        # Already gone: ``configure_logging()`` removes every handler with no id.
        pass


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
        self._bridge_off: Callable[[], object] | None = None

    def attach_stage_bridge(self) -> None:
        """Mirror the worker's ``stage.end`` / ``admission.end`` log lines into this sink.

        FOR-1669. MEASURED 2026-10-19 (wave ``h0-plumbing-1k-v3.HEpP7g``): the final-leg
        stages and their admission walks were in neither a metric nor a tag, only in log
        lines, so a RUNNING run with only ``system/*`` metrics could not be told from a
        hang. ONE handler on the process logger turns each such line into a
        ``h-worker-stage-v1`` record and logs it through this sink (the extractor places it
        on the training axis or refuses it); an ``admission.end`` also annotates
        ``admission/<caller>/memo``. The handler holds this sink WEAKLY and is removed by
        :meth:`finish` before any leg finishes, or when the sink is collected unfinished,
        so a sink that nobody finished cannot keep writing to its file for the rest of the
        process. Calling it twice is a no-op.
        """
        if self._bridge_off is not None:
            return
        target = weakref.ref(self)

        def forward(message: Any) -> None:
            sink = target()
            if sink is not None:
                sink._mirror_stage_event(message.record["extra"])  # pyright: ignore[reportPrivateUsage]

        handler_id = logger.add(forward, level="INFO", filter=_is_bridge_event)
        self._bridge_off = weakref.finalize(self, _remove_handler, handler_id)

    def _mirror_stage_event(self, extra: Mapping[str, Any]) -> None:
        # A mirror failure must never reach the worker's logging call. The MLflow leg counts
        # its own failures (``sink.write_failed`` sink=mlflow); the authoritative JSONL leg has
        # no counter, so a failure here is reported under its own sink name.
        try:
            fields = {name: extra[name] for name in _BRIDGE_FIELDS if name in extra}
            self.log({"schema": _BRIDGE_SCHEMA, **fields})
            caller, memo = fields.get("caller"), fields.get("memo")
            if (
                fields.get("event") == "admission.end"
                and isinstance(caller, str)
                and memo in _BRIDGE_MEMOS
            ):
                self.annotate({f"admission/{caller}/memo": memo})
        except Exception as exc:
            logger.bind(event="sink.write_failed", sink="stage_bridge", count=1).warning(
                f"stage bridge: mirroring a {extra.get('event')} line failed: {type(exc).__name__}"
            )

    def log(self, record: Mapping[str, Any]) -> None:
        self._authoritative.log(record)
        for sink in self._best_effort:
            sink.log(record)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        self._authoritative.log_artifact(path, artifact_path=artifact_path)
        for sink in self._best_effort:
            sink.log_artifact(path, artifact_path=artifact_path)

    def annotate(self, tags: Mapping[str, str]) -> None:
        # Validated here too (not only per adapter), so a refused mapping reaches NO leg.
        checked = require_annotation(tags)
        self._authoritative.annotate(checked)
        for sink in self._best_effort:
            sink.annotate(checked)

    def finish(self, *, status: str = DEFAULT_FINISH_STATUS) -> None:
        # DN-1643-PORT: refused BEFORE the authoritative leg finishes, so a bad status
        # cannot leave the run half-terminated. The default call stays byte-identical:
        # it passes nothing, so a sink that predates ``status`` is still finishable.
        require_finish_status(status)
        # FOR-1669: the bridge goes first, so a stage line logged after this point lands
        # nothing and is not counted as a failed write on a leg that is already finished.
        if self._bridge_off is not None:
            self._bridge_off()
            self._bridge_off = None
        passed = {} if status == DEFAULT_FINISH_STATUS else {"status": status}
        self._authoritative.finish(**passed)
        for sink in self._best_effort:
            sink.finish(**passed)

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
    log_path: str | Path | None = None,
) -> MetricSink:
    """Compose one authoritative JSONL sink plus an optional tracking mirror.

    Raises :class:`entropic_substrate.ladder.capture.reader.sinks.MlflowStartupError` when
    MLflow is enabled but the tracking server is unreachable/misconfigured —
    callers must let this abort BEFORE constructing a ``Trainer`` (see the
    CLI's ``cmd_train``), so a bad ``MLFLOW_TRACKING_URI`` never burns GPU
    hours.

    ``log_path`` (amendment 2026-10-12, see the module docstring) names the
    worker's structured log file. It is accepted identically in both branches:
    the bare-JSONL branch (MLflow off) ignores it and uploads nothing; the
    MLflow branch uploads it at ``finish()``. It may not alias ``results_path``.
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
    if log_path is not None and Path(log_path).resolve() == Path(results_path).resolve():
        msg = (
            "log_path must not alias results_path: the authoritative telemetry file has one writer"
        )
        raise ValueError(msg)
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
        log_path=None if log_path is None else Path(log_path),
        config_hash=descriptor.config_hash,
        dataset_pointer_path=descriptor.dataset_pointer_path,
        dataset_split_spec=descriptor.dataset_split_spec,
        client=client,
        metric_extractor=metric_extractor,
    )
    composite = CompositeSink(jsonl, mlflow_sink)
    if metric_extractor is not None:
        composite.attach_stage_bridge()
    return composite
