"""The closed vocabulary of the H worker's log events (FOR-1637).

A LEAF module: standard library only. It holds names and one rendering
helper; every emission is a direct ``logger.bind(event=..., **fields)`` call at
the milestone it narrates (``kernel.runtime.logging.logger``), so the human
stderr line and the structured ``worker-log.jsonl`` record come from the one
call. It is deliberately NOT a member of ``checkpoint._READER_MODULES``: later
wording edits here must not re-pin the reader identity.

The human line is ``<event> <field>=<value> ...`` (the substrate prepends its
own ``<ts> | LEVEL |``); the structured record carries the same fields as flat
JSON keys, with one exception: ``evaluation.end`` binds the projected ``eval/*``
instrument floats (FOR-1704) onto the structured record only, while its human
line still renders ``global_step`` and ``seconds`` alone. Field values are limited
to keys, sizes, digests, counts, durations, run ids, names, boolean flags such as
``sink.finish.end``'s ``settled``, exception TYPE names and, for ``evaluation.end``
only, those finite metric floats, never
``str(exception)`` of a provider error.

Three vocabulary decisions are recorded here because the set is a frozen door:

* ``sink.finish.end`` carries ``settled`` (did the metric drain finish inside its
  budget) and ``stranded``, NOT the ``drained`` the brief first named. The
  unmodified ``tests/test_mlflow_async_sink.py`` counts stderr lines containing
  ``drain``, and under the pre-existing conftest sink defect those counts are
  taken over loguru's error dump, so a ``drained=`` field would shift them.
  ``settled`` is the ratified name; renaming it back is a vocabulary amendment
  to be made only after the conftest fix lands beneath this change.
* The missing-declared-log WARNING in ``MlflowSink.finish()`` binds no event. An
  absent ``worker-log.jsonl`` is already visible as the absence of the artifact,
  so the closed set is not widened for it;
  ``test_a_declared_log_that_is_missing_warns_without_counting_or_raising`` pins
  the warning text and that it carries no ``event`` key.
* The multipart-abort-failed WARNING in ``Boto3D1TrainerMailboxObjectAdapter`` (FOR-1694)
  binds no event either. A failed ``abort_multipart_upload`` is an exceptional-path cleanup of
  a transport adapter, logged by exception TYPE only, never a key, upload id or message; it
  narrates no milestone of a run, so the closed set is not widened for it.
  ``test_a_failing_abort_is_logged_by_type_and_never_masks_the_cause`` pins the text.

Adapters under ``ladder.capture`` cannot import this module (layer edge
``capture -> run`` is not admitted), so they emit the literal names below;
``tests/test_h_worker_log_lines.py`` pins that the constant and the observed
set of emitted ``event`` names are equal in both directions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

__all__ = [
    "ADMISSION_CALLERS",
    "ADMISSION_END",
    "ADMISSION_MEMOS",
    "ADMISSION_ROOTS",
    "ADMISSION_START",
    "CHECKPOINT_SEAL",
    "COMPLETION_WRITTEN",
    "EVALUATION_END",
    "EVALUATION_START",
    "H_WORKER_LOG_EVENTS",
    "MLFLOW_RUN",
    "PHASE_END",
    "PHASE_START",
    "PUBLISH_END",
    "PUBLISH_RETRY",
    "PUBLISH_START",
    "PUBLISH_VERIFY_READ",
    "SINK_DRAIN_EXCEEDED",
    "SINK_FINISH_END",
    "SINK_FINISH_START",
    "SINK_WRITE_FAILED",
    "SNAPSHOT_END",
    "SNAPSHOT_START",
    "STAGE_END",
    "STAGE_START",
    "STOP_MARKER_OBSERVED",
    "STOP_MARKER_RETRY",
    "WORKER_END",
    "WORKER_STAGES",
    "WORKER_STAGE_SCHEMA",
    "WORKER_START",
    "project_worker_stage_record",
    "render",
]

WORKER_START = "worker.start"
WORKER_END = "worker.end"
PHASE_START = "phase.start"
PHASE_END = "phase.end"
STAGE_START = "stage.start"
STAGE_END = "stage.end"
CHECKPOINT_SEAL = "checkpoint.seal"
EVALUATION_START = "evaluation.start"
EVALUATION_END = "evaluation.end"
SNAPSHOT_START = "snapshot.start"
SNAPSHOT_END = "snapshot.end"
PUBLISH_START = "publish.start"
PUBLISH_END = "publish.end"
PUBLISH_VERIFY_READ = "publish.verify_read"
PUBLISH_RETRY = "publish.retry"
STOP_MARKER_OBSERVED = "stop_marker.observed"
STOP_MARKER_RETRY = "stop_marker.retry"
COMPLETION_WRITTEN = "completion.written"
# Emitted by the capture adapters under their literal names (layer edge).
SINK_FINISH_START = "sink.finish.start"
SINK_FINISH_END = "sink.finish.end"
SINK_DRAIN_EXCEEDED = "sink.drain_exceeded"
SINK_WRITE_FAILED = "sink.write_failed"
MLFLOW_RUN = "mlflow.run"
# FOR-1662: one pair per top-level admission site of the final leg, never per
# node, per call or per step. ``caller`` and ``root`` are CLOSED sets (never a
# filesystem path, so no operator-home string reaches a published log).
ADMISSION_START = "admission.start"
ADMISSION_END = "admission.end"
ADMISSION_CALLERS: frozenset[str] = frozenset(
    {
        "completion",
        "training_readback",
        "generation",
        "span_diagnostics",
        "trace_capture",
        "native_verification",
        "result_admission",
    }
)
ADMISSION_ROOTS: frozenset[str] = frozenset({"run", "capture"})
ADMISSION_MEMOS: frozenset[str] = frozenset({"fresh", "shared"})
# FOR-1669: every stage ``narrated_stage`` brackets (``worker_execution`` and
# ``object_worker`` call sites, grep-measured 2026-10-19). Closed so that a stage name can
# never mint an unbounded family of MLflow metric keys.
WORKER_STAGES: frozenset[str] = frozenset(
    {
        "materialization",
        "runtime_admission",
        "training_readback",
        "generation",
        "span_diagnostics",
        "trace_capture",
        "native_verification",
        "result_admission",
    }
)
# The record schema the capture layer's log bridge hands a sink for a ``stage.end`` /
# ``admission.end`` line (FOR-1669). It is a SINK record, not a log event: the closed log
# vocabulary above is untouched and ``H_WORKER_LOG_EVENTS`` does not list it.
WORKER_STAGE_SCHEMA = "h-worker-stage-v1"

H_WORKER_LOG_EVENTS: frozenset[str] = frozenset(
    {
        WORKER_START,
        WORKER_END,
        PHASE_START,
        PHASE_END,
        STAGE_START,
        STAGE_END,
        CHECKPOINT_SEAL,
        EVALUATION_START,
        EVALUATION_END,
        SNAPSHOT_START,
        SNAPSHOT_END,
        PUBLISH_START,
        PUBLISH_END,
        PUBLISH_VERIFY_READ,
        PUBLISH_RETRY,
        STOP_MARKER_OBSERVED,
        STOP_MARKER_RETRY,
        COMPLETION_WRITTEN,
        SINK_FINISH_START,
        SINK_FINISH_END,
        SINK_DRAIN_EXCEEDED,
        SINK_WRITE_FAILED,
        MLFLOW_RUN,
        ADMISSION_START,
        ADMISSION_END,
    }
)


def render(event: str, **fields: object) -> str:
    """One human sentence: the event name, then ``name=value`` per field.

    Floats render with millisecond precision; everything else with ``str()``.
    The result is a plain message (never a format string): callers pass the
    fields to ``logger.bind`` and this string to ``.info``/``.warning`` with no
    format arguments, so a brace in a key can never reach a formatter.
    """
    parts = [event]
    for name, value in fields.items():
        parts.append(f"{name}={value:.3f}" if isinstance(value, float) else f"{name}={value}")
    return " ".join(parts)


def _seconds(value: object) -> float | None:
    # Exact-type test: ``bool`` is an ``int`` subclass and must not pass as a duration.
    if type(value) is int or type(value) is float:
        number = float(value)
        if math.isfinite(number) and number >= 0.0:
            return number
    return None


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def project_worker_stage_record(record: Mapping[str, object]) -> dict[str, object] | None:
    """Normalise a bound ``stage.end`` / ``admission.end`` record into one sink record.

    ``record`` is the loguru ``extra`` of the bound call (or the same mapping carrying the
    ``schema`` key the log bridge adds). The result carries ONLY the closed-vocabulary
    fields, under :data:`WORKER_STAGE_SCHEMA`; anything else returns ``None``: any other
    event, a stage, caller, root or memo outside its closed set, a missing, non-numeric,
    negative or non-finite ``seconds`` or count. Nothing is guessed or defaulted, and the
    function is pure (no logging, no clock), so the extractor can call it on every record.
    """
    event = record.get("event")
    seconds = _seconds(record.get("seconds"))
    if seconds is None:
        return None
    if event == STAGE_END:
        stage = record.get("stage")
        if stage not in WORKER_STAGES:
            return None
        return {
            "schema": WORKER_STAGE_SCHEMA,
            "event": STAGE_END,
            "stage": stage,
            "seconds": seconds,
        }
    if event == ADMISSION_END:
        caller, root, memo = record.get("caller"), record.get("root"), record.get("memo")
        counts = {
            name: _count(record.get(name))
            for name in ("seals_walked", "evaluations", "journals_read")
        }
        if (
            caller not in ADMISSION_CALLERS
            or root not in ADMISSION_ROOTS
            or memo not in ADMISSION_MEMOS
            or any(value is None for value in counts.values())
        ):
            return None
        return {
            "schema": WORKER_STAGE_SCHEMA,
            "event": ADMISSION_END,
            "caller": caller,
            "root": root,
            "memo": memo,
            **counts,
            "seconds": seconds,
        }
    return None
