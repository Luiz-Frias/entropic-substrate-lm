"""The closed vocabulary of the H worker's log events (FOR-1637).

A LEAF module: standard library only. It holds names and one rendering
helper; every emission is a direct ``logger.bind(event=..., **fields)`` call at
the milestone it narrates (``kernel.runtime.logging.logger``), so the human
stderr line and the structured ``worker-log.jsonl`` record come from the one
call. It is deliberately NOT a member of ``checkpoint._READER_MODULES``: later
wording edits here must not re-pin the reader identity.

The human line is ``<event> <field>=<value> ...`` (the substrate prepends its
own ``<ts> | LEVEL |``); the structured record carries the same fields as flat
JSON keys. Field values are limited to keys, sizes, digests, counts,
durations, run ids, names and exception TYPE names, never ``str(exception)``
of a provider error.

Adapters under ``ladder.capture`` cannot import this module (layer edge
``capture -> run`` is not admitted), so they emit the literal names below;
``tests/test_h_worker_log_lines.py`` pins that the constant and the observed
set of emitted ``event`` names are equal in both directions.
"""

from __future__ import annotations

__all__ = [
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
    "WORKER_START",
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
