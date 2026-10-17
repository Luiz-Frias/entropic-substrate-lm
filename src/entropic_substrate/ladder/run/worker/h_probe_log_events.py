"""The closed vocabulary of the H host's log events (FOR-1645, FOR-1646).

A LEAF module: standard library only. It holds names, one rendering helper, the
progress-line builder and the checkpoint schedule arithmetic; every emission is
a direct ``logger.bind(event=..., **fields)`` call at the milestone it narrates
(``kernel.runtime.logging.logger``), so the human stderr line and the structured
JSON record come from the one call. It is deliberately NOT a member of
``checkpoint._READER_MODULES``: wording edits here must not re-pin the reader
identity, and it is disjoint from ``worker_log_events.H_WORKER_LOG_EVENTS``
(every name here starts with ``host.``).

The host process is the command process: ``h_probe_command`` runs the
controller, so controller events ride the command's substrate. Events narrate
only what that process observes (a stop handoff it wrote, a watchdog join it
waited on, an emergency cleanup it attempted), never a deletion or termination
it did not perform or observe. The watchdog child does the teardown.

Field values are limited to ids, codes, counts, steps, durations and exception
TYPE names. Never ``str(exception)``, raw provider output, environment values
or credentials. Host events never use ``logger.bind(report=True)`` (the report
sink is stdout, which carries the machine-readable evidence JSON) and never go
through the injected stdout/stderr text streams of the commands.

PROGRESS HAS TWO EVIDENCE SOURCES AND THEY ARE NOT INTERCHANGEABLE. The
controller reads R2 ``checkpoint-NNNNNNNN`` events, which advance only at seal
boundaries; the wave monitor reads the MLflow ``phase_index`` metric, which
advances every optimizer step. Both are on the global optimizer-step axis, so a
bare number would look the same. Every rendered line therefore names its source,
and the operator's hung rule ("zero new steps for 30 minutes") is defined on the
per-step MLflow source only. A run that is not found, a read that failed, a
listing that failed and "nothing yet" are each their own explicit state and are
never rendered as the previous step or as 0.
"""

from __future__ import annotations

__all__ = [
    "HOST_ALLOCATION_END",
    "HOST_ALLOCATION_START",
    "HOST_EMERGENCY_CLEANUP",
    "HOST_MONITOR_END",
    "HOST_MONITOR_TICK",
    "HOST_POD_ATTRIBUTED",
    "HOST_POLL_TICK",
    "HOST_PROGRESS_UNAVAILABLE",
    "HOST_PROVIDER_ABSENT_OBSERVED",
    "HOST_PROVIDER_TRANSIENT",
    "HOST_STOP_HANDOFF",
    "HOST_STOP_MARKER_CONFIRMED",
    "HOST_STOP_MARKER_REQUESTED",
    "HOST_STOP_MARKER_UNCONFIRMED",
    "HOST_TERMINAL_OBSERVED",
    "HOST_TERMINAL_PROBE_TRANSIENT",
    "HOST_WATCHDOG_JOIN_END",
    "HOST_WATCHDOG_JOIN_START",
    "H_HOST_LOG_EVENTS",
    "PROGRESS_SOURCES",
    "PROGRESS_STATES",
    "SOURCE_MLFLOW_METRIC",
    "SOURCE_R2_CHECKPOINT",
    "STATE_LIST_FAILED",
    "STATE_NO_PROGRESS",
    "STATE_READ_FAILED",
    "STATE_RUN_NOT_FOUND",
    "STATE_STEP",
    "checkpoint_steps",
    "progress_fields",
    "render",
]

HOST_ALLOCATION_START = "host.allocation.start"
HOST_ALLOCATION_END = "host.allocation.end"
HOST_POD_ATTRIBUTED = "host.pod.attributed"
HOST_POLL_TICK = "host.poll.tick"
HOST_PROGRESS_UNAVAILABLE = "host.progress.unavailable"
HOST_PROVIDER_TRANSIENT = "host.provider.transient"
HOST_TERMINAL_OBSERVED = "host.terminal.observed"
HOST_TERMINAL_PROBE_TRANSIENT = "host.terminal_probe.transient"
HOST_PROVIDER_ABSENT_OBSERVED = "host.provider_absent.observed"
HOST_STOP_HANDOFF = "host.stop.handoff"
HOST_WATCHDOG_JOIN_START = "host.watchdog_join.start"
HOST_WATCHDOG_JOIN_END = "host.watchdog_join.end"
HOST_EMERGENCY_CLEANUP = "host.emergency_cleanup"
HOST_STOP_MARKER_REQUESTED = "host.stop_marker.requested"
HOST_STOP_MARKER_CONFIRMED = "host.stop_marker.confirmed"
HOST_STOP_MARKER_UNCONFIRMED = "host.stop_marker.unconfirmed"
HOST_MONITOR_TICK = "host.monitor.tick"
HOST_MONITOR_END = "host.monitor.end"

H_HOST_LOG_EVENTS: frozenset[str] = frozenset(
    {
        HOST_ALLOCATION_START,
        HOST_ALLOCATION_END,
        HOST_POD_ATTRIBUTED,
        HOST_POLL_TICK,
        HOST_PROGRESS_UNAVAILABLE,
        HOST_PROVIDER_TRANSIENT,
        HOST_TERMINAL_OBSERVED,
        HOST_TERMINAL_PROBE_TRANSIENT,
        HOST_PROVIDER_ABSENT_OBSERVED,
        HOST_STOP_HANDOFF,
        HOST_WATCHDOG_JOIN_START,
        HOST_WATCHDOG_JOIN_END,
        HOST_EMERGENCY_CLEANUP,
        HOST_STOP_MARKER_REQUESTED,
        HOST_STOP_MARKER_CONFIRMED,
        HOST_STOP_MARKER_UNCONFIRMED,
        HOST_MONITOR_TICK,
        HOST_MONITOR_END,
    }
)

SOURCE_R2_CHECKPOINT = "r2_checkpoint_event"
SOURCE_MLFLOW_METRIC = "mlflow_metric"
PROGRESS_SOURCES: frozenset[str] = frozenset({SOURCE_R2_CHECKPOINT, SOURCE_MLFLOW_METRIC})

STATE_STEP = "step"
STATE_NO_PROGRESS = "no_progress_yet"
STATE_RUN_NOT_FOUND = "run_not_found"
STATE_READ_FAILED = "read_failed"
STATE_LIST_FAILED = "list_failed"
PROGRESS_STATES: frozenset[str] = frozenset(
    {STATE_STEP, STATE_NO_PROGRESS, STATE_RUN_NOT_FOUND, STATE_READ_FAILED, STATE_LIST_FAILED}
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


def progress_fields(
    *, source: str, state: str, step: int | None = None, total: int | None = None
) -> dict[str, object]:
    """The ONE builder of a progress line's fields (controller poll and wave monitor).

    ``step`` exists exactly when ``state == "step"``: a non-step state never carries
    a number, so it can never be mistaken for the previous step or for 0. ``total``
    is the run's final global step when the caller knows it.
    """
    if source not in PROGRESS_SOURCES or state not in PROGRESS_STATES:
        raise ValueError("progress source and state must be declared")
    if (state == STATE_STEP) != (step is not None):
        raise ValueError("a progress step is present exactly when the state is 'step'")
    fields: dict[str, object] = {"source": source, "state": state}
    if step is not None:
        if type(step) is not int or step < 1:
            raise ValueError("a progress step is a positive global optimizer step")
        fields["step"] = step
    if total is not None:
        if type(total) is not int or total < 1:
            raise ValueError("a progress total is a positive global optimizer step")
        fields["total"] = total
    return fields


def checkpoint_steps(
    *, warmup_steps: int, steps: int, checkpoint_interval: int, evaluation_interval: int
) -> tuple[int, ...]:
    """Global optimizer steps at which the worker seals, hence publishes ``checkpoint-N``.

    MEASURED against ``ladder.run.runner`` (the seal condition ``final or evaluate or
    step % checkpoint_interval == 0``, with ``evaluate`` true only in the prediction
    phase) and its event name ``checkpoint-{total:08d}`` where the total is the step
    plus ``warmup_steps`` in the prediction phase (``object_worker.checkpoint``). The
    schedule is known from the request alone, so a reader can name the candidates
    without listing the namespace.
    """
    for value in (warmup_steps, steps, checkpoint_interval, evaluation_interval):
        if type(value) is not int or value < 0:
            raise ValueError("schedule bounds must be non-negative integers")
    if checkpoint_interval < 1 or evaluation_interval < 1:
        raise ValueError("schedule intervals must be positive")
    warmup = [
        s for s in range(1, warmup_steps + 1) if s == warmup_steps or s % checkpoint_interval == 0
    ]
    prediction = [
        warmup_steps + s
        for s in range(1, steps + 1)
        if s == steps or s % checkpoint_interval == 0 or s % evaluation_interval == 0
    ]
    return (*warmup, *prediction)
