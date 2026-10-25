"""``StrictMlflowSink`` -- the STRICT :class:`~entropic_substrate.ladder.capture.ports.metric_sink.MetricSink` adapter.

Landed by FOR-1552 (design gate S1, 2026-10-22) beside, never inside,
:class:`~entropic_substrate.ladder.capture.adapters.mlflow.MlflowSink`. The two are
sibling implementations of one port with opposite failure semantics (Strategy beside
the port's other implementations): ``MlflowSink`` is best-effort after startup (a failed
write is counted and the run continues); this class is FAIL-LOUD (a failed write
raises on the caller's thread, unchanged, before the call returns). Strictness is a
type, not a flag, so the best-effort contract stays byte-identical. S2, a ``strict``
keyword on ``MlflowSink``, was ruled out: it would inherit the writer queue and thread,
the failed-writes counter and the startup probe's wait loop, then switch each off.

There is no worker thread, no queue, no counting and no flush: every ``MetricSink``
member either completes its write(s), legitimately has nothing to write (a well-formed
record with zero metrics), or raises. No ``except`` clause continues past a failure.
Every client method that accepts ``synchronous`` is called with ``synchronous=True``
explicitly: MEASURED against mlflow 3.15.0, ``synchronous=None`` resolves to
``not MLFLOW_ENABLE_ASYNC_LOGGING.get()``, so under that environment variable a failed
write returns a future instead of raising and a sink that omitted the keyword would be
strict only by ambient environment (same precedent as ``MlflowSink``'s batch writer).

Scope, deliberately narrow:

* The tracking client is a REQUIRED injected parameter typed by
  :class:`StrictMlflowClient`. This module imports nothing from ``mlflow`` at any scope
  (``.importlinter``'s ``mlflow-only-from-adapter`` contract) and builds no default
  client; a production client factory is outside this lane.
* It imports no D1 module. The D1 projection reaches it only through the generic
  scalar-projection record defined here (:func:`scalar_projection_record`).
* It does not own the run lifecycle. It writes to an EXISTING ``run_id`` and never
  creates or terminates a run, so :meth:`StrictMlflowSink.finish` issues no
  ``set_terminated``: the adapter that created the run owns that. It is therefore never
  a ``best_effort`` leg of a ``CompositeSink`` (a tracking blip would kill the run the
  wrapper exists to protect), and no source module constructs it in this lane.
* ``log`` carries exactly one record kind, the scalar projection. It applies no
  second allowlist or extractor; a record it cannot carry raises.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from entropic_substrate.ladder.capture.adapters.jsonl import (
    DEFAULT_FINISH_STATUS,
    require_annotation,
    require_finish_status,
)

__all__ = [
    "SCALAR_PROJECTION_KEY",
    "StrictMlflowClient",
    "StrictMlflowSink",
    "scalar_projection_record",
]

# The one place the projection record's shape is defined. A plain dict with
# JSON-serializable values: ``{"scalar_projection": {"step": int, "metrics": {key: float}}}``.
# It carries neither ``event`` nor ``schema``, so it is not recognizable as any existing
# record kind (run_start / train / eval / run_end / annotate, or an H schema row) and no
# schema-keyed reader can count it.
SCALAR_PROJECTION_KEY: Final = "scalar_projection"
_RECORD_FIELDS: Final = frozenset({SCALAR_PROJECTION_KEY})
_PROJECTION_FIELDS: Final = frozenset({"step", "metrics"})


def scalar_projection_record(step: int, metrics: Mapping[str, float]) -> dict[str, object]:
    """Build the record :meth:`StrictMlflowSink.log` carries: one step, its scalar metrics."""
    return {SCALAR_PROJECTION_KEY: {"step": step, "metrics": dict(metrics)}}


@runtime_checkable
class StrictMlflowClient(Protocol):
    """Slice of ``mlflow.tracking.MlflowClient`` the strict sink writes through.

    Declared here, not borrowed from ``D1MlflowMetricClient``, because it carries
    ``synchronous`` (which that retracted protocol lacks). Signatures MEASURED against
    mlflow 3.15.0: ``log_metric`` and ``set_tag`` accept ``synchronous``; ``log_artifact``
    does not.
    """

    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        *,
        step: int | None = None,
        synchronous: bool | None = None,
    ) -> object: ...

    def set_tag(
        self,
        run_id: str,
        key: str,
        value: str,
        *,
        synchronous: bool | None = None,
    ) -> object: ...

    def log_artifact(
        self,
        run_id: str,
        local_path: str,
        artifact_path: str | None = None,
    ) -> object: ...


def _require_client(client: object) -> StrictMlflowClient:
    if not isinstance(client, StrictMlflowClient):
        raise TypeError("client must implement the StrictMlflowClient contract")
    return client


def _has_exact_keys(mapping: Mapping[object, object], expected: frozenset[str]) -> bool:
    return len(mapping) == len(expected) and all(key in expected for key in mapping)


def _require_projection(record: object) -> tuple[int, dict[str, float]]:
    """Validate a scalar-projection record completely, before any write is issued."""
    if not isinstance(record, Mapping):
        raise TypeError(f"record must be a mapping, got {type(record).__name__}")
    if not _has_exact_keys(record, _RECORD_FIELDS):
        raise ValueError(
            f"record is not a scalar projection: keys {sorted(map(str, record))} "
            f"differ from [{SCALAR_PROJECTION_KEY!r}]"
        )
    body = record[SCALAR_PROJECTION_KEY]
    if not isinstance(body, Mapping):
        raise TypeError(f"{SCALAR_PROJECTION_KEY} must be a mapping, got {type(body).__name__}")
    if not _has_exact_keys(body, _PROJECTION_FIELDS):
        raise ValueError(
            f"{SCALAR_PROJECTION_KEY} fields {sorted(map(str, body))} "
            f"differ from {sorted(_PROJECTION_FIELDS)}"
        )
    step = body["step"]
    if type(step) is not int:
        raise TypeError(f"projection step must be an int, got {type(step).__name__}")
    if step < 0:
        raise ValueError(f"projection step must be non-negative, got {step}")
    metrics = body["metrics"]
    if not isinstance(metrics, Mapping):
        raise TypeError(f"projection metrics must be a mapping, got {type(metrics).__name__}")
    checked: dict[str, float] = {}
    for key, value in metrics.items():
        if type(key) is not str or not key.strip():
            raise ValueError(f"projection metric key must be a non-blank str, got {key!r}")
        if type(value) is not int and type(value) is not float:
            raise TypeError(
                f"projection metric {key!r} must be an int or float, got {type(value).__name__}"
            )
        checked[key] = float(value)
    return step, checked


class StrictMlflowSink:
    """FAIL-LOUD ``MetricSink``: every tracking-write failure raises, synchronously.

    See the module docstring for the contract. ``client`` and ``run_id`` are both
    required; the run must already exist and is owned by another adapter.
    """

    __slots__ = ("_client", "_finished", "_run_id")

    def __init__(self, *, client: StrictMlflowClient, run_id: str) -> None:
        if type(run_id) is not str or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        self._client = _require_client(client)
        self._run_id = run_id
        self._finished = False

    def _require_open(self) -> None:
        if self._finished:
            raise RuntimeError("StrictMlflowSink is finished; it accepts no further writes")

    def log(self, record: Mapping[str, object]) -> None:
        """Write one scalar projection: one ``log_metric`` per metric, at the record's step.

        The record is validated in full before the first write, so a malformed record
        reaches the client not at all. A client failure propagates from the first
        failing metric, unchanged; the metrics before it were already written.
        """
        step, metrics = _require_projection(record)
        self._require_open()
        for key, value in metrics.items():
            self._client.log_metric(self._run_id, key, value, step=step, synchronous=True)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        self._require_open()
        self._client.log_artifact(self._run_id, str(path), artifact_path=artifact_path)

    def annotate(self, tags: Mapping[str, str]) -> None:
        """Set one run tag per key (DN-1643-PORT), validated at entry before any write."""
        checked = require_annotation(tags)
        self._require_open()
        for key, value in checked.items():
            self._client.set_tag(self._run_id, key, value, synchronous=True)

    def finish(self, *, status: str = DEFAULT_FINISH_STATUS) -> None:
        """Refuse further writes. Idempotent; never terminates the run (not the owner).

        ``status`` is validated at entry (DN-1643-PORT), so a refused status changes no
        state. The first ``finish`` closes the sink; later calls are no-ops.
        """
        require_finish_status(status)
        self._finished = True
