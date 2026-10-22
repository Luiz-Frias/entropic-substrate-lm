"""Journal-first D1 projection into the non-authoritative tracking plane."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from entropic_substrate.ladder.capture.adapters.d1_tracking import D1MetricExtractor
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.capture.ports.run_descriptor import ExtractedMetrics
from entropic_substrate.ladder.capture.records.d1_journal import (
    AuthoritativeRunJournal,
    JournalTailQuarantineReceipt,
    read_journal,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    FrozenJson,
    JournalCheckpointState,
)

_SCALAR_PROJECTION_KEY = "scalar_projection"

__all__ = [
    "AuthoritativeJournalOnlyObserver",
    "D1EventMetricExtractor",
    "D1JournalObserver",
    "D1MlflowMetricClient",
    "D1MlflowObserver",
    "ObservedD1Journal",
]


class AuthoritativeJournalOnlyObserver:
    """Explicit null projection; the durable journal remains authoritative."""

    def observe(self, event: D1JournalEvent) -> None:
        del event


@runtime_checkable
class D1JournalObserver(Protocol):
    """Synchronous observer called only after a journal event is durable."""

    def observe(self, event: D1JournalEvent) -> None: ...


@runtime_checkable
class D1MlflowMetricClient(Protocol):
    """Strict slice of ``MlflowClient`` used by the D1 measurement mirror.

    RETRACTED 2026-10-22 (FOR-1552): ``D1MlflowObserver`` no longer takes this client.
    It takes a :class:`~entropic_substrate.ladder.capture.ports.metric_sink.MetricSink`,
    and the fail-loud MLflow path is
    :class:`~entropic_substrate.ladder.capture.adapters.mlflow_strict.StrictMlflowSink`,
    whose own client protocol carries ``synchronous`` (this one cannot). Kept public and
    unchanged for one read-old window so existing importers keep resolving; nothing in
    ``src`` consumes it. Retirement trigger, checkable: delete this class and its
    ``__all__`` entry in the first cut where
    ``grep -rn D1MlflowMetricClient src tests`` finds no hit outside this module and
    the ``mlflow_strict.py`` docstring.
    """

    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        step: int = 0,
    ) -> object: ...


@runtime_checkable
class D1EventMetricExtractor(Protocol):
    """Strategy projecting one durable D1 journal event into optional scalar metrics.

    Typed on :class:`D1JournalEvent`, so it is NOT the record-shaped
    ``MetricExtractor`` that ``build_sink_for_run`` consumes. The runtime
    ``isinstance`` against this class checks only that ``extract`` exists; it
    cannot tell this protocol apart from that one.
    """

    def extract(self, event: D1JournalEvent) -> ExtractedMetrics | None: ...


def _require_observer(observer: object) -> D1JournalObserver:
    if not isinstance(observer, D1JournalObserver):
        raise TypeError("observer must implement D1JournalObserver")
    return observer


def _require_sink(sink: object) -> MetricSink:
    if not isinstance(sink, MetricSink):
        raise TypeError("sink must implement the MetricSink port")
    return sink


def _require_extractor(extractor: object) -> D1EventMetricExtractor:
    if not isinstance(extractor, D1EventMetricExtractor):
        raise TypeError("extractor must implement MetricExtractor")
    return extractor


class D1MlflowObserver:
    """Project the registered scalars of one durable event onto a ``MetricSink``.

    The observer adds no guard and swallows nothing: an extractor failure or a sink
    failure reaches the caller after the authoritative journal has already synchronized
    the source event. Whether a tracking failure is loud therefore depends on the sink
    behind it. The measurement path is fail-loud only behind
    :class:`~entropic_substrate.ladder.capture.adapters.mlflow_strict.StrictMlflowSink`.
    Behind ``MlflowSink`` the projection record is dropped UNCOUNTED: MEASURED 2026-10-22,
    its default ``LegacyMetricExtractor`` and ``HTrainingStepExtractor`` both extract the
    projection to ``None``, and ``MlflowSink.log`` returns on ``None`` without touching
    ``_failed_writes``. ``JsonlSink`` appends the record verbatim.

    RETRACTED 2026-10-22 (FOR-1552): the earlier sentence "a best-effort sink counts and
    continues, by its own contract" is false for this record; it holds only for a write
    the sink accepted and then failed.

    RETRACTED 2026-10-22 (FOR-1552): the earlier docstring claimed this adapter, unlike
    the best-effort MLflow sink, did not catch client errors. That held only for the
    removed ``client=`` composition; with an arbitrary ``MetricSink`` it is false, so the
    claim is withdrawn rather than restated.
    """

    __slots__ = ("_extractor", "_sink")

    def __init__(
        self,
        *,
        sink: MetricSink,
        extractor: D1EventMetricExtractor | None = None,
    ) -> None:
        self._sink = _require_sink(sink)
        self._extractor = _require_extractor(
            extractor if extractor is not None else D1MetricExtractor()
        )

    def observe(self, event: D1JournalEvent) -> None:
        extracted = self._extractor.extract(event)
        if extracted is None:
            return
        # The shape StrictMlflowSink validates, built here so this module imports no
        # adapter: a plain mapping, no ``schema`` key, no ``event`` key. Drift from the
        # sink's key is pinned by tests, not by a shared import.
        self._sink.log(
            {_SCALAR_PROJECTION_KEY: {"step": extracted.step, "metrics": dict(extracted.metrics)}}
        )


class ObservedD1Journal:
    """Facade enforcing durable journal append before synchronous observation."""

    __slots__ = ("_journal", "_observer", "_path")

    def __init__(
        self,
        journal: AuthoritativeRunJournal,
        path: str | Path,
        observer: D1JournalObserver,
    ) -> None:
        if type(journal) is not AuthoritativeRunJournal:
            raise TypeError("journal must be an AuthoritativeRunJournal")
        self._journal = journal
        self._path = Path(path)
        self._observer = _require_observer(observer)

    @classmethod
    def fresh(
        cls,
        path: str | Path,
        start: D1JournalEvent,
        *,
        observer: D1JournalObserver,
    ) -> ObservedD1Journal:
        """Create, synchronize, then observe the canonical ``run_start`` event."""
        checked = _require_observer(observer)
        journal = AuthoritativeRunJournal.fresh(path, start)
        observed = cls(journal, path, checked)
        try:
            checked.observe(start)
        except BaseException:
            journal.close()
            raise
        return observed

    @classmethod
    def resume(
        cls,
        path: str | Path,
        checkpoint: JournalCheckpointState,
        *,
        checkpoint_sha256: str,
        segment_provenance: Mapping[str, FrozenJson],
        quarantine_path: Path | None = None,
        observer: D1JournalObserver,
    ) -> ObservedD1Journal:
        """Resume durably, then observe the journal-created ``run_resume`` event."""
        checked = _require_observer(observer)
        journal = AuthoritativeRunJournal.resume(
            path,
            checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            segment_provenance=segment_provenance,
            quarantine_path=quarantine_path,
        )
        observed = cls(journal, path, checked)
        try:
            checked.observe(read_journal(path)[-1])
        except BaseException:
            journal.close()
            raise
        return observed

    @property
    def state(self) -> JournalCheckpointState:
        return self._journal.state

    @property
    def tail_quarantine(self) -> JournalTailQuarantineReceipt | None:
        return self._journal.tail_quarantine

    @property
    def latest_checkpoint_sha256(self) -> str | None:
        return self._journal.latest_checkpoint_sha256

    def append(self, event: D1JournalEvent) -> JournalCheckpointState:
        state = self._journal.append(event)
        self._observer.observe(event)
        return state

    def append_checkpoint(self, checkpoint_sha256: str) -> JournalCheckpointState:
        state = self._journal.append_checkpoint(checkpoint_sha256)
        self._observer.observe(read_journal(self._path)[-1])
        return state

    def close(self) -> None:
        self._journal.close()

    def __enter__(self) -> ObservedD1Journal:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
