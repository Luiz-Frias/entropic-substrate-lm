"""Journal-first D1 projection into the non-authoritative tracking plane."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from entropic_substrate.d1_tracking import D1MetricExtractor
from entropic_substrate.evidence.d1_journal import (
    AuthoritativeRunJournal,
    JournalTailQuarantineReceipt,
    read_journal,
)
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalEvent,
    FrozenJson,
    JournalCheckpointState,
)
from entropic_substrate.tracking_contracts import MetricExtractor

__all__ = [
    "AuthoritativeJournalOnlyObserver",
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
    """Strict slice of ``MlflowClient`` used by the D1 measurement mirror."""

    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        step: int = 0,
    ) -> object: ...


def _require_observer(observer: object) -> D1JournalObserver:
    if not isinstance(observer, D1JournalObserver):
        raise TypeError("observer must implement D1JournalObserver")
    return observer


class D1MlflowObserver:
    """Strictly write the registered scalar projection of one durable event.

    Unlike the legacy best-effort MLflow sink, this measurement-path adapter
    does not catch extractor or client errors. The caller receives the failure
    after the authoritative journal has already synchronized the source event.
    """

    __slots__ = ("_client", "_extractor", "_run_id")

    def __init__(
        self,
        *,
        client: object,
        run_id: str,
        extractor: object | None = None,
    ) -> None:
        if not isinstance(client, D1MlflowMetricClient):
            raise TypeError("client must implement the MLflow metric client contract")
        if type(run_id) is not str or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        selected: object = extractor if extractor is not None else D1MetricExtractor()
        if not isinstance(selected, MetricExtractor):
            raise TypeError("extractor must implement MetricExtractor")
        self._client = client
        self._run_id = run_id
        self._extractor = selected

    def observe(self, event: D1JournalEvent) -> None:
        extracted = self._extractor.extract(event)
        if extracted is None:
            return
        for key, value in extracted.metrics.items():
            self._client.log_metric(
                self._run_id,
                key,
                value,
                step=extracted.step,
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
