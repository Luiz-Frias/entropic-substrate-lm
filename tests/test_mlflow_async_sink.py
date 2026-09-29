"""FOR-1622: ``MlflowSink`` metric writes run OFF the caller's (training) thread.

Contract pinned here (brief ``pre-for-1622``, frozen_after true):

* ``log()`` never performs a client write on the caller's thread: it extracts
  (still on the caller, so extractor failures are counted synchronously) and
  enqueues; ONE background worker drains a BOUNDED queue into ``log_batch``.
* A failed batch counts every record it carried in ``failed_writes`` and warns
  once per batch; ``log()`` never raises for network reasons.
* A dead worker is reported at the next ``log()`` and at ``finish()``, never
  swallowed; its undelivered records are counted.
* A full queue BLOCKS the producer, never drops: MLflow must never disagree
  with the authoritative JSONL by omission.
* ``flush(timeout)`` waits for every logged record to settle (join
  semantics, bounded) WITHOUT stopping the worker, so a writer that hangs past
  a failed flush and later recovers still delivers everything logged after it.
* ``finish()`` drains within its injectable budget (``drain_timeout_seconds``)
  before the existing finish sequence, so ``set_terminated`` never races queued
  metrics; then it stops and joins the worker, or, past the budget, counts the
  stranded records and reports the writer it could not join.

Every test injects a fake client (the ``client=`` seam): no thread in this
file ever talks to a real ``MlflowClient``. The autouse fixture below fails
any test that leaves a writer thread alive, so every worker a test starts is
joined inside that test (``finish()`` joins it; ``flush()`` does not).

Latches, not sleeps: the gated fake blocks inside ``log_batch`` on an Event the
test opens. ``_LATCH_BUDGET_S`` only bounds a wait that a correct
implementation satisfies immediately; it exists so a broken implementation
fails RED instead of hanging the suite. ``_OBSERVE_S`` is the one observation
window: it gives a MUTANT (a finish() that skips the drain, a log() that drops
on a full queue) time to show itself; a correct implementation passes it by
construction, whatever the host load.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from entropic_substrate.ladder.capture.ports.run_descriptor import (
    ExtractedMetrics,
    LegacyMetricExtractor,
)
from entropic_substrate.ladder.capture.reader.sinks import MlflowSink
from test_sinks import FakeMlflowClient

pytestmark = pytest.mark.unit

_WRITER_THREAD_NAME = "mlflow-sink-writer"
_LATCH_BUDGET_S = 5.0
_OBSERVE_S = 0.2


@pytest.fixture(autouse=True)
def _no_writer_thread_outlives_its_test() -> Iterator[None]:
    yield
    alive = [t.name for t in threading.enumerate() if t.name == _WRITER_THREAD_NAME]
    assert alive == [], f"a test leaked its MlflowSink writer thread: {alive}"


def _eval(step: int, **metrics: float) -> dict[str, object]:
    return {"event": "eval", "step": step, **(metrics or {"loss": float(step)})}


def _per_metric_shape(records: list[dict[str, object]]) -> list[tuple[str, float, int]]:
    """What the pre-FOR-1622 per-metric ``log_metric`` loop would have sent."""
    extractor = LegacyMetricExtractor()
    shape: list[tuple[str, float, int]] = []
    for record in records:
        extracted = extractor.extract(record)
        assert extracted is not None
        shape.extend((key, value, extracted.step) for key, value in extracted.metrics.items())
    return shape


class _GatedClient(FakeMlflowClient):
    """``log_batch`` records the thread it ran on, then blocks on ``gate``.

    ``log_metric`` also records its thread, so a mutant that writes per metric
    on the caller's thread is caught as well as one that calls ``log_batch``
    synchronously.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.terminated = threading.Event()
        self.completed = threading.Event()
        self.write_threads: list[int] = []
        self.batches: list[list[tuple[str, float, int]]] = []
        self.completed_batches = 0
        self.metrics_delivered_at_terminate: int | None = None

    def log_batch(
        self,
        run_id: str,
        metrics: Any = (),
        params: Any = (),
        tags: Any = (),
        synchronous: bool | None = None,
    ) -> None:
        self.write_threads.append(threading.get_ident())
        self.entered.set()
        self.gate.wait(_LATCH_BUDGET_S)
        self._before_write(metrics)
        self.batches.append([(m.key, m.value, m.step) for m in metrics])
        super().log_batch(run_id, metrics, params, tags, synchronous)
        self.completed_batches += 1
        self.completed.set()

    def _before_write(self, metrics: Any) -> None:
        """Hook for failure-injecting subclasses."""

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        self.write_threads.append(threading.get_ident())
        super().log_metric(run_id, key, value, step)

    def set_terminated(self, run_id: str, status: str = "FINISHED") -> None:
        self.metrics_delivered_at_terminate = sum(len(batch) for batch in self.batches)
        self.terminated.set()
        super().set_terminated(run_id, status)


def _sink(client: FakeMlflowClient, **kwargs: Any) -> MlflowSink:
    return MlflowSink(
        tracking_uri="http://fake",
        experiment_name="exp",
        run_name="run",
        client=client,
        **kwargs,
    )


def _writer_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == _WRITER_THREAD_NAME]


# ------------------------------------------------------------------ (a) off-thread
def test_log_returns_before_the_client_write_completes_and_never_writes_on_the_caller() -> None:
    client = _GatedClient()
    sink = _sink(client)
    caller = threading.get_ident()
    try:
        sink.log(_eval(1))
        # The gate is still closed: a log() that waited for the write could
        # not have returned with zero completed batches.
        assert client.completed_batches == 0
        assert caller not in client.write_threads
        assert client.entered.wait(_LATCH_BUDGET_S), "the worker never picked the record up"
    finally:
        client.gate.set()
        sink.finish()
    assert client.completed_batches == 1
    assert client.write_threads, "the record never reached the client"
    assert caller not in client.write_threads


def test_construction_starts_no_writer_thread_until_the_first_log() -> None:
    client = FakeMlflowClient()
    sink = _sink(client)
    assert _writer_threads() == [], "an untracked-until-first-log run must pay nothing"
    sink.log(_eval(1))
    assert len(_writer_threads()) == 1
    sink.finish()


def test_a_record_without_metrics_starts_no_writer_thread() -> None:
    client = FakeMlflowClient()
    sink = _sink(client)
    sink.log({"event": "train", "step": 1, "loss": 9.9})
    assert _writer_threads() == []
    sink.finish()


# ------------------------------------------------------------ (b) finish() drains
def test_finish_drains_every_queued_metric_batched_in_the_per_metric_shape() -> None:
    client = _GatedClient()
    sink = _sink(client)
    records = [
        _eval(step, loss=float(step), nats_per_byte=step / 10, perplexity=step * 2.0)
        for step in range(1, 5)
    ]
    sink.log(records[0])
    assert client.entered.wait(_LATCH_BUDGET_S), "the worker never picked record 1 up"
    for record in records[1:]:
        sink.log(record)  # queued behind the closed gate: one coalesced batch next

    finisher = threading.Thread(target=sink.finish)
    finisher.start()
    # A finish() that skips the drain reaches set_terminated inside this
    # window while record 1 is still held at the closed gate.
    client.terminated.wait(_OBSERVE_S)
    client.gate.set()
    finisher.join(_LATCH_BUDGET_S)
    assert not finisher.is_alive(), "finish() did not return after the gate opened"

    expected = _per_metric_shape(records)
    assert [metric for batch in client.batches for metric in batch] == expected
    assert len(client.batches) == 2, "records queued behind a busy worker must coalesce"
    assert client.metrics_delivered_at_terminate == len(expected)
    assert sink.failed_writes == 0
    run = client._runs[sink.run_id]
    assert run.data.tags["mlflow.status"] == "FINISHED"


def test_a_record_larger_than_the_mlflow_batch_limit_is_chunked_at_1000() -> None:
    class _Wide:
        def extract(self, event: object) -> ExtractedMetrics | None:
            return ExtractedMetrics(step=7, metrics={f"m/{i:04d}": float(i) for i in range(2500)})

    client = _GatedClient()
    client.gate.set()
    sink = _sink(client, metric_extractor=_Wide())
    sink.log({"event": "wide"})
    sink.finish()
    assert [len(batch) for batch in client.batches] == [1000, 1000, 500]
    assert all(step == 7 for batch in client.batches for _, _, step in batch)
    assert sink.failed_writes == 0


# ------------------------------------------------------ (c) failed batch accounting
class _FailingBatchClient(_GatedClient):
    def _before_write(self, metrics: Any) -> None:
        msg = "simulated tracking outage during log_batch"
        raise ConnectionError(msg)


def test_a_failed_batch_counts_every_record_it_carried_and_never_raises(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _FailingBatchClient()
    sink = _sink(client)
    sink.log(_eval(1))  # batch 1: one record
    assert client.entered.wait(_LATCH_BUDGET_S)
    sink.log(_eval(2))  # batch 2: two records, coalesced behind the gate
    sink.log(_eval(3))
    client.gate.set()
    assert sink.flush() is True  # drained: every record settled, delivered or counted
    assert sink.failed_writes == 3
    warnings = [line for line in capsys.readouterr().err.splitlines() if "metric batch" in line]
    assert len(warnings) == 2, f"one warning per failed batch, got {warnings}"
    sink.finish()
    assert client._runs[sink.run_id].data.tags["mlflow_failed_writes"] == "3"


def test_a_failed_chunk_counts_its_record_once_even_when_other_chunks_land() -> None:
    class _Wide:
        def extract(self, event: object) -> ExtractedMetrics | None:
            return ExtractedMetrics(step=1, metrics={f"m/{i:04d}": float(i) for i in range(2500)})

    class _SecondChunkFails(_GatedClient):
        calls = 0

        def _before_write(self, metrics: Any) -> None:
            self.calls += 1
            if self.calls == 2:
                msg = "simulated failure of chunk 2 of 3"
                raise ConnectionError(msg)

    client = _SecondChunkFails()
    client.gate.set()
    sink = _sink(client, metric_extractor=_Wide())
    sink.log({"event": "wide"})
    sink.finish()
    assert [len(batch) for batch in client.batches] == [1000, 500]
    assert sink.failed_writes == 1


# ------------------------------------------------------------- (d) dead worker
class _WorkerKilled(BaseException):
    """Not an ``Exception``: escapes the per-batch guard and kills the worker."""


class _KillingClient(_GatedClient):
    def _before_write(self, metrics: Any) -> None:
        raise _WorkerKilled


def _kill_via_client(monkeypatch: pytest.MonkeyPatch) -> FakeMlflowClient:
    client = _KillingClient()
    client.gate.set()
    return client


def _kill_via_worker_loop(monkeypatch: pytest.MonkeyPatch) -> FakeMlflowClient:
    import mlflow.entities

    def _crash(*args: object, **kwargs: object) -> None:
        msg = "simulated crash inside the worker loop itself"
        raise RuntimeError(msg)

    monkeypatch.setattr(mlflow.entities, "Metric", _crash)
    return FakeMlflowClient()


@pytest.mark.parametrize("arrange", [_kill_via_client, _kill_via_worker_loop])
def test_a_dead_worker_is_reported_at_the_next_log_and_at_finish(
    arrange: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = arrange(monkeypatch)
    sink = _sink(client)
    sink.log(_eval(1))
    assert sink.flush() is False, "a dead worker must not read as a clean drain"
    assert sink.failed_writes == 1, "the in-flight record is counted, not lost"
    capsys.readouterr()
    before = {t.ident for t in _writer_threads()}

    sink.log(_eval(2))  # must not raise, must not silently restart the worker
    assert {t.ident for t in _writer_threads()} <= before, "a dead writer was restarted"
    assert sink.failed_writes == 2
    at_log = [line for line in capsys.readouterr().err.splitlines() if "worker died" in line]
    assert len(at_log) == 1, at_log

    sink.log(_eval(3))
    assert sink.failed_writes == 3
    again = [line for line in capsys.readouterr().err.splitlines() if "worker died" in line]
    assert again == [], "reported once at log(), not once per record"

    sink.finish()
    at_finish = [line for line in capsys.readouterr().err.splitlines() if "worker died" in line]
    assert len(at_finish) == 1, at_finish
    assert client._runs[sink.run_id].data.tags["mlflow_failed_writes"] == "3"
    assert client._runs[sink.run_id].data.tags["mlflow.status"] == "FINISHED"


# ---------------------------------------------------------- (e) backpressure
def test_a_full_queue_blocks_the_producer_until_drained_and_never_drops() -> None:
    client = _GatedClient()
    sink = _sink(client, max_queued_records=1)
    sink.log(_eval(1))
    assert client.entered.wait(_LATCH_BUDGET_S), "the worker never picked record 1 up"
    sink.log(_eval(2))  # fills the single queue slot

    returned = threading.Event()

    def _produce() -> None:
        sink.log(_eval(3))
        returned.set()

    producer = threading.Thread(target=_produce)
    producer.start()
    try:
        assert not returned.wait(_OBSERVE_S), "log() returned on a full queue: it dropped"
    finally:
        client.gate.set()
        producer.join(_LATCH_BUDGET_S)
    assert returned.is_set(), "the producer never unblocked after the drain"
    sink.finish()
    delivered_steps = [step for batch in client.batches for _, _, step in batch]
    assert delivered_steps == [1, 2, 3]
    assert sink.failed_writes == 0


@pytest.mark.parametrize("bound", [0, -1])
def test_an_unbounded_or_negative_queue_bound_is_refused(bound: int) -> None:
    with pytest.raises(ValueError, match="max_queued_records"):
        _sink(FakeMlflowClient(), max_queued_records=bound)


@pytest.mark.parametrize("budget", [0.0, -1.0, float("inf"), float("nan")])
def test_a_non_positive_or_non_finite_drain_budget_is_refused(budget: float) -> None:
    with pytest.raises(ValueError, match="drain_timeout_seconds"):
        _sink(FakeMlflowClient(), drain_timeout_seconds=budget)


# ------------------------------------------------ stale stop (conductor ruling 2)
def test_a_writer_that_recovers_after_a_failed_flush_still_delivers_later_records() -> None:
    """A timed-out ``flush`` must leave nothing behind that later stops the
    writer: once the hung write returns, every record logged afterwards is
    delivered at ``finish()``."""
    client = _GatedClient()
    sink = _sink(client, drain_timeout_seconds=_LATCH_BUDGET_S)
    sink.log(_eval(1))
    assert client.entered.wait(_LATCH_BUDGET_S), "the worker never picked record 1 up"
    assert sink.flush(0.05) is False, "record 1 is still held at the closed gate"
    client.gate.set()
    assert client.completed.wait(_LATCH_BUDGET_S), "the hung write never returned"

    sink.log(_eval(2))
    sink.log(_eval(3))
    sink.finish()

    assert [step for batch in client.batches for _, _, step in batch] == [1, 2, 3]
    assert sink.failed_writes == 0
    assert client._runs[sink.run_id].data.tags["mlflow.status"] == "FINISHED"


# --------------------------------------------- finish() abandon (conductor ruling 3)
def test_finish_past_its_drain_budget_counts_stranded_records_and_still_terminates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _GatedClient()
    sink = _sink(client, drain_timeout_seconds=0.05)
    sink.log(_eval(1))
    assert client.entered.wait(_LATCH_BUDGET_S), "the worker never picked record 1 up"
    sink.log(_eval(2))  # queued behind the closed gate
    sink.log(_eval(3))
    capsys.readouterr()
    try:
        sink.finish()  # the gate is still closed: the drain cannot finish in 0.05 s

        assert sink.failed_writes == 3, "every unsettled record, in flight or queued"
        run = client._runs[sink.run_id]
        assert run.data.tags["mlflow_failed_writes"] == "3"
        assert run.data.tags["mlflow.status"] == "FINISHED"
        budget = [line for line in capsys.readouterr().err.splitlines() if "drain" in line]
        assert len(budget) == 1, budget
        assert "3 undelivered record(s)" in budget[0]
        assert "not joined" in budget[0], "an unjoined writer must be reported"
    finally:
        client.gate.set()
        for thread in _writer_threads():
            thread.join(_LATCH_BUDGET_S)
    assert _writer_threads() == [], "the abandoned writer must stop once its write returns"
    assert sink.failed_writes == 3, "a late settle must not count the records twice"


# ------------------------------------------------------------ (f) idempotence
def test_finish_twice_after_logging_is_idempotent_and_joins_the_worker() -> None:
    client = FakeMlflowClient()
    sink = _sink(client)
    sink.log(_eval(1))
    sink.finish()
    assert _writer_threads() == []
    sink.finish()
    assert client._runs[sink.run_id].data.metrics == {"loss": 1.0}
    assert client._runs[sink.run_id].data.tags["mlflow.status"] == "FINISHED"


def test_flush_with_nothing_logged_is_a_clean_no_op() -> None:
    sink = _sink(FakeMlflowClient())
    assert sink.flush() is True
    assert _writer_threads() == []
    sink.finish()


def test_flush_drains_without_stopping_the_worker_and_finish_joins_it() -> None:
    client = FakeMlflowClient()
    sink = _sink(client)
    sink.log(_eval(1))
    assert sink.flush() is True
    assert client._runs[sink.run_id].data.metrics == {"loss": 1.0}
    assert len(_writer_threads()) == 1, "flush() drains; only finish() stops the writer"
    sink.log(_eval(2))
    sink.finish()
    assert _writer_threads() == []
    metrics: Mapping[str, float] = client._runs[sink.run_id].data.metrics
    assert metrics == {"loss": 2.0}
    assert sink.failed_writes == 0
