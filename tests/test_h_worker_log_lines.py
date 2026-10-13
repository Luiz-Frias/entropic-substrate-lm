"""FOR-1637: the H worker narrates every mechanism through the log substrate.

Every assertion observes ONLY through the two channels an operator has: the
substrate's structured JSON file (``worker-log.jsonl``) and the human stderr
stream (``capsys``). Nothing here patches ``logger`` or reads module state, so
"RED" means the behaviour is absent, not that a symbol failed to import
(constraint C13: the vocabulary module is imported inside the comparing tests
only).

The closed-set test at the bottom is self-contained and order-independent
(constraint G17): it drives every emitter itself, shares no accumulator with a
sibling test, and compares the OBSERVED ``event`` names with the vocabulary
constant in both directions.
"""

from __future__ import annotations

import json
import threading
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_probe_operator_stop import _MarkerStore
from tests.test_sentence_execution import engine
from tests.test_sentence_worker_execution import prepared
from tests.test_sentence_worker_transport import _FlakyReadStore
from tests.test_sinks import FakeMlflowClient

from entropic_substrate.kernel.runtime.logging import configure_logging, logger
from entropic_substrate.ladder.capture.adapters.composite import build_sink_for_run
from entropic_substrate.ladder.capture.adapters.jsonl import JsonlSink
from entropic_substrate.ladder.capture.adapters.mlflow import MlflowSink
from entropic_substrate.ladder.capture.ports.run_descriptor import TrackingRunDescriptor
from entropic_substrate.ladder.run.object_worker import (
    encode_stop_marker,
    run_object_worker,
    stop_marker_key,
)
from entropic_substrate.ladder.run.runner import run_probe_training
from entropic_substrate.ladder.run.worker_command import run_worker_command
from entropic_substrate.ladder.run.worker_snapshot import publish_worker_snapshot
from entropic_substrate.ladder.run.worker_transport import (
    VerificationReadRetry,
    publish_probe_request,
    publish_verified_object,
)

pytestmark = pytest.mark.unit

_WRITER_THREAD_NAME = "mlflow-sink-writer"
_TERMINAL_MARKERS = {"complete", "paused", "stopped", "failure"}


# ---------------------------------------------------------------- observation
@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    """Where this test's structured log lands; ``bind_logs`` points the substrate at it."""
    return tmp_path / "obs" / "worker-log.jsonl"


def bind_logs(path: Path) -> None:
    """Re-bind the substrate's sinks to THIS test: current stderr + a JSON file.

    Called in the test BODY, never from a fixture: MEASURED 2026-10-12 on
    pytest 9.1.1, ``sys.stderr`` at fixture setup is not the object ``capsys``
    reads at call time, so sinks bound by the autouse conftest fixture (or by
    any fixture) write into a closed stream (the same reason
    ``tests/test_logging.py`` configures inside its tests).
    """
    configure_logging(json_path=path)


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def of(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("event") == name]


def index_of(rows: list[dict[str, Any]], name: str, **fields: object) -> int:
    for position, row in enumerate(rows):
        if row.get("event") == name and all(row.get(k) == v for k, v in fields.items()):
            return position
    raise AssertionError(f"no {name} row with {fields} in {[r.get('event') for r in rows]}")


def is_seconds(value: object) -> bool:
    return isinstance(value, float) and value >= 0.0


def worker_log_path(workspace: Path) -> Path:
    return workspace.parent / f"{workspace.name}.observability" / "worker-log.jsonl"


# --------------------------------------------------- (substrate) JSON projection
def test_json_sink_carries_bound_fields_but_never_the_clock_or_report_routing(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """G1: bound fields appear; ``local_ts``/``report`` never do, on any record."""
    bind_logs(log_file)
    logger.bind(event="probe.one", key="k/1", size=3).info("shown on stderr")
    logger.bind(report=True, event="probe.two", key="k/2").info("routed to the stdout report sink")
    rows = read_log(log_file)
    assert [r["event"] for r in rows] == ["probe.one", "probe.two"]
    assert rows[0]["key"] == "k/1" and rows[0]["size"] == 3
    for row in rows:
        assert "local_ts" not in row and "report" not in row, row
    assert "shown on stderr" in capsys.readouterr().err


def test_json_sink_never_lets_a_bound_field_overwrite_a_base_key(log_file: Path) -> None:
    bind_logs(log_file)
    logger.bind(
        event="probe.base", message="forged", level="CRITICAL", logger="forged", timestamp="forged"
    ).info("real")
    (row,) = read_log(log_file)
    assert row["event"] == "probe.base"  # the non-conflicting bound field IS carried
    assert row["message"] == "real" and row["level"] == "INFO"
    assert row["logger"] != "forged" and row["timestamp"] != "forged"


def test_json_sink_survives_non_scalar_bound_values_without_a_logging_error_block(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """G2: a Path / dict / set value neither raises, nor drops the record, nor
    makes loguru print its 'Logging error' block."""
    bind_logs(log_file)
    logger.bind(event="probe.shape", where=Path("/x/y"), blob={"a": {1, 2}}).info("odd values")
    (row,) = read_log(log_file)
    assert row["event"] == "probe.shape" and row["message"] == "odd values"
    assert row["where"] == "/x/y"
    assert "Logging error" not in capsys.readouterr().err


# ---------------------------------------------------------------- (a) the runner
def test_training_loop_narrates_phases_seals_and_evaluations(
    tmp_path: Path, log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bind_logs(log_file)
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    result = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
    )
    assert result.status == "training_complete"
    rows = read_log(log_file)

    starts = of(rows, "phase.start")
    ends = of(rows, "phase.end")
    assert [(r["phase"], r["step_from"], r["step_to"]) for r in starts] == [
        ("codec_warmup", 0, 2),
        ("prediction", 0, 4),
    ]
    assert [(r["phase"], r["step_from"], r["step_to"]) for r in ends] == [
        ("codec_warmup", 0, 2),
        ("prediction", 0, 4),
    ]
    assert all(is_seconds(r["seconds"]) for r in ends)

    seals = of(rows, "checkpoint.seal")
    assert [(r["phase"], r["step"], r["global_step"]) for r in seals] == [
        ("codec_warmup", 2, 2),
        ("prediction", 2, 4),
        ("prediction", 4, 6),
    ]
    assert all(is_seconds(r["seconds"]) for r in seals)

    eval_starts, eval_ends = of(rows, "evaluation.start"), of(rows, "evaluation.end")
    assert [r["global_step"] for r in eval_starts] == [4, 6]
    assert [r["global_step"] for r in eval_ends] == [4, 6]
    assert all(is_seconds(r["seconds"]) for r in eval_ends)

    # Ordering: a phase brackets its seals; an evaluation brackets itself.
    assert index_of(rows, "phase.start", phase="codec_warmup") < index_of(
        rows, "checkpoint.seal", phase="codec_warmup"
    )
    assert index_of(rows, "checkpoint.seal", phase="codec_warmup") < index_of(
        rows, "phase.end", phase="codec_warmup"
    )
    assert index_of(rows, "phase.end", phase="codec_warmup") < index_of(
        rows, "phase.start", phase="prediction"
    )
    assert index_of(rows, "evaluation.start", global_step=4) < index_of(
        rows, "evaluation.end", global_step=4
    )
    # Cardinality (C7): nothing per step. 6 steps ran; no event is per step.
    assert len(rows) == 2 + 2 + 3 + 2 + 2, [r.get("event") for r in rows]

    err = capsys.readouterr().err
    assert "phase.start phase=codec_warmup step_from=0 step_to=2" in err
    assert "checkpoint.seal phase=prediction step=4 global_step=6" in err
    assert "evaluation.end global_step=6" in err


# --------------------------------------------------------------- (b) R2 publishes
def test_publish_names_key_bytes_and_each_verification_read(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bind_logs(log_file)
    store = _ObjectStore()
    key = "k/{brace}.json"  # C1: a brace in a field must never reach a formatter
    identity = publish_verified_object(store, key, b"payload", retry=VerificationReadRetry())
    assert identity.size_bytes == 7
    rows = read_log(log_file)
    (start,) = of(rows, "publish.start")
    (verify,) = of(rows, "publish.verify_read")
    (end,) = of(rows, "publish.end")
    assert (start["key"], start["bytes"]) == (key, 7)
    assert (verify["key"], verify["attempt"], verify["max_attempts"]) == (key, 1, 3)
    assert is_seconds(verify["seconds"])
    assert (end["key"], end["bytes"]) == (key, 7) and is_seconds(end["seconds"])
    assert [r["event"] for r in rows] == ["publish.start", "publish.verify_read", "publish.end"]
    assert "publish.start key=k/{brace}.json bytes=7" in capsys.readouterr().err


def test_publish_without_a_retry_object_still_reports_its_single_verification_read(
    log_file: Path,
) -> None:
    bind_logs(log_file)
    publish_verified_object(_ObjectStore(), "k/plain.json", b"abc")
    rows = read_log(log_file)
    (verify,) = of(rows, "publish.verify_read")
    assert (verify["attempt"], verify["max_attempts"]) == (1, 1)
    assert [r["event"] for r in rows][-1] == "publish.end"


def test_a_failed_verification_read_is_a_warning_retry_and_changes_no_budget(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bind_logs(log_file)
    key = "k/flaky.json"
    store = _FlakyReadStore(fail_reads_remaining={key: 1})
    slept: list[float] = []
    retry = VerificationReadRetry(sleep=slept.append)
    publish_verified_object(store, key, b"payload", retry=retry)
    rows = read_log(log_file)
    (warning,) = of(rows, "publish.retry")
    assert warning["level"] == "WARNING"
    assert (warning["key"], warning["attempt"], warning["max_attempts"]) == (key, 1, 3)
    assert warning["reason"] == "_ReadFailure"
    (verify,) = of(rows, "publish.verify_read")
    assert verify["attempt"] == 2 and verify["level"] == "INFO"
    # Behaviour invariance (C8): the existing schedule and counters are untouched.
    assert slept == [0.05]
    assert retry.failures == 1
    assert store.publish_calls == [key]
    assert "WARNING" in capsys.readouterr().err


def test_an_exhausted_retry_budget_warns_between_attempts_and_ends_nothing(
    log_file: Path,
) -> None:
    bind_logs(log_file)
    key = "k/dead.json"
    store = _FlakyReadStore(fail_reads_remaining={key: 99})
    with pytest.raises(Exception, match="transient provider failure"):
        publish_verified_object(
            store, key, b"payload", retry=VerificationReadRetry(sleep=lambda _: None)
        )
    rows = read_log(log_file)
    assert [r["attempt"] for r in of(rows, "publish.retry")] == [1, 2]
    assert of(rows, "publish.end") == [] and of(rows, "publish.verify_read") == []
    assert len(of(rows, "publish.start")) == 1


def test_snapshot_publication_is_bracketed_and_every_object_put_is_reported(
    tmp_path: Path, log_file: Path
) -> None:
    bind_logs(log_file)
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "snapshot-worker"
    workspace.mkdir()
    run_object_worker(objects, marker, workspace, max_request_bytes=1_000_000)
    before = len(read_log(log_file))
    publish_worker_snapshot(
        objects, request, workspace, ("worker/start.json",), name="direct-probe"
    )
    rows = read_log(log_file)[before:]
    (start,) = of(rows, "snapshot.start")
    (end,) = of(rows, "snapshot.end")
    assert (start["name"], start["files"]) == ("direct-probe", 1)
    assert (end["name"], end["files"]) == ("direct-probe", 1) and is_seconds(end["seconds"])
    published = [r["key"] for r in of(rows, "publish.start")]
    assert published == [
        f"{request.prefix}/outputs/worker/start.json",
        f"{request.prefix}/snapshots/direct-probe.json",
    ]
    assert rows[0]["event"] == "snapshot.start" and rows[-1]["event"] == "snapshot.end"


# ------------------------------------------------------------- (c) object worker
def test_object_worker_reports_stages_snapshots_and_the_completion_marker_once(
    tmp_path: Path, log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bind_logs(log_file)
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "object-worker"
    workspace.mkdir()
    before = len(read_log(log_file))
    terminal = run_object_worker(objects, marker, workspace, max_request_bytes=1_000_000)
    rows = read_log(log_file)[before:]
    complete_key = f"{request.prefix}/complete.json"
    assert terminal.key == complete_key

    (written,) = of(rows, "completion.written")
    assert written["key"] == complete_key and written["marker"] == "complete"
    assert written["bytes"] == len(objects.values[complete_key])
    # Marker-last (D1 ruling): the line follows the put it reports.
    assert index_of(rows, "publish.end", key=complete_key) < index_of(rows, "completion.written")

    assert [r["name"] for r in of(rows, "snapshot.start")] == [
        "started",
        "checkpoint-00000002",
        "checkpoint-00000004",
        "checkpoint-00000006",
        "complete",
    ]
    assert len(of(rows, "snapshot.end")) == 5
    stages = of(rows, "stage.start")
    assert {r["stage"] for r in stages} == {
        "materialization",
        "runtime_admission",
        "training_readback",
        "generation",
        "span_diagnostics",
        "trace_capture",
        "native_verification",
        "result_admission",
    }
    assert sorted(r["stage"] for r in of(rows, "stage.end")) == sorted(r["stage"] for r in stages)
    assert all(is_seconds(r["seconds"]) for r in of(rows, "stage.end"))
    # Never for the non-terminal markers.
    assert not any(
        r["key"].endswith(("started.json",)) or "/checkpoint-" in r["key"]
        for r in of(rows, "completion.written")
    )
    assert "completion.written key=" in capsys.readouterr().err


def test_a_stop_marker_is_reported_when_observed_and_a_failed_poll_is_a_warning(
    tmp_path: Path, log_file: Path
) -> None:
    bind_logs(log_file)
    request, inputs = prepared(tmp_path)
    objects = _MarkerStore(
        control_key=stop_marker_key(request),
        control_payloads=(None, None, encode_stop_marker(request)),
        raises=1,
    )
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "stopped-worker"
    workspace.mkdir()
    terminal = run_object_worker(objects, marker, workspace, max_request_bytes=1_000_000)
    assert terminal.key == f"{request.prefix}/stopped.json"
    rows = read_log(log_file)

    (retry,) = of(rows, "stop_marker.retry")
    assert retry["level"] == "WARNING" and retry["key"] == stop_marker_key(request)
    assert (retry["attempt"], retry["max_attempts"]) == (1, 3)
    assert retry["reason"] == "RuntimeError"
    (observed,) = of(rows, "stop_marker.observed")
    assert observed["key"] == stop_marker_key(request) and observed["level"] == "INFO"
    # No per-poll noise (C7): two boundaries were polled, one marker was observed.
    assert not [r for r in rows if "stop_marker.polled" in json.dumps(r)]
    (written,) = of(rows, "completion.written")
    assert written["marker"] == "stopped"


# ------------------------------------------------------------------- (d) sinks
class _FailingBatchClient(FakeMlflowClient):
    def log_batch(self, *args: Any, **kwargs: Any) -> None:
        raise ConnectionError("simulated tracking outage during log_batch")


class _GatedClient(FakeMlflowClient):
    """``log_batch`` blocks on ``gate`` so a drain can be made to exceed its budget."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.entered = threading.Event()

    def log_batch(
        self,
        run_id: str,
        metrics: Any = (),
        params: Any = (),
        tags: Any = (),
        synchronous: bool | None = None,
    ) -> None:
        self.entered.set()
        self.gate.wait(5.0)
        super().log_batch(run_id, metrics, params, tags, synchronous)


def _mlflow_sink(client: FakeMlflowClient, **kwargs: Any) -> MlflowSink:
    return MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client, **kwargs
    )


def _eval(step: int) -> dict[str, object]:
    return {"event": "eval", "step": step, "loss": float(step)}


def _exceed_the_drain_budget() -> MlflowSink:
    client = _GatedClient()
    sink = _mlflow_sink(client, drain_timeout_seconds=0.05)
    sink.log(_eval(1))
    assert client.entered.wait(5.0), "the writer never picked record 1 up"
    sink.log(_eval(2))
    sink.log(_eval(3))
    try:
        sink.finish()
    finally:
        client.gate.set()
        for thread in threading.enumerate():
            if thread.name == _WRITER_THREAD_NAME:
                thread.join(5.0)
    return sink


def test_composite_finish_reports_each_sink_start_and_end_in_authoritative_first_order(
    tmp_path: Path, log_file: Path
) -> None:
    bind_logs(log_file)
    client = FakeMlflowClient()
    sink = build_sink_for_run(
        TrackingRunDescriptor(run_name="r", tiny=True, params={}, tags={}),
        results_path=tmp_path / "metrics.jsonl",
        client=client,
        log_path=log_file,
    )
    sink.log(_eval(1))
    sink.finish()
    sink.finish()  # idempotent: no second pair
    rows = read_log(log_file)
    pairs = [(r["event"], r["sink"]) for r in rows if str(r.get("event", "")).startswith("sink.")]
    assert pairs == [
        ("sink.finish.start", "jsonl"),
        ("sink.finish.end", "jsonl"),
        ("sink.finish.start", "mlflow"),
        ("sink.finish.end", "mlflow"),
    ]
    ends = of(rows, "sink.finish.end")
    assert [(r["settled"], r["stranded"]) for r in ends] == [(True, 0), (True, 0)]
    assert all(is_seconds(r["seconds"]) for r in ends)


def test_a_bare_jsonl_sink_reports_its_own_finish_once(log_file: Path, tmp_path: Path) -> None:
    bind_logs(log_file)
    sink = JsonlSink(tmp_path / "metrics.jsonl")
    sink.finish()
    sink.finish()
    rows = read_log(log_file)
    assert [r["event"] for r in rows] == ["sink.finish.start", "sink.finish.end"]
    assert rows[1]["sink"] == "jsonl"


def test_finish_uploads_the_worker_log_beside_the_results_and_only_when_declared(
    tmp_path: Path, log_file: Path
) -> None:
    """M4's unit arm: the declared log reaches ``results/`` and nothing else does."""
    bind_logs(log_file)
    results = tmp_path / "metrics.jsonl"
    declared = FakeMlflowClient()
    sink = build_sink_for_run(
        TrackingRunDescriptor(run_name="r", tiny=True, params={}, tags={}),
        results_path=results,
        client=declared,
        log_path=log_file,
    )
    sink.log(_eval(1))
    sink.finish()
    run_id = sink.best_effort[0].run_id  # type: ignore[attr-defined]
    assert declared.artifacts_logged == [
        (run_id, str(results), "results"),
        (run_id, str(log_file), "results"),
    ]

    plain = FakeMlflowClient()
    other = build_sink_for_run(
        TrackingRunDescriptor(run_name="r2", tiny=True, params={}, tags={}),
        results_path=tmp_path / "metrics-2.jsonl",
        client=plain,
    )
    other.log(_eval(1))
    other.finish()
    assert [path for _, path, _ in plain.artifacts_logged] == [str(tmp_path / "metrics-2.jsonl")]


def test_a_failed_log_upload_is_counted_warned_and_never_skips_termination(
    tmp_path: Path, log_file: Path
) -> None:
    bind_logs(log_file)

    class _LogUploadDown(FakeMlflowClient):
        def log_artifact(self, run_id: str, path: str, artifact_path: str | None = None) -> None:
            if path.endswith("worker-log.jsonl"):
                raise ConnectionError("simulated artifact store outage")
            super().log_artifact(run_id, path, artifact_path)

    client = _LogUploadDown()
    sink = _mlflow_sink(client, results_path=tmp_path / "metrics.jsonl", log_path=log_file)
    sink.finish()
    assert sink.failed_writes == 1
    run = client._runs[sink.run_id]
    assert run.data.tags["mlflow.status"] == "FINISHED"
    assert run.data.tags["mlflow_failed_writes"] == "1"
    (failed,) = of(read_log(log_file), "sink.write_failed")
    assert (failed["sink"], failed["count"], failed["level"]) == ("mlflow", 1, "WARNING")


def test_a_declared_log_that_is_missing_warns_without_counting_or_raising(
    tmp_path: Path, log_file: Path
) -> None:
    bind_logs(log_file)
    client = FakeMlflowClient()
    sink = _mlflow_sink(client, log_path=tmp_path / "never-written.jsonl")
    sink.finish()
    assert sink.failed_writes == 0
    assert client._runs[sink.run_id].data.tags["mlflow.status"] == "FINISHED"


def test_the_log_path_may_not_alias_the_authoritative_results_file(tmp_path: Path) -> None:
    same = tmp_path / "metrics.jsonl"
    with pytest.raises(ValueError, match="log_path"):
        build_sink_for_run(
            TrackingRunDescriptor(run_name="r", tiny=True, params={}, tags={}),
            results_path=same,
            log_path=same,
        )


def test_an_exceeded_drain_is_a_warning_event_with_its_budget_and_stranded_count(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bind_logs(log_file)
    sink = _exceed_the_drain_budget()
    assert sink.failed_writes == 3
    rows = read_log(log_file)
    (exceeded,) = of(rows, "sink.drain_exceeded")
    assert exceeded["level"] == "WARNING" and exceeded["sink"] == "mlflow"
    assert exceeded["budget_seconds"] == 0.05 and exceeded["stranded"] == 3
    (end,) = of(rows, "sink.finish.end")
    assert (end["settled"], end["stranded"]) == (False, 3)
    err = capsys.readouterr().err
    # G3: the pre-existing sentence is converted in place, one record, same text.
    drain_lines = [line for line in err.splitlines() if "drain" in line]
    assert len(drain_lines) == 1, drain_lines
    assert "3 undelivered record(s)" in drain_lines[0] and "not joined" in drain_lines[0]


def test_a_failed_metric_batch_is_a_write_failed_warning_with_its_record_count(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bind_logs(log_file)
    client = _FailingBatchClient()
    sink = _mlflow_sink(client)
    sink.log(_eval(1))
    assert sink.flush() is True
    sink.finish()
    rows = read_log(log_file)
    failed = [r for r in of(rows, "sink.write_failed") if "metric batch" in r["message"]]
    assert [(r["sink"], r["count"], r["level"]) for r in failed] == [("mlflow", 1, "WARNING")]
    assert len([ln for ln in capsys.readouterr().err.splitlines() if "metric batch" in ln]) == 1


def test_the_tracking_run_line_is_an_event_and_never_carries_uri_credentials(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bind_logs(log_file)
    sink = MlflowSink(
        tracking_uri="https://operator:hunter2@mlflow.example:5000/base",  # pragma: allowlist secret
        experiment_name="exp",
        run_name="run",
        client=FakeMlflowClient(),
    )
    sink.finish()
    (row,) = of(read_log(log_file), "mlflow.run")
    assert row["run_id"] == sink.run_id
    assert row["uri"] == "https://mlflow.example:5000/base"
    assert "hunter2" not in log_file.read_text() and "hunter2" not in capsys.readouterr().err


# --------------------------------------------------------- (worker.start / end)
def _run_args(workspace: Path, tmp_path: Path, marker: Any) -> list[str]:
    return [
        "run",
        "--repo-root",
        str(tmp_path),
        "--max-object-bytes",
        "10000000",
        "--max-objects",
        "100",
        "--workspace-root",
        str(workspace),
        "--request-key",
        marker.key,
        "--request-sha256",
        marker.sha256,
        "--request-size",
        str(marker.size_bytes),
        "--local-tiny",
    ]


def test_run_command_brackets_the_worker_and_keeps_stdout_and_stderr_channels_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "command-worker"
    workspace.mkdir()
    out, err = StringIO(), StringIO()
    status = run_worker_command(
        _run_args(workspace, tmp_path, marker), environ={}, objects=objects, stdout=out, stderr=err
    )
    assert status == 0, err.getvalue()
    # G6: the injected channels stay byte-identical to the pre-FOR-1637 contract.
    assert err.getvalue() == ""
    assert json.loads(out.getvalue())["status"] == "worker_returned"

    rows = read_log(worker_log_path(workspace))
    (start,) = of(rows, "worker.start")
    assert start["request_sha256"] == request.digest
    assert "image" in start and "code_commit" in start
    (end,) = of(rows, "worker.end")
    assert end["status"] == "ok" and is_seconds(end["seconds"])
    # worker.start leads the file; worker.end is logged AFTER sink.finish() and
    # therefore is not in the file that finish() uploaded (C15) -- it is the
    # human stream's last word.
    assert rows[0]["event"] == "worker.start"
    assert index_of(rows, "worker.end") == len(rows) - 1
    assert index_of(rows, "sink.finish.end", sink="jsonl") < index_of(rows, "worker.end")
    assert "local_ts" not in json.dumps(rows) and '"report"' not in json.dumps(rows)
    assert "worker.start request_sha256=" in capsys.readouterr().err


def test_a_failed_run_ends_the_worker_exactly_once_with_its_stage_and_error_type(
    tmp_path: Path,
) -> None:
    class _BrokenCheckpoint(_ObjectStore):
        def publish(self, key: str, payload: bytes, *, expected_sha256: str) -> Any:
            if key.endswith("/snapshots/checkpoint-00000002.json"):
                raise RuntimeError("injected checkpoint upload failure")
            return super().publish(key, payload, expected_sha256=expected_sha256)

    request, inputs = prepared(tmp_path)
    objects = _BrokenCheckpoint()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "failing-worker"
    workspace.mkdir()
    out, err = StringIO(), StringIO()
    status = run_worker_command(
        _run_args(workspace, tmp_path, marker), environ={}, objects=objects, stdout=out, stderr=err
    )
    assert status == 2
    assert json.loads(err.getvalue())["stage"] == "worker_execution"
    rows = read_log(worker_log_path(workspace))
    (end,) = of(rows, "worker.end")
    assert end["status"] == "failed" and end["stage"] == "worker_execution"
    assert end["error_type"] == "RuntimeError"
    assert "injected checkpoint upload failure" not in json.dumps(rows)  # C9: no str(exc)
    assert index_of(rows, "sink.finish.end", sink="jsonl") < index_of(rows, "worker.end")


# --------------------------------------------------------- (e) closed set, both ways
def test_the_event_vocabulary_is_closed_both_ways(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The constant equals the set of events the emitters actually produce.

    Self-contained (G17): drives every emitter in this function, observes only
    through the JSON files, shares nothing with sibling tests. The standalone
    emitters run FIRST under one JSON file; ``run_worker_command`` then
    re-points the substrate (by design, G5) at its own ``worker-log.jsonl``;
    the observed set is the union of the two files.
    """
    from entropic_substrate.ladder.run.worker_log_events import H_WORKER_LOG_EVENTS

    standalone = tmp_path / "standalone.jsonl"
    configure_logging(json_path=standalone)

    # mlflow.run, sink.finish.*, sink.write_failed (a failed batch), sink.drain_exceeded
    failing = _mlflow_sink(_FailingBatchClient())
    failing.log(_eval(1))
    assert failing.flush() is True
    failing.finish()
    _exceed_the_drain_budget()

    # publish.* including publish.retry
    key = "k/closed-set.json"
    publish_verified_object(
        _FlakyReadStore(fail_reads_remaining={key: 1}),
        key,
        b"payload",
        retry=VerificationReadRetry(sleep=lambda _: None),
    )

    # stop_marker.retry, stop_marker.observed, completion.written (stopped), stage/snapshot
    (tmp_path / "stop").mkdir()
    request, inputs = prepared(tmp_path / "stop")
    stopping = _MarkerStore(
        control_key=stop_marker_key(request),
        control_payloads=(None, None, encode_stop_marker(request)),
        raises=1,
    )
    stop_marker = publish_probe_request(stopping, request, inputs)
    stop_workspace = tmp_path / "stop-worker"
    stop_workspace.mkdir()
    run_object_worker(stopping, stop_marker, stop_workspace, max_request_bytes=1_000_000)

    # worker.start/end, phase.*, checkpoint.seal, evaluation.*, completion.written (complete)
    (tmp_path / "full").mkdir()
    request2, inputs2 = prepared(tmp_path / "full")
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request2, inputs2)
    workspace = tmp_path / "full-worker"
    workspace.mkdir()
    status = run_worker_command(
        _run_args(workspace, tmp_path, marker),
        environ={},
        objects=objects,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    assert status == 0

    observed = {
        row["event"]
        for path in (standalone, worker_log_path(workspace))
        for row in read_log(path)
        if "event" in row
    }
    assert isinstance(H_WORKER_LOG_EVENTS, frozenset)
    assert observed - H_WORKER_LOG_EVENTS == set(), "an emitted event is not in the vocabulary"
    assert H_WORKER_LOG_EVENTS - observed == set(), "a declared event is never emitted"
    capsys.readouterr()
