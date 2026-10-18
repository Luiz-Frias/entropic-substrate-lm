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

import ast
import json
import threading
import time
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
from entropic_substrate.ladder.run import object_worker
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.object_worker import (
    STOP_MARKER_SCHEMA,
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


class _Unprintable:
    """A bound value whose ``str()`` raises: the sink must still write the record."""

    def __str__(self) -> str:
        raise RuntimeError("str() deliberately fails")


def _no_json_constant(name: str) -> object:
    raise AssertionError(f"non-standard JSON constant {name} reached worker-log.jsonl")


def test_json_sink_renders_non_finite_floats_as_strings_never_as_json_constants(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """E5d: NaN/inf are not JSON; a bare NaN token would break every strict reader."""
    bind_logs(log_file)
    logger.bind(event="probe.float", ratio=float("nan"), top=float("inf")).info("non-finite")
    (line,) = log_file.read_text().splitlines()
    row = json.loads(line, parse_constant=_no_json_constant)
    assert row["event"] == "probe.float"
    assert row["ratio"] == "nan" and row["top"] == "inf"
    assert "Logging error" not in capsys.readouterr().err


def test_json_sink_keeps_the_record_when_a_bound_value_cannot_be_printed(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """E5e: a ``str()`` that raises degrades to the type name; the row still lands."""
    bind_logs(log_file)
    logger.bind(event="probe.opaque", opaque=_Unprintable()).info("unprintable value")
    (row,) = read_log(log_file)
    assert row["event"] == "probe.opaque" and row["message"] == "unprintable value"
    assert isinstance(row["opaque"], str) and "_Unprintable" in row["opaque"]
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
    # The final +2 is the FOR-1662 completion admission pair: exactly one
    # admission.start and one admission.end for the whole run, none for the
    # mid-training checkpoint walks (and this run passes no on_checkpoint).
    completion_admission_pair = 2
    assert len(rows) == 2 + 2 + 3 + 2 + 2 + completion_admission_pair, [
        r.get("event") for r in rows
    ]
    (admitted,) = of(rows, "admission.end")
    assert (admitted["caller"], admitted["root"], admitted["memo"]) == (
        "completion",
        "run",
        "fresh",
    )
    assert index_of(rows, "evaluation.end", global_step=6) < index_of(rows, "admission.start")

    err = capsys.readouterr().err
    assert "phase.start phase=codec_warmup step_from=0 step_to=2" in err
    assert "checkpoint.seal phase=prediction step=4 global_step=6" in err
    assert "evaluation.end global_step=6" in err


def test_evaluation_start_is_on_disk_while_the_evaluation_runs(
    log_file: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # START-BEFORE-BLOCK (C6): evaluation.start must already be on disk inside evaluate().
    bind_logs(log_file)
    original = SentenceExecution.evaluate
    seen: list[list[str]] = []

    def spy(self: SentenceExecution) -> Any:
        seen.append(
            [r["event"] for r in read_log(log_file) if r.get("event", "").startswith("evaluation.")]
        )
        return original(self)

    monkeypatch.setattr(SentenceExecution, "evaluate", spy)
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
    )
    assert seen == [
        ["evaluation.start"],
        ["evaluation.start", "evaluation.end", "evaluation.start"],
    ]


def test_checkpoint_seal_is_logged_after_the_l1_window_closes(
    log_file: Path, tmp_path: Path
) -> None:
    """FOR-1567 timing comparability: no emission inside the L1 window.

    ``publication_timing["l1_seconds"]`` is assigned exactly when the L1 window
    closes; at each such assignment the seal rows already on disk must number
    0, 1, 2: the seal line for checkpoint N is written strictly after it.
    """
    bind_logs(log_file)
    at_close: list[int] = []

    class _Timing(dict[str, float]):
        def __setitem__(self, name: str, value: float) -> None:
            if name == "l1_seconds":
                at_close.append(len(of(read_log(log_file), "checkpoint.seal")))
            super().__setitem__(name, value)

    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
        publication_timing=_Timing(),
    )
    assert at_close == [0, 1, 2]
    assert len(of(read_log(log_file), "checkpoint.seal")) == 3


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


def test_publish_start_is_on_disk_while_the_put_is_in_flight(log_file: Path) -> None:
    # START-BEFORE-BLOCK (C6): the start row must already be on disk while the put is in flight.
    bind_logs(log_file)
    seen: list[list[str]] = []

    class Spy(_ObjectStore):
        def publish(self, key: str, payload: bytes, *, expected_sha256: str) -> Any:
            seen.append([r["event"] for r in read_log(log_file) if r.get("key") == key])
            return super().publish(key, payload, expected_sha256=expected_sha256)

    publish_verified_object(Spy(), "k/a.json", b"x", retry=VerificationReadRetry())
    assert seen == [["publish.start"]]


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


def test_a_publish_that_raises_after_landing_the_object_is_a_warning_retry_then_recovers(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The path a real R2 post-PUT readback failure takes: ``objects.publish`` itself raises."""
    bind_logs(log_file)
    key = "k/publish-raised.json"

    class _RaisesAfterLanding(_ObjectStore):
        def publish(self, key: str, payload: bytes, *, expected_sha256: str) -> Any:
            super().publish(key, payload, expected_sha256=expected_sha256)
            raise RuntimeError("injected post-put readback failure")

    slept: list[float] = []
    retry = VerificationReadRetry(sleep=slept.append)
    identity = publish_verified_object(_RaisesAfterLanding(), key, b"payload", retry=retry)
    assert identity.key == key
    rows = read_log(log_file)
    (warning,) = of(rows, "publish.retry")
    assert warning["level"] == "WARNING"
    assert (warning["key"], warning["attempt"], warning["max_attempts"]) == (key, 1, 3)
    assert warning["reason"] == "RuntimeError"
    assert "injected post-put readback failure" not in json.dumps(rows)  # C9: no str(exc)
    assert retry.failures == 1  # behaviour invariance (C8): the failure was already counted
    assert index_of(rows, "publish.retry") < index_of(rows, "publish.end")
    # S1: the raised publish already spent attempt 1, so the recovery read is 2 of 3.
    (verify,) = of(rows, "publish.verify_read")
    assert (verify["key"], verify["attempt"], verify["max_attempts"]) == (key, 2, 3)
    assert "WARNING" in capsys.readouterr().err


def test_a_failed_recovery_read_after_a_raised_publish_numbers_its_attempts_from_the_publish(
    log_file: Path,
) -> None:
    """S1b: publish raised (attempt 1), the first recovery read fails (attempt 2),
    the next read succeeds as attempt 3 of 3: every number continues the same sequence."""
    bind_logs(log_file)
    key = "k/publish-raised-then-flaky.json"

    class _RaisesThenFlaky(_FlakyReadStore):
        def publish(self, key: str, payload: bytes, *, expected_sha256: str) -> Any:
            super().publish(key, payload, expected_sha256=expected_sha256)
            raise RuntimeError("injected post-put readback failure")

    store = _RaisesThenFlaky(fail_reads_remaining={key: 1})
    retry = VerificationReadRetry(sleep=lambda _: None)
    publish_verified_object(store, key, b"payload", retry=retry)
    rows = read_log(log_file)
    warnings = of(rows, "publish.retry")
    assert [(w["attempt"], w["max_attempts"], w["reason"]) for w in warnings] == [
        (1, 3, "RuntimeError"),
        (2, 3, "_ReadFailure"),
    ]
    (verify,) = of(rows, "publish.verify_read")
    assert (verify["attempt"], verify["max_attempts"]) == (3, 3)
    assert retry.failures == 2  # behaviour invariance (C8): one per failed attempt


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


def test_stage_start_is_on_disk_while_the_stage_body_runs(log_file: Path) -> None:
    # START-BEFORE-BLOCK (C6): stage.start must already be on disk while the stage body runs.
    # Imported here, not at module level (C13): on the parent tree the symbol does not
    # exist, and a module-level import would turn the whole file into one collection error.
    from entropic_substrate.ladder.run.worker_execution import narrated_stage

    bind_logs(log_file)
    with narrated_stage("probe"):
        inside = [r["event"] for r in read_log(log_file)]
    assert inside == ["stage.start"]


def test_a_raising_stage_logs_no_end_and_the_exception_propagates(log_file: Path) -> None:
    # C5: .end is logged only after the operation returns; a raising stage logs no .end.
    from entropic_substrate.ladder.run.worker_execution import narrated_stage

    bind_logs(log_file)
    with pytest.raises(RuntimeError, match="stage body failed"):
        with narrated_stage("probe"):
            raise RuntimeError("stage body failed")
    assert [r["event"] for r in read_log(log_file)] == ["stage.start"]


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


def test_the_stop_poll_backs_off_between_failed_reads_on_the_existing_schedule(
    tmp_path: Path, log_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # C8: one failed stop-marker poll sleeps exactly one backoff, on the retry schedule.
    slept: list[float] = []
    monkeypatch.setattr(
        object_worker, "VerificationReadRetry", lambda: VerificationReadRetry(sleep=slept.append)
    )
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
    run_object_worker(objects, marker, workspace, max_request_bytes=1_000_000)
    assert slept == [0.05]


def test_a_present_marker_for_another_request_is_never_reported_as_observed(
    tmp_path: Path, log_file: Path
) -> None:
    # Negative arm (classify_stop_marker False): present, canonical, but another request's digest.
    bind_logs(log_file)
    request, inputs = prepared(tmp_path)
    foreign = json.dumps(
        {"schema": STOP_MARKER_SCHEMA, "request_sha256": "0" * 64}, sort_keys=True
    ).encode()
    objects = _MarkerStore(control_key=stop_marker_key(request), control_payloads=(foreign,))
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "foreign-marker-worker"
    workspace.mkdir()
    terminal = run_object_worker(objects, marker, workspace, max_request_bytes=1_000_000)
    rows = read_log(log_file)
    assert terminal.key == f"{request.prefix}/complete.json"
    assert of(rows, "stop_marker.observed") == []
    assert of(rows, "stop_marker.retry") == []
    assert [r["marker"] for r in of(rows, "completion.written")] == ["complete"]


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


class _GatedFailingClient(FakeMlflowClient):
    """``log_batch`` blocks on ``gate`` and then rejects the batch: while it is
    blocked on record 1, records 2 and 3 queue up and are coalesced into ONE batch."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.entered = threading.Event()

    def log_batch(self, *args: Any, **kwargs: Any) -> None:
        self.entered.set()
        self.gate.wait(5.0)
        raise ConnectionError("simulated tracking outage during log_batch")


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


def test_mlflow_finish_start_is_on_disk_while_the_drain_is_blocked(log_file: Path) -> None:
    # START-BEFORE-BLOCK (C6): sink.finish.start must already be on disk while the drain blocks.
    bind_logs(log_file)
    client = _GatedClient()
    sink = _mlflow_sink(client, drain_timeout_seconds=30.0)
    sink.log(_eval(1))
    assert client.entered.wait(5.0)
    finisher = threading.Thread(target=sink.finish)
    finisher.start()
    deadline = time.monotonic() + 3.0
    started = False
    while time.monotonic() < deadline and not started:
        started = any(
            r.get("event") == "sink.finish.start" and r.get("sink") == "mlflow"
            for r in read_log(log_file)
        )
        time.sleep(0.01)
    client.gate.set()
    finisher.join(15.0)
    assert started, "sink.finish.start was not on disk while the drain was still blocked"


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
    tmp_path: Path, log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bind_logs(log_file)
    client = FakeMlflowClient()
    sink = _mlflow_sink(client, log_path=tmp_path / "never-written.jsonl")
    sink.finish()
    assert sink.failed_writes == 0
    assert client._runs[sink.run_id].data.tags["mlflow.status"] == "FINISHED"
    assert client.artifacts_logged == []  # nothing was uploaded
    # The warning itself (brief: "missing file is a WARNING, never a raise"). It is a
    # deliberate non-event: the closed vocabulary is not widened for an absent file,
    # so the JSON row carries no `event` key (the closed-set test sees only bound names).
    assert "worker log file" in capsys.readouterr().err
    (warned,) = [r for r in read_log(log_file) if "worker log file" in r.get("message", "")]
    assert warned["level"] == "WARNING" and "event" not in warned


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


def test_a_coalesced_failing_batch_reports_how_many_records_it_carried(
    log_file: Path,
) -> None:
    """E16b: records 2 and 3 are one rejected batch, so ``count`` is 2, not 1."""
    bind_logs(log_file)
    client = _GatedFailingClient()
    sink = _mlflow_sink(client)
    try:
        sink.log(_eval(1))
        assert client.entered.wait(5.0), "the writer never picked record 1 up"
        sink.log(_eval(2))
        sink.log(_eval(3))
    finally:
        client.gate.set()
    assert sink.flush() is True
    sink.finish()
    failed = [
        r for r in of(read_log(log_file), "sink.write_failed") if "metric batch" in r["message"]
    ]
    assert [r["count"] for r in failed] == [1, 2]
    assert sink.failed_writes == 3
    assert "(2 record(s)" in failed[1]["message"]


def test_the_tracking_run_line_drops_query_and_fragment_from_the_uri(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """E8b: a token in the query string or fragment is as readable as userinfo."""
    bind_logs(log_file)
    sink = MlflowSink(
        tracking_uri="https://mlflow.example:5000/base?token=q-secret-1#frag-secret-2",  # pragma: allowlist secret
        experiment_name="exp",
        run_name="run",
        client=FakeMlflowClient(),
    )
    sink.finish()
    (row,) = of(read_log(log_file), "mlflow.run")
    assert row["uri"] == "https://mlflow.example:5000/base"
    on_disk, on_stderr = log_file.read_text(), capsys.readouterr().err
    for leaked in ("q-secret-1", "frag-secret-2", "token="):
        assert leaked not in on_disk and leaked not in on_stderr, leaked


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


class ReadTimeout(Exception):
    """A transport read timeout: recognised by TYPE NAME, like the real one."""


class _ColdOnce(FakeMlflowClient):
    def __init__(self) -> None:
        super().__init__()
        self.cold_replies = 1

    def get_experiment_by_name(self, name: str) -> Any:
        if self.cold_replies:
            self.cold_replies -= 1
            raise ReadTimeout("read timed out")
        return super().get_experiment_by_name(name)


def test_the_cold_start_retry_warning_never_carries_uri_credentials(
    log_file: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CodeRabbit on PR 341: the warning lands in worker-log.jsonl, which finish() uploads."""
    bind_logs(log_file)
    monkeypatch.setattr(
        "entropic_substrate.ladder.capture.adapters.mlflow.time.sleep", lambda _s: None
    )
    sink = MlflowSink(
        tracking_uri="https://operator:hunter2@mlflow.example:5000/base?token=q-secret-1#frag-secret-2",  # pragma: allowlist secret
        experiment_name="exp",
        run_name="run",
        client=_ColdOnce(),
    )
    sink.finish()
    on_disk, on_stderr = log_file.read_text(), capsys.readouterr().err
    assert "https://mlflow.example:5000/base answered but read-timed-out" in on_stderr
    for leaked in ("hunter2", "operator", "q-secret-1", "frag-secret-2", "token="):
        assert leaked not in on_disk and leaked not in on_stderr, leaked


# ------------------------------------------- (d) write-failed sites on their rows
class _ArtifactsDown(FakeMlflowClient):
    def log_artifact(self, run_id: str, path: str, artifact_path: str | None = None) -> None:
        raise ConnectionError("simulated artifact store outage")


def _write_failed(rows: list[dict[str, Any]], needle: str) -> dict[str, Any]:
    (row,) = [r for r in of(rows, "sink.write_failed") if needle in r["message"]]
    return row


def test_a_log_after_finish_is_a_write_failed_warning_on_the_json_row(log_file: Path) -> None:
    bind_logs(log_file)
    sink = _mlflow_sink(FakeMlflowClient())
    sink.finish()
    sink.log(_eval(1))
    row = _write_failed(read_log(log_file), "after finish()")
    assert (row["sink"], row["count"], row["level"]) == ("mlflow", 1, "WARNING")


def test_a_dead_metric_writer_is_a_write_failed_warning_on_the_json_row(log_file: Path) -> None:
    bind_logs(log_file)
    sink = _mlflow_sink(FakeMlflowClient())
    sink._worker_error = RuntimeError("boom")
    sink.log(_eval(1))
    row = _write_failed(read_log(log_file), "writer worker died")
    assert (row["sink"], row["count"], row["level"]) == ("mlflow", 1, "WARNING")


def test_an_extractor_failure_is_a_write_failed_warning_on_the_json_row(log_file: Path) -> None:
    class _Boom:
        def extract(self, record: object) -> object:
            raise RuntimeError("boom")

    bind_logs(log_file)
    sink = _mlflow_sink(FakeMlflowClient(), metric_extractor=_Boom())
    sink.log(_eval(1))
    row = _write_failed(read_log(log_file), "metric write failed")
    assert (row["sink"], row["count"], row["level"]) == ("mlflow", 1, "WARNING")


def test_a_failed_artifact_upload_is_a_write_failed_warning_on_the_json_row(
    log_file: Path, tmp_path: Path
) -> None:
    bind_logs(log_file)
    sink = _mlflow_sink(_ArtifactsDown())
    sink.log_artifact(tmp_path / "blob.bin")
    row = _write_failed(read_log(log_file), "artifact upload failed")
    assert (row["sink"], row["count"], row["level"]) == ("mlflow", 1, "WARNING")


def test_a_failed_results_upload_is_a_write_failed_warning_on_the_json_row(
    log_file: Path, tmp_path: Path
) -> None:
    results = tmp_path / "metrics.jsonl"
    results.write_text("{}\n")
    bind_logs(log_file)
    sink = _mlflow_sink(_ArtifactsDown(), results_path=results)
    sink.finish()
    row = _write_failed(read_log(log_file), "results JSONL upload failed")
    assert (row["sink"], row["count"], row["level"]) == ("mlflow", 1, "WARNING")


def test_an_unparseable_tracking_uri_is_rendered_as_a_placeholder_never_its_secret(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bind_logs(log_file)
    sink = MlflowSink(
        tracking_uri="http://operator:hunter2@[::1",  # pragma: allowlist secret
        experiment_name="exp",
        run_name="run",
        client=FakeMlflowClient(),
    )
    sink.finish()
    (row,) = of(read_log(log_file), "mlflow.run")
    assert row["uri"] == "<unparseable tracking uri>"
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
    human = capsys.readouterr().err
    assert "Logging error" not in human, human
    assert "worker.start request_sha256=" in human


def test_worker_start_carries_the_request_image_and_code_commit_unswapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """S2: a local-tiny request has no image/commit (both None), so a swapped pair
    would be invisible. Give the admitted request two DISTINCT values (set past the
    dataclass validation, as a remote request would carry them) and stop the run
    right after ``worker.start``."""
    from entropic_substrate.ladder.run import worker_command

    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    image = "ghcr.io/luiz-frias/entropic-substrate-lm@sha256:" + "d" * 64
    commit = "c" * 40
    object.__setattr__(request, "image", image)
    object.__setattr__(request, "code_commit", commit)
    monkeypatch.setattr(worker_command, "read_published_probe_request", lambda *_a, **_k: request)

    def stop_here(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("stop right after worker.start")

    monkeypatch.setattr(worker_command, "build_sink_for_run", stop_here)
    workspace = tmp_path / "start-fields-worker"
    workspace.mkdir()
    status = run_worker_command(
        _run_args(workspace, tmp_path, marker),
        environ={},
        objects=objects,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    assert status == 2
    (start,) = of(read_log(worker_log_path(workspace)), "worker.start")
    assert start["image"] == request.image == image
    assert start["code_commit"] == request.code_commit == commit
    assert f"image={image}" in capsys.readouterr().err


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


def test_a_stdout_write_that_fails_after_the_worker_ended_does_not_end_it_twice(
    tmp_path: Path,
) -> None:
    """C15: ``worker.end`` exactly once, even when the final response write raises."""

    class _BadOut(StringIO):
        def write(self, text: str, /) -> int:
            raise OSError("stdout closed")

    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "bad-stdout-worker"
    workspace.mkdir()
    status = run_worker_command(
        _run_args(workspace, tmp_path, marker),
        environ={},
        objects=objects,
        stdout=_BadOut(),
        stderr=StringIO(),
    )
    assert status == 2  # the refusal envelope is still returned
    ends = of(read_log(worker_log_path(workspace)), "worker.end")
    assert [r["status"] for r in ends] == ["ok"]


def test_run_command_declares_the_worker_log_to_the_sink_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production wiring: without ``log_path`` the log never reaches MLflow.

    Every other test builds the sink itself; this one pins that the worker command
    passes the file it writes (the sibling ``.observability`` directory, never a
    path under the workspace root) to ``build_sink_for_run``.
    """
    from entropic_substrate.ladder.run import worker_command

    declared: list[dict[str, Any]] = []
    real = worker_command.build_sink_for_run

    def spy(*args: Any, **kwargs: Any) -> Any:
        declared.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(worker_command, "build_sink_for_run", spy)
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "wiring-worker"
    workspace.mkdir()
    status = run_worker_command(
        _run_args(workspace, tmp_path, marker),
        environ={},
        objects=objects,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    assert status == 0
    (kwargs,) = declared
    assert kwargs["log_path"] == worker_log_path(workspace)
    assert kwargs["results_path"] == worker_log_path(workspace).with_name("metrics.jsonl")


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


# ----------------------------------------------------------- (g) layer edge (C10)
def test_ladder_capture_never_imports_the_run_layer() -> None:
    """C10: the capture adapters never import ``ladder.run``; they emit literal event names.

    ``import-linter`` cannot see this edge: ``capture`` and ``run`` both sit inside
    the single ``ladder`` layer. This static walk resolves every import under
    ``ladder/capture`` (relative and function-scope ones included) and fails on any
    that lands in ``entropic_substrate.ladder.run``.
    """
    src = Path(__file__).resolve().parents[1] / "src"
    forbidden = "entropic_substrate.ladder.run"
    scanned = 0
    offenders: list[str] = []
    for path in sorted((src / "entropic_substrate" / "ladder" / "capture").rglob("*.py")):
        scanned += 1
        package = path.relative_to(src).with_suffix("").parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                head = package[: len(package) - (node.level - 1)] if node.level else ()
                tail = tuple(node.module.split(".")) if node.module else ()
                targets = [".".join((*head, *tail))]
                targets += [".".join((*head, *tail, alias.name)) for alias in node.names]
            else:
                continue
            offenders += [
                f"{path.relative_to(src)}:{node.lineno} -> {target}"
                for target in targets
                if target == forbidden or target.startswith(forbidden + ".")
            ]
    assert scanned > 0, "the capture walk found no modules"
    assert offenders == []
