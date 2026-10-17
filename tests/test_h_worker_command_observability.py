"""The H worker command builds one MetricSink per run and routes committed
training-step rows through it (RUNG 2b, conductor CONTINUATION #7).

MLFLOW_TRACKING_URI is unset in this process (unit-test regime), so
``build_sink_for_run`` composes a bare JsonlSink -- the observability mirror
is exercised end to end without a live server, and its JSONL bytes are
asserted directly. The D1 custody snapshot's own file set
(``run``/``capture``/``verification``/``worker``) is untouched by this --
the sink writes to a sibling ``observability/`` directory that
``_paths()``'s walked sections never enumerate.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_execution import prepared
from tests.test_sentence_worker_transport import _FlakyReadStore, _ReadFailure

from entropic_substrate.kernel.runtime.logging import configure_logging
from entropic_substrate.ladder.capture.adapters.jsonl import JsonlSink
from entropic_substrate.ladder.capture.reader.h_telemetry import final_h_run_end
from entropic_substrate.ladder.run import object_worker
from entropic_substrate.ladder.run.object_worker import (
    encode_stop_marker,
    run_object_worker,
    stop_marker_key,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
)
from entropic_substrate.ladder.run.worker_command import run_worker_command
from entropic_substrate.ladder.run.worker_transport import publish_probe_request

pytestmark = pytest.mark.unit


def test_successful_run_command_routes_committed_steps_through_a_jsonl_sink(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "command-worker"
    workspace.mkdir()
    output, error = StringIO(), StringIO()

    status = run_worker_command(
        [
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
        ],
        environ={},
        objects=objects,
        stdout=output,
        stderr=error,
    )

    assert status == 0, error.getvalue()
    metrics_path = tmp_path / "command-worker.observability" / "metrics.jsonl"
    assert metrics_path.exists(), "worker command must build and use a MetricSink"
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    assert records, "no committed H training-step rows reached the sink"
    # RESTATED 2026-09-26 (exit-gate lane, operator ruling "adding evaluation
    # rows to the sink is a small change"): the sink now also mirrors each
    # sealed evaluation report, so the training assertions filter on their
    # schema and the evaluation rows are asserted present, not tolerated.
    training = [r for r in records if r.get("schema") == "h-training-step-v2"]
    evaluations = [r for r in records if r.get("schema") == "h-predictive-evaluation-v1"]
    assert training, "no committed H training-step rows reached the sink"
    assert evaluations, "no sealed evaluation reports reached the sink"
    # RESTATED 2026-10-14 (FOR-1639): the worker also writes its two
    # h-run-lifecycle-v1 records through the sink, so totality now counts them.
    # The all-schemas guard is kept, not dropped: every record is one of the
    # three known schemas.
    lifecycle = [r for r in records if r.get("schema") == "h-run-lifecycle-v1"]
    # RESTATED 2026-10-15 (FOR-1642, DN-1643-PORT): the worker also annotates each sealed
    # checkpoint through the sink; those rows are `{"event": "annotate", ...}` and carry
    # no `schema` key. The totality guard is kept with a fourth, named bucket.
    annotations = [r for r in records if r.get("event") == "annotate"]
    assert annotations and all("schema" not in r for r in annotations)
    assert len(training) + len(evaluations) + len(lifecycle) + len(annotations) == len(records)
    # FOR-1639: exactly one run_start, first, and one run_end, last.
    assert [r["event"] for r in lifecycle] == ["run_start", "run_end"]
    assert records[0]["event"] == "run_start"
    assert records[-1]["event"] == "run_end"
    start, end = lifecycle
    assert start["profile_identity"] and start["run_class"] == "local_tiny"
    assert start["request_sha256"] == request.digest
    assert start["runtime"]["kind"] == "local_tiny"
    assert end["status"] == "ok" and end["result_status"] == "probe_workload_complete"
    assert end["successful_optimizer_steps"] == len(training)
    assert final_h_run_end(metrics_path) == end
    assert all(r["status"] == "committed" for r in training)
    assert [r["step"] for r in training if r["phase"] == "prediction"]

    # FOR-1637: the worker also narrates itself. The substrate's JSON sink is
    # pointed at ``<workspace>.observability/worker-log.jsonl`` (beside
    # metrics.jsonl) once the request is admitted, and the human stderr stream
    # carries the same milestones as one sentence each (this is what the
    # RunPod container log shows). The injected ``stderr`` stays the
    # structured-error channel only: still empty on success.
    assert error.getvalue() == ""
    log_path = tmp_path / "command-worker.observability" / "worker-log.jsonl"
    assert log_path.exists(), "the worker's structured log file must exist beside metrics.jsonl"
    log_rows = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    assert log_rows and all("event" in row for row in log_rows if row["level"] != "DEBUG")
    names = [row["event"] for row in log_rows]
    assert "phase.start" in names and "completion.written" in names
    human = capsys.readouterr().err
    assert "Logging error" not in human, human
    assert "phase.start phase=codec_warmup" in human
    assert "completion.written key=" in human


def test_request_admission_failure_never_touches_disk(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Guard-teeth companion: sink construction must sit AFTER admission,
    never before it -- a request this command refuses cannot leave a
    partial workspace behind (the existing worker/refused-worker contract
    this test mirrors: entropic_substrate.sentence_prediction.worker_command's
    test_local_request_cannot_execute_as_remote_by_omitting_flag)."""
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "refused-worker"
    workspace.mkdir()
    output, error = StringIO(), StringIO()
    # Bind the substrate to THIS test's stderr (the autouse conftest sink writes
    # into a stream capsys does not read; see tests/test_h_worker_log_lines.py).
    configure_logging(json_path=None)

    status = run_worker_command(
        [
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
        ],
        environ={"RUNPOD_POD_ID": "test-owned-pod"},
        objects=objects,
        stdout=output,
        stderr=error,
    )

    assert status == 2
    assert not output.getvalue() and not tuple(workspace.iterdir())
    assert json.loads(error.getvalue())["code"] == "worker_command_refused"
    # S8c: the sibling observability root and the structured sink come AFTER
    # admission too -- a refused request builds neither and narrates no start.
    assert not (tmp_path / "refused-worker.observability").exists()
    assert "worker.start" not in capsys.readouterr().err
    # FOR-1639: a refused request never reaches the worker, so neither
    # lifecycle record exists. Characterization pin (GREEN before the change
    # too: no lifecycle record is written anywhere at the parent).
    assert not (tmp_path / "refused-worker.observability" / "metrics.jsonl").exists()


# ---------------------------------------------------------------- FOR-1639
# run_object_worker writes the h-run-lifecycle-v1 records through the sink it
# is handed. These tests drive the REAL worker over the in-memory store; only
# the failure injections are scripted.

LIFECYCLE = "h-run-lifecycle-v1"
MAX_REQUEST = 1_000_000


class _CheckpointUploadFails(_ObjectStore):
    """Object store whose second checkpoint snapshot upload raises."""

    def publish(
        self, key: str, payload: bytes, *, expected_sha256: str
    ) -> D1TrainerMailboxObjectIdentity:
        if key.endswith("/snapshots/checkpoint-00000002.json"):
            raise RuntimeError("injected checkpoint upload failure")
        return super().publish(key, payload, expected_sha256=expected_sha256)


class _ListSink:
    """A MetricSink double that keeps every record in order."""

    def __init__(self, *, raise_on_event: str | None = None) -> None:
        self.records: list[Mapping[str, Any]] = []
        # DN-1643-PORT: annotate rows are kept apart from `records`, which stays the
        # ordered schema-bearing stream every existing assertion indexes.
        self.annotations: list[dict[str, str]] = []
        self.raise_on_event = raise_on_event

    def log(self, record: Mapping[str, Any]) -> None:
        if self.raise_on_event is not None and record.get("event") == self.raise_on_event:
            raise OSError(f"injected sink failure on {self.raise_on_event}")
        self.records.append(record)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        raise AssertionError("the lifecycle records never use log_artifact")

    def annotate(self, tags: Mapping[str, str]) -> None:
        self.annotations.append(dict(tags))

    def finish(self, *, status: str = "FINISHED") -> None:
        raise AssertionError("run_object_worker does not own the sink's lifecycle")


def _lifecycle(sink: _ListSink) -> list[Mapping[str, Any]]:
    return [r for r in sink.records if r.get("schema") == LIFECYCLE]


def _worker(
    tmp_path: Path, store: _ObjectStore | None = None, *, pause: tuple[str, int] | None = None
) -> tuple[Any, _ObjectStore, D1TrainerMailboxObjectIdentity, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    request, inputs = prepared(tmp_path, pause=pause)
    objects = store if store is not None else _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    root = tmp_path / "object-worker"
    root.mkdir()
    return request, objects, marker, root


def test_ok_run_writes_one_start_and_one_end_around_every_other_row(tmp_path: Path) -> None:
    request, objects, marker, root = _worker(tmp_path)
    sink = _ListSink()

    run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)

    assert [r["event"] for r in _lifecycle(sink)] == ["run_start", "run_end"]
    assert sink.records[0]["event"] == "run_start" and sink.records[-1]["event"] == "run_end"
    assert len(sink.records) > 2, "step rows must sit between the two lifecycle records"
    start, end = sink.records[0], sink.records[-1]
    persisted = json.loads((root / "worker" / "start.json").read_text())
    assert start == {
        "schema": LIFECYCLE,
        "event": "run_start",
        "request_sha256": request.digest,
        "image": request.image,
        "code_commit": request.code_commit,
        "profile_identity": request.profile.identity,
        "run_class": request.profile.run_class,
        "runtime": persisted["runtime"],
    }
    result = json.loads((root / "worker" / "result.json").read_text())
    assert end == {
        "schema": LIFECYCLE,
        "event": "run_end",
        "status": "ok",
        "result_status": "probe_workload_complete",
        "wall_seconds": result["wall_seconds"],
        "successful_optimizer_steps": result["successful_optimizer_steps"],
    }
    assert end["successful_optimizer_steps"] == 6


def test_paused_and_operator_stopped_results_are_ok_run_ends_with_their_own_result_status(
    tmp_path: Path,
) -> None:
    """The run_end status set is exactly {ok, failed}: a pause is a successful
    terminal result, told apart from a complete run by result_status."""
    sink = _ListSink()
    _, objects, marker, root = _worker(tmp_path / "paused", pause=("prediction", 2))
    run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)
    (end,) = [r for r in _lifecycle(sink) if r["event"] == "run_end"]
    assert end["status"] == "ok" and end["result_status"] == "paused"

    stopped = _ListSink()
    request, objects, marker, root = _worker(tmp_path / "stopped")
    objects.seed(stop_marker_key(request), encode_stop_marker(request))
    run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=stopped)
    (end,) = [r for r in _lifecycle(stopped) if r["event"] == "run_end"]
    assert end["status"] == "ok" and end["result_status"] == "stopped_by_operator"
    assert type(end["successful_optimizer_steps"]) is int


def test_training_failure_writes_exactly_one_failed_end_after_its_start(tmp_path: Path) -> None:
    request, objects, marker, root = _worker(tmp_path, _CheckpointUploadFails())
    sink = _ListSink()

    with pytest.raises(RuntimeError, match="checkpoint upload failure"):
        run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)

    assert [r["event"] for r in _lifecycle(sink)] == ["run_start", "run_end"]
    end = sink.records[-1]
    assert end["event"] == "run_end" and end["status"] == "failed"
    assert end["result_status"] == "execution_defect"
    assert end["successful_optimizer_steps"] is None
    assert isinstance(end["wall_seconds"], float) and end["wall_seconds"] > 0
    # custody is unchanged: the R2 failure marker was still published
    assert f"{request.prefix}/failure.json" in objects.values
    assert f"{request.prefix}/complete.json" not in objects.values


def test_final_snapshot_failure_is_one_failed_end_never_an_ok_one(tmp_path: Path) -> None:
    """The run_end must match the terminal marker that was actually published:
    admission and result.json succeed before the final snapshot, but the run
    ends failed because the complete marker never lands."""
    request, inputs = prepared(tmp_path)
    fault_key = f"{request.prefix}/snapshots/complete.json"
    objects = _FlakyReadStore(fail_reads_remaining={fault_key: 3})
    marker = publish_probe_request(objects, request, inputs)
    root = tmp_path / "object-worker"
    root.mkdir()
    sink = _ListSink()

    with pytest.raises(_ReadFailure):
        run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)

    ends = [r for r in _lifecycle(sink) if r["event"] == "run_end"]
    assert len(ends) == 1
    assert ends[0]["status"] == "failed" and ends[0]["result_status"] == "execution_defect"
    assert ends[0]["successful_optimizer_steps"] is None
    assert f"{request.prefix}/complete.json" not in objects.values


def test_failure_before_the_run_started_writes_neither_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pairing: run_end only ever follows a run_start that was logged. A failure
    in materialization (before start()) is still published as a failure, but
    leaves no unpaired run_end in the stream. Characterization pin (GREEN
    before the change: no lifecycle record exists at the parent)."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected materialization failure")

    monkeypatch.setattr(object_worker, "materialize_probe_request", refuse)
    request, objects, marker, root = _worker(tmp_path)
    sink = _ListSink()

    with pytest.raises(RuntimeError, match="materialization failure"):
        run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)

    assert sink.records == []
    assert f"{request.prefix}/failure.json" in objects.values


def test_a_failure_whose_evidence_delivery_also_fails_writes_one_failed_run_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ExceptionGroup exit (a failed failure-snapshot delivery) still closes
    the run with one failed run_end after its run_start, and that run_end's
    wall_seconds is the span of the fake clock (whole ticks, small), not a
    constant or the unstarted 0.0 clock."""

    class BrokenDelivery(_ObjectStore):
        def publish(
            self, key: str, payload: bytes, *, expected_sha256: str
        ) -> D1TrainerMailboxObjectIdentity:
            if key.endswith("/snapshots/checkpoint-00000002.json") or key.endswith(
                "/snapshots/failure.json"
            ):
                raise RuntimeError("injected evidence delivery failure")
            return super().publish(key, payload, expected_sha256=expected_sha256)

    now = {"ticks": 1_000_000.0}

    def fake_perf_counter() -> float:
        now["ticks"] += 10.0
        return now["ticks"]

    monkeypatch.setattr(object_worker, "time", SimpleNamespace(perf_counter=fake_perf_counter))
    _, objects, marker, root = _worker(tmp_path, BrokenDelivery())
    sink = _ListSink()

    with pytest.raises(ExceptionGroup):
        run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)

    assert [r["event"] for r in _lifecycle(sink)] == ["run_start", "run_end"]
    assert sink.records[0]["event"] == "run_start"
    end = sink.records[-1]
    assert end["event"] == "run_end" and end["status"] == "failed"
    assert end["result_status"] == "execution_defect"
    assert end["successful_optimizer_steps"] is None
    wall = end["wall_seconds"]
    assert wall % 10.0 == 0.0 and 0.0 < wall <= 1000.0, wall


def test_the_failed_run_end_clock_is_read_after_failure_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented end point of the failed run_end clock (docs/architecture.md):
    wall_seconds is read AFTER failure custody (snapshot publish, verify-read
    retries, backoff), so the time custody takes counts. The injected delivery
    failure advances the shared fake clock by a large step while custody runs;
    a run_end whose clock was read before custody would not include it."""
    custody_step = 5_000.0
    now = {"ticks": 1_000_000.0}

    class SlowBrokenDelivery(_ObjectStore):
        def publish(
            self, key: str, payload: bytes, *, expected_sha256: str
        ) -> D1TrainerMailboxObjectIdentity:
            if key.endswith("/snapshots/failure.json"):
                now["ticks"] += custody_step
                raise RuntimeError("injected evidence delivery failure")
            if key.endswith("/snapshots/checkpoint-00000002.json"):
                raise RuntimeError("injected checkpoint upload failure")
            return super().publish(key, payload, expected_sha256=expected_sha256)

    def fake_perf_counter() -> float:
        now["ticks"] += 10.0
        return now["ticks"]

    monkeypatch.setattr(object_worker, "time", SimpleNamespace(perf_counter=fake_perf_counter))
    _, objects, marker, root = _worker(tmp_path, SlowBrokenDelivery())
    sink = _ListSink()

    with pytest.raises(ExceptionGroup):
        run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)

    end = sink.records[-1]
    assert end["event"] == "run_end" and end["status"] == "failed"
    assert end["wall_seconds"] >= custody_step, end["wall_seconds"]


def test_the_failed_run_end_clock_origin_is_the_execution_call_not_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented START point of the failed run_end clock (docs/architecture.md):
    the origin is read immediately before the execute_probe_request call, so time
    spent materializing the request does not count. The wrapped materializer
    advances the shared fake clock by a large step; a clock origin read at the top
    of run_object_worker would include it and push wall_seconds past that step."""
    materialize_step = 5_000.0
    now = {"ticks": 1_000_000.0}

    def fake_perf_counter() -> float:
        now["ticks"] += 10.0
        return now["ticks"]

    real_materialize = object_worker.materialize_probe_request

    def slow_materialize(*args: Any, **kwargs: Any) -> Any:
        now["ticks"] += materialize_step
        return real_materialize(*args, **kwargs)

    monkeypatch.setattr(object_worker, "time", SimpleNamespace(perf_counter=fake_perf_counter))
    monkeypatch.setattr(object_worker, "materialize_probe_request", slow_materialize)
    _, objects, marker, root = _worker(tmp_path, _CheckpointUploadFails())
    sink = _ListSink()

    with pytest.raises(RuntimeError):
        run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)

    end = sink.records[-1]
    assert end["event"] == "run_end" and end["status"] == "failed"
    assert end["wall_seconds"] < materialize_step, end["wall_seconds"]


def test_a_failing_run_start_log_writes_no_run_end_and_surfaces_the_sink_error(
    tmp_path: Path,
) -> None:
    """Pairing at its other end: run_end is written only for a run_start that
    start() actually logged. When the run_start log itself fails, the stream
    stays record-free, the failure marker is still published, and the sink's
    own error is the one that surfaces."""
    request, objects, marker, root = _worker(tmp_path)
    sink = _ListSink(raise_on_event="run_start")

    with pytest.raises(OSError, match="injected sink failure on run_start"):
        run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)

    assert sink.records == []
    assert f"{request.prefix}/failure.json" in objects.values
    assert f"{request.prefix}/complete.json" not in objects.values


def test_a_sink_failure_on_the_failed_end_surfaces_beside_the_original_error(
    tmp_path: Path,
) -> None:
    request, objects, marker, root = _worker(tmp_path, _CheckpointUploadFails())
    sink = _ListSink(raise_on_event="run_end")

    with pytest.raises(ExceptionGroup) as raised:
        run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)

    kinds = sorted(type(e).__name__ for e in raised.value.exceptions)
    assert kinds == ["OSError", "RuntimeError"], kinds
    assert raised.value.__cause__ is not None
    # R2 custody came first and is complete despite the sink defect
    assert f"{request.prefix}/failure.json" in objects.values
    assert f"{request.prefix}/snapshots/failure.json" in objects.values


def test_a_sink_failure_on_the_ok_end_propagates_without_a_failure_marker(
    tmp_path: Path,
) -> None:
    """The complete marker is already published when the ok run_end is logged;
    a sink defect there must not publish a contradictory failure marker."""
    request, objects, marker, root = _worker(tmp_path)
    sink = _ListSink(raise_on_event="run_end")

    with pytest.raises(OSError, match="injected sink failure on run_end"):
        run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)

    assert f"{request.prefix}/complete.json" in objects.values
    assert f"{request.prefix}/failure.json" not in objects.values
    assert not (root / "worker" / "failure.json").exists()


def test_the_sink_never_changes_the_r2_objects_the_worker_publishes(tmp_path: Path) -> None:
    """Characterization pin (GREEN before the change): with or without a sink the
    worker publishes the same keys in the same order."""
    request, plain, marker, root = _worker(tmp_path / "plain")
    run_object_worker(plain, marker, root, max_request_bytes=MAX_REQUEST)
    _, mirrored, marker, root = _worker(tmp_path / "mirrored")
    run_object_worker(mirrored, marker, root, max_request_bytes=MAX_REQUEST, sink=_ListSink())

    # Journal object names embed a per-run content hash (wall-clock fields), so
    # their order inside one snapshot is not stable run to run. Compare the
    # key set by shape, and the terminal marker (always published last) exactly.
    def shape(keys: list[str]) -> list[str]:
        return sorted(re.sub(r"/[0-9a-f]{32}-", "/<run>-", key) for key in keys)

    assert shape(mirrored.publishes) == shape(plain.publishes)
    assert mirrored.publishes[-1] == plain.publishes[-1] == f"{request.prefix}/complete.json"


def test_the_worker_file_is_admissible_to_the_h_capture_predicate(tmp_path: Path) -> None:
    """Emitter-to-reader agreement, offline: the real worker's metrics.jsonl goes
    through final_h_run_end. The reader holds its own schema literals, so only
    this test notices if the two drift apart."""
    _, objects, marker, root = _worker(tmp_path)
    metrics = tmp_path / "observability" / "metrics.jsonl"

    run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=JsonlSink(metrics))

    end = final_h_run_end(metrics)
    assert end["status"] == "ok" and end["result_status"] == "probe_workload_complete"


def test_a_failed_runs_file_stays_admissible_and_says_failed(tmp_path: Path) -> None:
    _, objects, marker, root = _worker(tmp_path, _CheckpointUploadFails())
    metrics = tmp_path / "observability" / "metrics.jsonl"

    with pytest.raises(RuntimeError):
        run_object_worker(
            objects, marker, root, max_request_bytes=MAX_REQUEST, sink=JsonlSink(metrics)
        )

    end = final_h_run_end(metrics)
    assert end["status"] == "failed" and end["successful_optimizer_steps"] is None


# ------------------------------------------------------------------ FOR-1642 / FOR-1643
# Checkpoint evidence per sealed checkpoint (DN-1643-PORT) and the worker's terminal
# status by outcome. The evidence is the SNAPSHOT MARKER identity that
# publish_worker_snapshot returns (the identity materialize_worker_snapshot accepts),
# annotated only after the checkpoint event is published, never for the final snapshot.

_EVIDENCE_FIELDS = ("key", "sha256", "size_bytes", "publication_seconds")


class _OrderedSink(_ListSink):
    """Remembers which objects were already published when each annotate arrived."""

    def __init__(self, objects: _ObjectStore) -> None:
        super().__init__()
        self._objects = objects
        self.published_at_annotate: list[list[str]] = []

    def annotate(self, tags: Mapping[str, str]) -> None:
        self.published_at_annotate.append(list(self._objects.publishes))
        super().annotate(tags)


def _checkpoint_events(objects: _ObjectStore, prefix: str) -> dict[int, dict[str, Any]]:
    events: dict[int, dict[str, Any]] = {}
    for key, payload in objects.values.items():
        match = re.fullmatch(re.escape(prefix) + r"/checkpoint-(\d{8})\.json", key)
        if match:
            events[int(match.group(1))] = json.loads(payload)
    return events


def test_every_sealed_checkpoint_is_evidenced_on_the_run_by_its_snapshot_marker(
    tmp_path: Path,
) -> None:
    """M4 (worker half): one annotate per sealed checkpoint, the four tags equal to the
    snapshot marker the checkpoint event records, publication_seconds the event's own
    measured l3_seconds (equality under float(), so no new clock), and the annotation
    arrives only after the checkpoint event was published."""
    request, objects, marker, root = _worker(tmp_path)
    sink = _OrderedSink(objects)

    run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)

    events = _checkpoint_events(objects, request.prefix)
    assert events, "the fast profile must seal at least one checkpoint"
    assert len(sink.annotations) == len(events)  # none for `started` or the final snapshot
    for annotation, (total, event), published in zip(
        sink.annotations, sorted(events.items()), sink.published_at_annotate, strict=True
    ):
        base = f"checkpoint/{total:08d}"
        assert set(annotation) == {f"{base}/{field}" for field in _EVIDENCE_FIELDS}
        assert all(type(v) is str for v in annotation.values())
        snapshot = event["snapshot"]
        assert snapshot["key"] == f"{request.prefix}/snapshots/checkpoint-{total:08d}.json"
        assert annotation[f"{base}/key"] == snapshot["key"]
        assert annotation[f"{base}/sha256"] == snapshot["sha256"]
        assert annotation[f"{base}/size_bytes"] == str(snapshot["size_bytes"])
        assert (
            float(annotation[f"{base}/publication_seconds"]) == event["publication"]["l3_seconds"]
        )
        assert f"{request.prefix}/checkpoint-{total:08d}.json" in published


def test_checkpoint_evidence_lands_in_the_authoritative_jsonl_and_leaves_it_admissible(
    tmp_path: Path,
) -> None:
    _, objects, marker, root = _worker(tmp_path)
    metrics = tmp_path / "observability" / "metrics.jsonl"

    run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=JsonlSink(metrics))

    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line]
    annotations = [r for r in rows if r.get("event") == "annotate"]
    assert annotations and all("schema" not in r for r in annotations)
    assert any(k.endswith("/sha256") for k in annotations[0])
    assert final_h_run_end(metrics)["status"] == "ok"


class _RecordingSink:
    """Wraps the REAL sink (so the run is exercised end to end) and records the lifecycle."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.finishes: list[str] = []
        self.annotations: list[dict[str, str]] = []

    def log(self, record: Mapping[str, Any]) -> None:
        self.inner.log(record)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        self.inner.log_artifact(path, artifact_path=artifact_path)

    def annotate(self, tags: Mapping[str, str]) -> None:
        self.annotations.append(dict(tags))
        self.inner.annotate(tags)

    def finish(self, *, status: str = "FINISHED") -> None:
        self.finishes.append(status)
        self.inner.finish(status=status)


def _command(
    tmp_path: Path, store: _ObjectStore, monkeypatch: pytest.MonkeyPatch, name: str
) -> tuple[list[_RecordingSink], int, StringIO, Path]:
    from entropic_substrate.ladder.run import worker_command

    request, inputs = prepared(tmp_path)
    marker = publish_probe_request(store, request, inputs)
    workspace = tmp_path / name
    workspace.mkdir()
    built: list[_RecordingSink] = []
    real = worker_command.build_sink_for_run

    def spy(*args: Any, **kwargs: Any) -> _RecordingSink:
        built.append(_RecordingSink(real(*args, **kwargs)))
        return built[-1]

    monkeypatch.setattr(worker_command, "build_sink_for_run", spy)
    error = StringIO()
    status = run_worker_command(
        [
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
        ],
        environ={},
        objects=store,
        stdout=StringIO(),
        stderr=error,
    )
    return built, status, error, tmp_path / f"{name}.observability" / "metrics.jsonl"


def test_a_returned_worker_finishes_its_sink_finished_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built, status, error, metrics = _command(tmp_path, _ObjectStore(), monkeypatch, "ok-worker")
    assert status == 0, error.getvalue()
    (sink,) = built
    assert sink.finishes == ["FINISHED"]
    assert sink.annotations, "the command threads the sink the evidence is annotated on"
    assert final_h_run_end(metrics)["status"] == "ok"


def test_a_raising_worker_finishes_its_sink_failed_and_keeps_its_exit_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M5: the run terminates FAILED on the exception path, the original exception still
    reaches the outer handler unchanged (stage worker_execution, exit 2), and the run_end
    in the authoritative JSONL says failed."""
    built, status, error, metrics = _command(
        tmp_path, _CheckpointUploadFails(), monkeypatch, "failing-worker"
    )
    assert status == 2
    report = json.loads(error.getvalue())
    assert report["stage"] == "worker_execution" and report["error_type"] == "RuntimeError"
    (sink,) = built
    assert sink.finishes == ["FAILED"]
    assert final_h_run_end(metrics)["status"] == "failed"


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(4)])
def test_an_interrupted_worker_finishes_its_sink_failed_and_the_interrupt_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    """G2: an `except Exception: failed = True` shape would terminate an interrupted run
    FINISHED. Every raised exit, BaseException included, is FAILED."""
    from entropic_substrate.ladder.run import worker_command

    def interrupt(*_a: object, **_k: object) -> object:
        raise exc

    monkeypatch.setattr(worker_command, "run_object_worker", interrupt)
    store = _ObjectStore()
    built: list[_RecordingSink] = []
    real = worker_command.build_sink_for_run

    def spy(*args: Any, **kwargs: Any) -> _RecordingSink:
        built.append(_RecordingSink(real(*args, **kwargs)))
        return built[-1]

    monkeypatch.setattr(worker_command, "build_sink_for_run", spy)
    request, inputs = prepared(tmp_path)
    marker = publish_probe_request(store, request, inputs)
    workspace = tmp_path / "interrupted-worker"
    workspace.mkdir()
    with pytest.raises(type(exc)):
        run_worker_command(
            [
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
            ],
            environ={},
            objects=store,
            stdout=StringIO(),
            stderr=StringIO(),
        )
    (sink,) = built
    assert sink.finishes == ["FAILED"]


def test_a_refused_request_builds_no_run_and_finishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K13: the admission-failure path builds no sink, so nothing is created to finish."""
    from entropic_substrate.ladder.run import worker_command

    built: list[object] = []
    monkeypatch.setattr(worker_command, "build_sink_for_run", lambda *a, **k: built.append((a, k)))
    request, inputs = prepared(tmp_path)
    store = _ObjectStore()
    marker = publish_probe_request(store, request, inputs)
    workspace = tmp_path / "refused-worker-2"
    workspace.mkdir()
    status = run_worker_command(
        [
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
        ],
        environ={"RUNPOD_POD_ID": "test-owned-pod"},
        objects=store,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    assert status == 2 and built == []


def test_checkpoint_evidence_reaches_the_mlflow_run_tags_through_the_real_adapters(
    tmp_path: Path,
) -> None:
    """M4: the worker's annotate lands as run tags on a (fake-client) MLflow run, and the
    composite carries the same evidence into the authoritative JSONL."""
    from tests.test_sinks import FakeMlflowClient

    from entropic_substrate.ladder.capture.adapters.composite import CompositeSink
    from entropic_substrate.ladder.capture.adapters.mlflow import MlflowSink

    request, objects, marker, root = _worker(tmp_path)
    client = FakeMlflowClient()
    mlflow = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    metrics = tmp_path / "observability" / "metrics.jsonl"
    sink = CompositeSink(JsonlSink(metrics), mlflow)

    run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST, sink=sink)
    sink.finish()

    tags = client._runs[mlflow.run_id].data.tags
    events = _checkpoint_events(objects, request.prefix)
    assert events
    for total, event in events.items():
        base = f"checkpoint/{total:08d}"
        assert tags[f"{base}/key"] == event["snapshot"]["key"]
        assert tags[f"{base}/sha256"] == event["snapshot"]["sha256"]
        assert float(tags[f"{base}/publication_seconds"]) == event["publication"]["l3_seconds"]
    assert mlflow.failed_writes == 0
