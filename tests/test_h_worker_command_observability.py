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
from io import StringIO
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_execution import prepared

from entropic_substrate.kernel.runtime.logging import configure_logging
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
    training = [r for r in records if r["schema"] == "h-training-step-v2"]
    evaluations = [r for r in records if r["schema"] == "h-predictive-evaluation-v1"]
    assert training, "no committed H training-step rows reached the sink"
    assert evaluations, "no sealed evaluation reports reached the sink"
    assert len(training) + len(evaluations) == len(records)
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
