"""LIVE integration test for the H lane's MetricSink routing (RUNG 2c,
ruling D-D) -- a REAL MLflow server, never a mock.

Modeled on ``tests/test_mlflow_live.py``: uses the ALREADY-RUNNING stack via
``MLFLOW_TRACKING_URI`` and never brings up its own compose project. Skipped
outright when that variable is unset, so ``just test-all`` (no live stack)
never tries to reach a server that isn't there. Because it is
``integration``-marked and not on ``scripts/run_container_test.sh``'s
``--ignore`` list, phase 1 of the container gate (the same pytest process
that already sweeps ``test_mlflow_live.py``) picks it up automatically --
no edit to that script (outside this phase's envelope) is needed.

Drives the smallest CPU-feasible H path: the same fast profile
``tests/test_sentence_execution.py``'s ``engine()`` fixture builds (2
``codec_warmup`` + 4 ``prediction`` steps, the schedule
``tests/test_sentence_runner.py`` already pins) through the landed
``CompositeSink`` under a ``TrackingRunDescriptor``, and asserts the
resulting server-side run actually carries an H metric -- proving
``HTrainingStepExtractor`` (RUNG 2b), not just that a run object exists.

The run id is written to ``results/h-lane-live-run-id.txt`` (gitignored,
same convention as every other ``results/*`` artifact -- CLAUDE.md's Data
flow section) in ADDITION to being printed. MEASURED: the container gate
(``scripts/run_container_test.sh``, outside this phase's envelope) invokes
phase 1 as ``pytest -m integration -q --disable-warnings``, and
``pyproject.toml``'s ``addopts`` carries no ``-r`` report-chars flag, so
pytest's default report NEVER surfaces a passing test's captured stdout --
a bare ``print()`` is invisible in that gate's own output on the (expected)
happy path. The file write is the only way this test's run id actually
reaches whoever writes the PR body after a green `just container-test`,
run byte-for-byte as configured.
"""

from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path
from uuid import uuid4

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_execution import engine
from tests.test_sentence_worker_execution import prepared

from entropic_substrate.ladder.capture.adapters.composite import build_sink_for_run
from entropic_substrate.ladder.capture.adapters.mlflow import build_mlflow_client
from entropic_substrate.ladder.capture.ports.run_descriptor import TrackingRunDescriptor
from entropic_substrate.ladder.capture.reader.capture_results import capture_finished_runs
from entropic_substrate.ladder.capture.reader.h_telemetry import final_h_run_end
from entropic_substrate.ladder.run.h_metric_extractor import HTrainingStepExtractor
from entropic_substrate.ladder.run.runner import run_probe_training
from entropic_substrate.ladder.run.worker_command import run_worker_command
from entropic_substrate.ladder.run.worker_transport import publish_probe_request

MLFLOW_TRACKING_ENV = "MLFLOW_TRACKING_URI"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID_FILE = REPO_ROOT / "results" / "h-lane-live-run-id.txt"
CAPTURE_EVIDENCE_FILE = REPO_ROOT / "results" / "h-capture-live-run-id.txt"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get(MLFLOW_TRACKING_ENV, "").strip(),
        reason=f"${MLFLOW_TRACKING_ENV} not set -- run via `just container-test`",
    ),
]


def _tracking_uri() -> str:
    return os.environ[MLFLOW_TRACKING_ENV]


def test_live_h_training_steps_reach_a_real_mlflow_run(tmp_path: Path) -> None:
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    descriptor = TrackingRunDescriptor(
        run_name=f"h-lane-live-{template.profile.identity}",
        tiny=True,
        params={"run_class": template.profile.run_class},
        tags={"schema": "h-training-step-v2", "gate": "h-lane-live"},
    )
    sink = build_sink_for_run(
        descriptor,
        results_path=tmp_path / "metrics.jsonl",
        metric_extractor=HTrainingStepExtractor(
            warmup_steps=template.profile.schedule.warmup_steps
        ),
    )
    try:
        result = run_probe_training(
            template.profile,
            template.source_root,
            output,
            max_source_bytes=1_000_000,
            max_artifact_bytes=10_000_000,
            sink=sink,
        )
    finally:
        sink.finish()

    assert result.status == "training_complete"
    mlflow_sink = sink.best_effort[0]  # type: ignore[attr-defined]

    import mlflow.tracking

    run = mlflow.tracking.MlflowClient(tracking_uri=_tracking_uri()).get_run(
        mlflow_sink.run_id  # type: ignore[attr-defined]
    )
    metric_keys = sorted(run.data.metrics)
    print(f"live H run id: {mlflow_sink.run_id}")
    print(f"live H metric keys ({len(metric_keys)} total): {metric_keys}")
    RUN_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUN_ID_FILE.write_text(f"{mlflow_sink.run_id}\n")
    assert {"objective", "aggregate_norm", "decoder_ce"} & set(metric_keys), metric_keys


def test_live_h_worker_run_is_captured_by_the_capture_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FOR-1639: the PRODUCTION call path, captured back off a real server.

    ``run_worker_command --local-tiny`` builds the real sink with the
    production ``h-probe-worker-result-v1`` tag (a test-invented tag would
    prove nothing about the partition). ``build_sink_for_run`` reads the
    experiment from ``os.environ``, not from the ``environ=`` handed to the
    command, so the per-test experiment name is set on the process BEFORE the
    command runs and capture is pointed at that experiment ONLY -- never the
    default one, which holds other live tests' runs and any legacy H runs.
    """
    experiment = f"h-capture-live-{uuid4().hex}"
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", experiment)
    monkeypatch.setenv("ENTROPIC_MLFLOW_TINY", "1")
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "command-worker"
    workspace.mkdir()
    out, err = StringIO(), StringIO()

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
        stdout=out,
        stderr=err,
    )
    assert status == 0, err.getvalue()

    log = workspace.parent / f"{workspace.name}.observability" / "worker-log.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines() if line]
    (mlflow_row,) = [row for row in rows if row.get("event") == "mlflow.run"]
    run_id: str = mlflow_row["run_id"]

    client = build_mlflow_client(_tracking_uri())
    captured = capture_finished_runs(client, [experiment], tmp_path / "captured")

    names = sorted(path.name for path in captured)
    CAPTURE_EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAPTURE_EVIDENCE_FILE.write_text(f"{experiment}\n{run_id}\n{names}\n")
    assert len(captured) == 2, names
    assert all(f"-{run_id[:8]}-" in name for name in names), names
    assert any(name.endswith("-metrics.jsonl") for name in names), names
    assert any(name.endswith("-worker-log.jsonl") for name in names), names
    (metrics,) = [path for path in captured if path.name.endswith("-metrics.jsonl")]
    end = final_h_run_end(metrics)
    assert end["status"] == "ok" and end["result_status"] == "probe_workload_complete"
    # a second pass over the same destination is a clean no-op, not a re-capture
    assert capture_finished_runs(client, [experiment], tmp_path / "captured") == []
