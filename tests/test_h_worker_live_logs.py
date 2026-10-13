"""LIVE integration test for the H worker's log capture (FOR-1637) -- a REAL
MLflow server and the REAL R2 bucket, never a mock.

Modeled on ``tests/test_h_lane_live.py``: uses the ALREADY-RUNNING stack via
``MLFLOW_TRACKING_URI``, skipped outright when it is unset (so ``just
test-all`` without a stack never reaches for a server), and picked up by
``scripts/run_container_test.sh`` phase 1 (``pytest -m integration``) with no
edit to that script. Two legs:

1. The fast H path (the 2 ``codec_warmup`` + 4 ``prediction`` step profile)
   through a real MLflow run built by ``build_sink_for_run(..., log_path=...)``.
   After ``finish()`` the server must list ``results/worker-log.jsonl`` beside
   ``results/metrics.jsonl``, and the downloaded records must carry the
   narrated milestones including the AUTHORITATIVE ``JsonlSink``'s
   ``sink.finish.end`` (the MLflow sink's own ``sink.finish.end`` is logged
   after its upload and is, by construction, not in the file it uploaded).
2. One snapshot-sized object published through the REAL R2 adapter under a
   dedicated per-run prefix ``tests/for-1637/<uuid>/`` (never ``h-probes/``,
   ``mlflow/``, ``mlflow-live-gate/`` or the DVC cache layout), observed as
   ``publish.start`` / ``publish.verify_read`` / ``publish.end``, and deleted in
   teardown by a test-owned raw boto3 client with the post-teardown listing
   asserted EMPTY. Missing R2 credentials FAIL LOUD here (this runs inside the
   gate, whose preflight already requires them); they never skip.

The MLflow run id and the R2 listing evidence are written under ``results/``
(gitignored): the container gate's default pytest report never surfaces a
passing test's stdout, so a file is the only way they reach the PR body.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from tests.test_sentence_execution import engine

from entropic_substrate.kernel.runtime.logging import configure_logging
from entropic_substrate.ladder.capture.adapters.composite import build_sink_for_run
from entropic_substrate.ladder.capture.ports.run_descriptor import TrackingRunDescriptor
from entropic_substrate.ladder.run.h_metric_extractor import HTrainingStepExtractor
from entropic_substrate.ladder.run.runner import run_probe_training
from entropic_substrate.ladder.run.transport.d1_dvc_remote_config import load_d1_dvc_s3_remote
from entropic_substrate.ladder.run.worker_command import build_worker_objects
from entropic_substrate.ladder.run.worker_transport import (
    VerificationReadRetry,
    publish_verified_object,
)

MLFLOW_TRACKING_ENV = "MLFLOW_TRACKING_URI"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID_FILE = REPO_ROOT / "results" / "h-worker-live-logs-run-id.txt"
R2_EVIDENCE_FILE = REPO_ROOT / "results" / "h-worker-live-logs-r2.txt"
R2_TEST_PREFIX_ROOT = "tests/for-1637"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get(MLFLOW_TRACKING_ENV, "").strip(),
        reason=f"${MLFLOW_TRACKING_ENV} not set -- run via `just container-test`",
    ),
]


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_live_worker_log_is_uploaded_as_a_run_artifact(tmp_path: Path) -> None:
    observability = tmp_path / "live.observability"
    log_path = observability / "worker-log.jsonl"
    configure_logging(json_path=log_path)
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    descriptor = TrackingRunDescriptor(
        run_name=f"h-worker-live-logs-{template.profile.identity}",
        tiny=True,
        params={"run_class": template.profile.run_class},
        tags={"schema": "h-worker-log-v1", "gate": "h-worker-live-logs"},
    )
    sink = build_sink_for_run(
        descriptor,
        results_path=observability / "metrics.jsonl",
        log_path=log_path,
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
    run_id: str = mlflow_sink.run_id

    import mlflow.tracking

    client = mlflow.tracking.MlflowClient(tracking_uri=os.environ[MLFLOW_TRACKING_ENV])
    listed = sorted(item.path for item in client.list_artifacts(run_id, "results"))
    print(f"live worker-log run id: {run_id}; results artifacts: {listed}")
    RUN_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUN_ID_FILE.write_text(f"{run_id}\n{listed}\n")
    assert "results/worker-log.jsonl" in listed, listed
    assert "results/metrics.jsonl" in listed, listed  # the existing upload is undisturbed

    downloaded = Path(
        client.download_artifacts(run_id, "results/worker-log.jsonl", str(tmp_path / "dl"))
    )
    uploaded = _rows(downloaded)
    events = [row["event"] for row in uploaded if "event" in row]
    assert "mlflow.run" in events
    assert "phase.start" in events and "checkpoint.seal" in events
    assert "evaluation.end" in events
    # The authoritative JsonlSink finishes first, so its end record is already
    # in the file the MLflow sink uploads.
    assert any(
        row.get("event") == "sink.finish.end" and row.get("sink") == "jsonl" for row in uploaded
    ), events
    # The artifact is a faithful prefix of the local file (later records, e.g.
    # the MLflow sink's own finish.end, are logged after the upload read it).
    local = _rows(log_path)
    assert uploaded == local[: len(uploaded)]


@pytest.fixture
def r2_prefix() -> Iterator[str]:
    """A dedicated per-run prefix; teardown deletes it with a RAW boto3 client."""
    access, private = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access or not private:
        pytest.fail(
            "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY are required inside the live gate "
            "(scripts/run_container_test.sh preflights them); refusing to skip the R2 leg"
        )
    import boto3

    topology = load_d1_dvc_s3_remote(REPO_ROOT)
    raw = boto3.client(
        "s3",
        endpoint_url=topology.endpoint_url,
        region_name="auto",
        aws_access_key_id=access,
        aws_secret_access_key=private,
    )
    prefix = f"{R2_TEST_PREFIX_ROOT}/{uuid4().hex}"
    assert prefix.startswith("tests/for-1637/") and "h-probes" not in prefix
    try:
        yield prefix
    finally:
        listed = raw.list_objects_v2(Bucket=topology.bucket, Prefix=f"{prefix}/")
        for item in listed.get("Contents", []):
            raw.delete_object(Bucket=topology.bucket, Key=item["Key"])
        after = raw.list_objects_v2(Bucket=topology.bucket, Prefix=f"{prefix}/")
        remaining = [item["Key"] for item in after.get("Contents", [])]
        evidence = (
            f"prefix: {prefix}\n"
            f"deleted: {[item['Key'] for item in listed.get('Contents', [])]}\n"
            f"listing after teardown: {remaining}\n"
        )
        R2_EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        R2_EVIDENCE_FILE.write_text(evidence)
        print(evidence)
        assert remaining == [], f"R2 test prefix {prefix} was not emptied: {remaining}"


def test_live_r2_publish_is_narrated_and_leaves_nothing_behind(
    tmp_path: Path, r2_prefix: str
) -> None:
    log_path = tmp_path / "r2.observability" / "worker-log.jsonl"
    configure_logging(json_path=log_path)
    objects = build_worker_objects(
        REPO_ROOT, os.environ, max_object_bytes=1_000_000, max_objects=10
    )
    key = f"{r2_prefix}/snapshot.json"
    payload = json.dumps({"schema": "for-1637-live-probe", "prefix": r2_prefix}).encode()
    identity = publish_verified_object(objects, key, payload, retry=VerificationReadRetry())
    assert identity.key == key and identity.size_bytes == len(payload)

    rows = [row for row in _rows(log_path) if row.get("key") == key]
    names = [row["event"] for row in rows]
    assert names == ["publish.start", "publish.verify_read", "publish.end"], names
    assert rows[0]["bytes"] == len(payload) == rows[2]["bytes"]
