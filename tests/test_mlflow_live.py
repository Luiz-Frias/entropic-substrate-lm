"""LIVE integration tests — a REAL MLflow server (podman-compose: Postgres +
mlflow, artifacts proxied to REAL Cloudflare R2), never a mock.

Gated behind ``just container-test`` (``scripts/run_container_test.sh``
brings the stack up, sets ``MLFLOW_TRACKING_URI``/``ENTROPIC_MLFLOW_TINY``,
runs this file, tears the stack down). Every test here is skipped outright
if ``MLFLOW_TRACKING_URI`` is not set, so ``just test-all`` (no live stack)
never tries to reach a server that isn't there.

R2, never MinIO: the CRC32-checksum incompatibility CLAUDE.md warns about is
invisible against MinIO — these tests exercise the actual bucket
(``s3://entropic-substrate-lm/mlflow-live-gate/``, the gate's own venue-split
prefix — see compose.yaml's --artifacts-destination comment) end to end.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from entropic_substrate.mlflow_aggregate import aggregate
from entropic_substrate.sinks import MLFLOW_TRACKING_ENV, build_sink
from entropic_substrate.train import Trainer
from helpers import tiny_config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get(MLFLOW_TRACKING_ENV, "").strip(),
        reason=f"${MLFLOW_TRACKING_ENV} not set — run via `just container-test`",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[1]
# scripts/run_container_test.sh derives the compose project name
# PER-WORKTREE (not a fixed, shared name — see that script's own comment for
# why: a fixed name is what fused one agent's mlflow container to a
# DIFFERENT agent's postgres under concurrent runs) and exports it as
# MLFLOW_COMPOSE_PROJECT precisely so this module's own `podman compose -p
# ...` calls (the mid-run mlflow-kill test below) target the SAME project
# the script brought up. A hardcoded fallback here would reintroduce the
# exact cross-talk defect this coupling exists to prevent — read the env var
# only, and fail loud (inside `_stop_mlflow`, at the point it's actually
# needed) rather than silently defaulting to the old shared name.
_PROJECT = os.environ.get("MLFLOW_COMPOSE_PROJECT", "")
_COMPOSE = ["podman", "compose", "-p", _PROJECT, "-f", str(REPO_ROOT / "compose.yaml")]


def _tracking_uri() -> str:
    return os.environ[MLFLOW_TRACKING_ENV]


def _client() -> object:
    import mlflow.tracking

    return mlflow.tracking.MlflowClient(tracking_uri=_tracking_uri())


def _fast_e3_full_shape_config(tmp_path: Path) -> object:
    """9 layers x 12 heads (108 kappa series) — full geometry shape, tiny
    everything else, so this runs in seconds on CPU while still producing
    the exact shape acceptance #7 asks for.
    """
    cfg = tiny_config(
        tmp_path,
        d_model=24,  # 24 % 12 == 0; kappa is one scalar per head, independent of d_model
        n_layers=9,
        n_heads=12,
        block_size=16,
        block="ngpt",
        norm="ngpt_norm",
        attn_kernel="kappa",
        kappa_init=[-1.0, 0.0, 1.0],
    )
    cfg.name = "live-e3-full-shape"
    cfg.data.batch_size = 2
    cfg.data.seq_len = 16
    cfg.data.synthetic_bytes = 4_000
    cfg.data.num_eval_batches = 1
    cfg.train.max_steps = 2
    cfg.train.eval_interval = 1
    cfg.train.log_interval = 1
    cfg.train.save_interval = 0
    return cfg


# --------------------------------------------------------------- acceptance #7
def test_live_kappa_metrics_are_108_shaped(tmp_path: Path) -> None:
    cfg = _fast_e3_full_shape_config(tmp_path)
    sink = build_sink(cfg, tiny=True, results_path=tmp_path / "out.jsonl")
    trainer = Trainer(cfg, results_path=tmp_path / "out.jsonl", sink=sink)
    trainer.fit(verbose=False)

    mlflow_sink = sink.best_effort[0]  # type: ignore[attr-defined]
    run = _client().get_run(mlflow_sink.run_id)  # type: ignore[attr-defined]
    metric_keys = sorted(run.data.metrics)

    head_keys = [k for k in metric_keys if k.startswith("kappa/L") and "/h" in k]
    assert len(head_keys) == 9 * 12 == 108, metric_keys
    assert "kappa/spread_max" in metric_keys
    assert "kappa/mean_abs" in metric_keys
    print(f"live metric keys ({len(metric_keys)} total): {metric_keys}")


# --------------------------------------------------------------- acceptance #4
def test_live_artifact_round_trips_through_real_r2(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)
    cfg.name = "live-r2-roundtrip"
    sink = build_sink(cfg, tiny=True, results_path=tmp_path / "out.jsonl")
    trainer = Trainer(cfg, results_path=tmp_path / "out.jsonl", sink=sink)
    trainer.fit(verbose=False)  # produces a "final" checkpoint, logged as an artifact

    mlflow_sink = sink.best_effort[0]  # type: ignore[attr-defined]
    client = _client()
    artifacts = client.list_artifacts(mlflow_sink.run_id, "checkpoints")  # type: ignore[union-attr]
    assert artifacts, "no artifacts logged for the final checkpoint"
    remote_path = artifacts[0].path

    download_dir = client.download_artifacts(mlflow_sink.run_id, remote_path, str(tmp_path))  # type: ignore[union-attr]
    downloaded = Path(download_dir)
    assert downloaded.exists()
    assert downloaded.stat().st_size > 0

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL",
            "https://62c7fd87a1874f1dee6b149ad40ac657.r2.cloudflarestorage.com",
        ),
        region_name="auto",
    )
    # NOTE: the real key layout is "<dest>/{experiment_id}/{run_id}/artifacts/..."
    # — verified empirically (a prefix of "<dest>/{run_id}/" alone matches
    # nothing; the experiment_id segment is load-bearing). Search broadly by
    # run_id substring instead of assuming that internal layout. The prefix is
    # the GATE'S OWN venue-split destination (compose.yaml) — listing the
    # deployed server's "mlflow/" here would re-create the commingled read the
    # split exists to end, and (MEASURED, first post-split gate run 2026-08-17)
    # finds nothing: this very assertion failed with keys=[] until the probe
    # followed the destination it verifies.
    run_id = mlflow_sink.run_id  # type: ignore[attr-defined]
    resp = s3.list_objects_v2(Bucket="entropic-substrate-lm", Prefix="mlflow-live-gate/")
    keys = [o["Key"] for o in resp.get("Contents", []) if run_id in o["Key"]]
    print(f"observed R2 object key(s) for run {run_id}: {keys}")
    assert any(k.endswith(remote_path) for k in keys), (run_id, remote_path, keys)


# --------------------------------------------------------------- acceptance #8/9
def test_live_aggregate_is_idempotent_and_absent_when_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client()
    experiment_name = f"live-aggregate-{int(time.time())}"
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", experiment_name)

    def _seed_run(seed: int, npb: float) -> None:
        cfg = tiny_config(tmp_path)
        cfg.name = "live-agg-exp"
        cfg.train.seed = seed
        sink = build_sink(cfg, tiny=True, results_path=tmp_path / f"seed-{seed}.jsonl")
        mlflow_sink = sink.best_effort[0]  # type: ignore[attr-defined]
        client.log_metric(mlflow_sink.run_id, "nats_per_byte", npb)  # type: ignore[union-attr]
        client.log_metric(mlflow_sink.run_id, "loss", npb * 2)  # type: ignore[union-attr]
        sink.finish()  # type: ignore[attr-defined]

    _seed_run(1, 1.5)
    _seed_run(2, 1.6)

    incomplete = aggregate(client, experiment_name=experiment_name, n_seeds=3)
    assert incomplete == [], "no parent without a complete aggregate — 2/3 seeds present"

    _seed_run(3, 1.4)
    first = aggregate(client, experiment_name=experiment_name, n_seeds=3)
    second = aggregate(client, experiment_name=experiment_name, n_seeds=3)

    assert len(first) == 1
    assert first[0]["parent_run_id"] == second[0]["parent_run_id"]
    assert first[0]["n_seeds"] == second[0]["n_seeds"] == 3
    assert first[0]["nats_per_byte"] == second[0]["nats_per_byte"]
    print(f"idempotent aggregate: {first[0]}")


# --------------------------------------------------------------- acceptance #6
def _kill_mid_run_config(root: Path, *, name: str) -> object:
    cfg = tiny_config(root)
    cfg.name = name
    cfg.train.max_steps = 30
    cfg.train.eval_interval = 2
    cfg.train.log_interval = 1
    cfg.train.seed = 99
    return cfg


def _stop_mlflow() -> None:
    if not _PROJECT:
        # Fail loud, never silently default to the old shared project name —
        # that default is exactly the concurrency defect this coupling was
        # written to prevent (see the module-level comment on _PROJECT).
        msg = (
            "MLFLOW_COMPOSE_PROJECT is not set — this test must run via "
            "`just container-test` (scripts/run_container_test.sh derives and "
            "exports the per-run compose project name before pytest starts)."
        )
        raise RuntimeError(msg)
    subprocess.run([*_COMPOSE, "stop", "mlflow"], check=True, cwd=REPO_ROOT, timeout=30)  # nosec B603 B607


def test_kill_mid_run_jsonl_stays_complete_same_record_count_as_untracked(
    tmp_path: Path,
) -> None:
    """SACRED INVARIANT: JSONL IS AUTHORITATIVE, MLFLOW IS DERIVED.

    A single continuous ``fit()`` call, same seed/config on both sides; a
    background timer kills the real mlflow container partway through the
    tracked run's training loop (several evals happen both before and after
    the kill, per ``eval_interval=2`` over ``max_steps=30``).

    ORDERED LAST in this module (pytest collects top-to-bottom; this suite
    runs without xdist) — deliberately. This test used to restart mlflow in
    its own ``finally`` block and block on up to 60s of health-status
    polling before returning control to whatever test ran next; under
    full-suite load that poll blew its deadline on 2 of 4 verbatim re-runs
    (an isolated stop/start of the same container recovered in ~11s), and
    the resulting ``RuntimeError`` cascaded into
    ``test_live_aggregate_is_idempotent...``'s setup, which then hit a dead
    tracking server. With this test last, no later test in the module
    depends on mlflow coming back up inside the test process at all —
    ``scripts/run_container_test.sh``'s ``trap cleanup EXIT`` tears the
    whole stack down (``compose down -v``) immediately afterward regardless
    of container state. The flake mechanism is removed structurally, not
    tuned away with a longer timeout.
    """
    untracked_cfg = _kill_mid_run_config(tmp_path / "untracked", name="kill-mid-run-untracked")
    untracked_results = tmp_path / "untracked.jsonl"
    untracked_trainer = Trainer(untracked_cfg, results_path=untracked_results)
    untracked_trainer.fit(verbose=False)

    tracked_cfg = _kill_mid_run_config(tmp_path / "tracked", name="kill-mid-run-tracked")
    tracked_results = tmp_path / "tracked.jsonl"
    sink = build_sink(tracked_cfg, tiny=True, results_path=tracked_results)
    tracked_trainer = Trainer(tracked_cfg, results_path=tracked_results, sink=sink)

    killer = threading.Timer(1.0, _stop_mlflow)
    killer.start()
    try:
        tracked_trainer.fit(verbose=False)  # must complete despite tracking dying mid-loop
    finally:
        killer.cancel()
        killer.join()  # block until any already-fired _stop_mlflow() call has returned

    untracked_records = [
        json.loads(line) for line in untracked_results.read_text().splitlines() if line
    ]
    tracked_records = [
        json.loads(line) for line in tracked_results.read_text().splitlines() if line
    ]

    print(
        f"untracked: {len(untracked_records)} records; tracked-then-killed: {len(tracked_records)}"
    )
    assert len(tracked_records) == len(untracked_records), (
        "results/*.jsonl must be COMPLETE regardless of MLflow's mid-run state",
        untracked_records,
        tracked_records,
    )
    untracked_events = [r["event"] for r in untracked_records]
    tracked_events = [r["event"] for r in tracked_records]
    assert tracked_events == untracked_events

    mlflow_sink = sink.best_effort[0]  # type: ignore[attr-defined]
    assert mlflow_sink.failed_writes > 0, "the kill must have actually caused counted failures"
