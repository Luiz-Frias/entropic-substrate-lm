"""Unit tests for :mod:`entropic_substrate.mlflow_aggregate` — seed aggregation.

Against the same ``FakeMlflowClient`` as ``tests/test_sinks.py`` (imported
from there rather than duplicated). Live idempotency against a real server
is proven separately by the ``integration``-marked tests.
"""

from __future__ import annotations

import statistics

import pytest

from entropic_substrate.mlflow_aggregate import PARENT_TAG, aggregate
from test_sinks import FakeMlflowClient

pytestmark = pytest.mark.unit


def _seed_run(
    client: FakeMlflowClient, exp_id: str, *, config_hash: str, seed: int, npb: float
) -> str:
    run = client.create_run(exp_id, tags={"config_hash": config_hash, "exp": "e1-fisher"})
    client.log_param(run.info.run_id, "seed", seed)
    client.log_metric(run.info.run_id, "nats_per_byte", npb)
    client.log_metric(run.info.run_id, "loss", npb * 2)
    client.log_metric(run.info.run_id, "bits_per_byte", npb / 0.693)
    client.log_metric(run.info.run_id, "perplexity", 2.0**npb)
    client.set_terminated(run.info.run_id)
    return run.info.run_id


def test_no_parent_when_fewer_than_n_seeds_exist() -> None:
    """NO PARENT WITHOUT A COMPLETE AGGREGATE — absence is the signal."""
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    _seed_run(client, exp_id, config_hash="h1", seed=1, npb=1.5)
    _seed_run(client, exp_id, config_hash="h1", seed=2, npb=1.6)

    summaries = aggregate(client, experiment_name="exp", n_seeds=3)

    assert summaries == []
    assert not any(r.data.tags.get(PARENT_TAG) == "1" for r in client._runs.values())


def test_parent_created_once_three_seeds_are_present() -> None:
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    _seed_run(client, exp_id, config_hash="h1", seed=1, npb=1.5)
    _seed_run(client, exp_id, config_hash="h1", seed=2, npb=1.6)
    _seed_run(client, exp_id, config_hash="h1", seed=3, npb=1.4)

    summaries = aggregate(client, experiment_name="exp", n_seeds=3)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["n_seeds"] == 3
    assert summary["nats_per_byte"]["mean"] == pytest.approx((1.5 + 1.6 + 1.4) / 3)

    parents = [r for r in client._runs.values() if r.data.tags.get(PARENT_TAG) == "1"]
    assert len(parents) == 1
    parent = parents[0]
    assert parent.data.params["n_seeds"] == "3"
    assert parent.data.metrics["nats_per_byte_mean"] == pytest.approx((1.5 + 1.6 + 1.4) / 3)
    assert parent.data.metrics["nats_per_byte_std"] == pytest.approx(
        statistics.pstdev([1.5, 1.6, 1.4])
    )


def test_running_twice_produces_one_parent_with_identical_stats() -> None:
    """Idempotency: two invocations, one parent, same mean/std/n_seeds."""
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    _seed_run(client, exp_id, config_hash="h1", seed=1, npb=1.5)
    _seed_run(client, exp_id, config_hash="h1", seed=2, npb=1.6)
    _seed_run(client, exp_id, config_hash="h1", seed=3, npb=1.4)

    first = aggregate(client, experiment_name="exp", n_seeds=3)
    second = aggregate(client, experiment_name="exp", n_seeds=3)

    assert first[0]["parent_run_id"] == second[0]["parent_run_id"]
    assert first[0]["n_seeds"] == second[0]["n_seeds"]
    assert first[0]["nats_per_byte"] == second[0]["nats_per_byte"]

    parents = [r for r in client._runs.values() if r.data.tags.get(PARENT_TAG) == "1"]
    assert len(parents) == 1, "two aggregate() calls must not create two parents"


def test_children_are_adopted_via_parent_run_id_tag() -> None:
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    run_ids = [
        _seed_run(client, exp_id, config_hash="h1", seed=s, npb=1.5 + s * 0.01) for s in (1, 2, 3)
    ]

    summaries = aggregate(client, experiment_name="exp", n_seeds=3)
    parent_id = summaries[0]["parent_run_id"]

    for run_id in run_ids:
        assert client._runs[run_id].data.tags["mlflow.parentRunId"] == parent_id


def test_two_config_hashes_aggregate_independently() -> None:
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    for s in (1, 2, 3):
        _seed_run(client, exp_id, config_hash="h1", seed=s, npb=1.5)
    for s in (1, 2, 3):
        _seed_run(client, exp_id, config_hash="h2", seed=s, npb=2.5)

    summaries = aggregate(client, experiment_name="exp", n_seeds=3)

    by_hash = {s["config_hash"]: s for s in summaries}
    assert set(by_hash) == {"h1", "h2"}
    assert by_hash["h1"]["nats_per_byte"]["mean"] == pytest.approx(1.5)
    assert by_hash["h2"]["nats_per_byte"]["mean"] == pytest.approx(2.5)


def test_abort_between_seeds_leaves_no_parent() -> None:
    """A group short one seed (simulating an aborted ladder) creates nothing."""
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    _seed_run(client, exp_id, config_hash="h1", seed=1, npb=1.5)
    _seed_run(client, exp_id, config_hash="h1", seed=2, npb=1.6)
    # seed=3 never ran — the ladder aborted between seeds.

    aggregate(client, experiment_name="exp", n_seeds=3)
    aggregate(client, experiment_name="exp", n_seeds=3)  # idempotent even while incomplete

    assert not any(r.data.tags.get(PARENT_TAG) == "1" for r in client._runs.values())


def test_missing_experiment_returns_empty_without_raising() -> None:
    client = FakeMlflowClient()
    summaries = aggregate(client, experiment_name="does-not-exist", n_seeds=3)
    assert summaries == []


def test_parent_runs_are_excluded_from_their_own_child_group() -> None:
    """A parent run's own config_hash tag must not make it count as a 4th seed."""
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    for s in (1, 2, 3):
        _seed_run(client, exp_id, config_hash="h1", seed=s, npb=1.5)

    aggregate(client, experiment_name="exp", n_seeds=3)
    # Running again: the parent (also tagged config_hash="h1") must not be
    # picked up as a leaf, or the group would balloon to 4 "seeds".
    summaries = aggregate(client, experiment_name="exp", n_seeds=3)
    assert summaries[0]["n_seeds"] == 3


def test_incomplete_metric_across_group_is_skipped_not_averaged_partially() -> None:
    """One run missing ONE metric must exclude only that metric, never silently

    average the 2 of 3 that do have it — while other, fully-present metrics
    still aggregate normally.
    """
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    for npb in (1.5, 1.6, 1.7):
        run = client.create_run(exp_id, tags={"config_hash": "h1", "exp": "e1-fisher"})
        client.log_metric(run.info.run_id, "loss", npb * 2)  # present on all 3
        if npb != 1.7:  # nats_per_byte missing on exactly one run
            client.log_metric(run.info.run_id, "nats_per_byte", npb)

    summaries = aggregate(client, experiment_name="exp", n_seeds=3)
    assert len(summaries) == 1
    assert "nats_per_byte" not in summaries[0]
    assert summaries[0]["loss"]["mean"] == pytest.approx((3.0 + 3.2 + 3.4) / 3)
