"""Retroactive MLflow attachment (Lane A's write side) — best-effort by
design. All against ``FakeMlflowClient`` (imported from test_sinks, same
idiom as tests/test_mlflow_aggregate.py): no network, no real server.
"""

from __future__ import annotations

import pytest

from entropic_substrate.analysis.mlflow_attach import (
    attach_figures_to_run,
    attach_seed_band_to_aggregate,
)
from entropic_substrate.mlflow_aggregate import CONFIG_HASH_TAG, PARENT_TAG
from test_sinks import FakeMlflowClient

pytestmark = pytest.mark.unit


def _figure():
    from matplotlib.figure import Figure

    return Figure()


# --------------------------------------------------------- attach_figures_to_run
def test_attach_figures_to_run_logs_each_figure_under_charts(tmp_path) -> None:
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    run = client.create_run(exp_id)

    attached = attach_figures_to_run(
        client, run.info.run_id, {"e1_paired": _figure(), "e2_texture": _figure()}
    )

    assert sorted(attached) == ["charts/e1_paired.png", "charts/e2_texture.png"]
    logged_files = sorted(name for _run_id, name in client.figures_logged)
    assert logged_files == ["charts/e1_paired.png", "charts/e2_texture.png"]
    assert all(rid == run.info.run_id for rid, _name in client.figures_logged)


def test_attach_figures_to_run_is_best_effort_one_failure_does_not_sink_the_rest() -> None:
    """MLflow writes are best-effort-style calls by run_id — never authoritative,
    never required for a chart to exist (repo SACRED INVARIANTS)."""
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    run = client.create_run(exp_id)
    client.kill()  # every subsequent call raises ConnectionError

    attached = attach_figures_to_run(client, run.info.run_id, {"e1_paired": _figure()})

    assert attached == []  # nothing attached, but no exception propagated


def test_attach_figures_to_run_on_an_unknown_run_id_does_not_raise() -> None:
    client = FakeMlflowClient()
    attached = attach_figures_to_run(client, "no-such-run", {"e1_paired": _figure()})
    assert attached == []


# ------------------------------------------------------ attach_seed_band_to_aggregate
def test_attach_seed_band_to_aggregate_finds_the_parent_and_attaches(tmp_path) -> None:
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    parent = client.create_run(exp_id, tags={PARENT_TAG: "1", CONFIG_HASH_TAG: "abc123"})

    attached = attach_seed_band_to_aggregate(
        client,
        experiment_name="exp",
        config_hash="abc123",
        figures={"e1_paired_seed_band": _figure()},
    )

    assert attached == ["charts/e1_paired_seed_band.png"]
    assert client.figures_logged == [(parent.info.run_id, "charts/e1_paired_seed_band.png")]


def test_attach_seed_band_to_aggregate_without_a_parent_yet_is_a_noop() -> None:
    """NO PARENT WITHOUT A COMPLETE AGGREGATE (entropic_substrate.mlflow_aggregate)
    — this must never race ahead of that invariant, and never raise either."""
    client = FakeMlflowClient()
    client.create_experiment("exp")

    attached = attach_seed_band_to_aggregate(
        client,
        experiment_name="exp",
        config_hash="abc123",
        figures={"e1_paired_seed_band": _figure()},
    )

    assert attached == []


def test_attach_seed_band_to_aggregate_on_an_unknown_experiment_is_a_noop() -> None:
    client = FakeMlflowClient()
    attached = attach_seed_band_to_aggregate(
        client, experiment_name="no-such-experiment", config_hash="abc123", figures={"x": _figure()}
    )
    assert attached == []
