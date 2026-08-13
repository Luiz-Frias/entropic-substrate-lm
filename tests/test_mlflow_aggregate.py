"""Unit tests for :mod:`entropic_substrate.mlflow_aggregate` — seed aggregation.

Mostly against the same ``FakeMlflowClient`` as ``tests/test_sinks.py``
(imported from there rather than duplicated); a few tests use a real,
local, sqlite-backed ``mlflow.tracking.MlflowClient`` instead (retroactive
nesting, growth-under-write-once-params), to verify against mlflow's actual
storage semantics rather than an in-memory approximation of them. Live
idempotency against a real server is proven separately by the
``integration``-marked tests. The final section drives
:func:`entropic_substrate.cli.main` end to end (the ``mlflow-aggregate``
subcommand), covering the CLI's controlled-refusal wrapping around
:func:`aggregate`.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import pytest

from entropic_substrate.mlflow_aggregate import PARENT_TAG, aggregate
from test_sinks import FakeMlflowClient

pytestmark = pytest.mark.unit


def _seed_run(
    # `Any`, not `FakeMlflowClient`: this helper is called with a real
    # `mlflow.tracking.MlflowClient` too (see
    # test_retroactive_parent_tag_nests_in_the_real_mlflow_ui below) — both
    # are used purely structurally (create_run/log_param/log_metric/
    # set_terminated), so the narrower type would be dishonest.
    client: Any,
    exp_id: str,
    *,
    config_hash: str,
    seed: int,
    npb: float,
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
    # A TAG, not a param — see mlflow_aggregate.py's module docstring: params
    # are write-once server-side, and n_seeds must be updatable on re-runs.
    assert parent.data.tags["n_seeds"] == "3"
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


def test_growing_seed_group_updates_the_existing_parent_not_a_stale_one() -> None:
    """FIX 1 (audit finding, round A): a parent must always be updatable to
    the CURRENT aggregate, never left EXISTING + STALE + UNUPDATABLE — the
    third state the "no parent without a complete aggregate" invariant never
    contemplated. Simulates the operator sequence from the finding: an
    operator adds a 4th seed for confidence AFTER a 3-seed parent already
    exists. Before the fix (``log_param``, write-once), the second
    ``aggregate()`` call raised ``MlflowException`` here too — see
    ``FakeMlflowClient.log_param``'s write-once enforcement in
    ``tests/test_sinks.py``, and the real-server twin of this test below.
    """
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    _seed_run(client, exp_id, config_hash="h1", seed=1, npb=1.5)
    _seed_run(client, exp_id, config_hash="h1", seed=2, npb=1.6)
    _seed_run(client, exp_id, config_hash="h1", seed=3, npb=1.4)

    first = aggregate(client, experiment_name="exp", n_seeds=3)
    assert first[0]["n_seeds"] == 3

    _seed_run(client, exp_id, config_hash="h1", seed=4, npb=1.45)  # operator adds a 4th seed
    second = aggregate(client, experiment_name="exp", n_seeds=3)  # must not raise

    assert second[0]["parent_run_id"] == first[0]["parent_run_id"], (
        "growth must update the SAME parent, not create a second one"
    )
    assert second[0]["n_seeds"] == 4
    assert second[0]["nats_per_byte"]["mean"] == pytest.approx((1.5 + 1.6 + 1.4 + 1.45) / 4)

    parents = [r for r in client._runs.values() if r.data.tags.get(PARENT_TAG) == "1"]
    assert len(parents) == 1, "growth must not create a second parent"
    assert parents[0].data.tags["n_seeds"] == "4"


def test_growing_seed_group_against_a_real_mlflow_client_does_not_raise(tmp_path: Path) -> None:
    """Real-server twin of the fake-based growth test above — against a real
    (sqlite-backed, local) ``mlflow.tracking.MlflowClient``, not the fake, so
    this exercises MLflow's actual write-once-param enforcement rather than
    an in-memory approximation of it. This is the exact scenario captured
    verbatim in the phase's ``reproBefore`` (aggregate 3 seeds, add a 4th,
    aggregate again) — pinned here so a regression can never land silently.
    """
    import mlflow.tracking

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    exp_id = client.create_experiment("real-growth-probe")
    _seed_run(client, exp_id, config_hash="h1", seed=1, npb=1.5)
    _seed_run(client, exp_id, config_hash="h1", seed=2, npb=1.6)
    _seed_run(client, exp_id, config_hash="h1", seed=3, npb=1.4)

    first = aggregate(client, experiment_name="real-growth-probe", n_seeds=3)
    assert first[0]["n_seeds"] == 3

    _seed_run(client, exp_id, config_hash="h1", seed=4, npb=1.45)
    second = aggregate(client, experiment_name="real-growth-probe", n_seeds=3)  # must not raise

    assert second[0]["parent_run_id"] == first[0]["parent_run_id"]
    assert second[0]["n_seeds"] == 4
    refreshed_parent = client.get_run(second[0]["parent_run_id"])
    assert refreshed_parent.data.tags["n_seeds"] == "4"


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


def test_retroactive_parent_tag_nests_in_the_real_mlflow_ui(tmp_path: Path) -> None:
    """Empirical answer to the phase-scope question the spec flagged as unverified:

    "VERIFY empirically whether setting mlflow.parentRunId on EXISTING runs
    nests them retroactively in the MLflow UI... Do not assume either way."

    Against a REAL ``mlflow.tracking.MlflowClient`` (sqlite-backed, local,
    no server process, no network — still hermetic/fast) rather than
    ``FakeMlflowClient``, so this exercises mlflow's actual tag storage and
    retrieval round-trip, not an in-memory dict.

    Findings, both checked directly against the installed mlflow package:

    1. ``mlflow.entities.RunInfo`` in this version carries NO
       ``parent_run_id`` field at all (confirmed via ``vars(run.info)`` on a
       freshly created run) — there is no separate, creation-time-only slot
       nesting could be reading instead of the tag.
    2. The MLflow UI's shipped, built JS bundle
       (``mlflow/server/js/build/static/js/main.*.js``, vendored inside the
       installed ``mlflow`` package) determines run-tree nesting by reading
       the ``"mlflow.parentRunId"`` key out of each run's ``data.tags`` at
       render time — the exact same tag `aggregate()` retroactively sets via
       ``client.set_tag()``.

    So: YES, retroactive nesting works — the UI does not distinguish a tag
    set at ``create_run()`` time from one set afterward, because it never
    reads a separate immutable field to begin with.
    """
    import mlflow.tracking

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    exp_id = client.create_experiment("real-nesting-probe")
    run_ids = [
        _seed_run(client, exp_id, config_hash="h1", seed=s, npb=1.5 + s * 0.01) for s in (1, 2, 3)
    ]

    summaries = aggregate(client, experiment_name="real-nesting-probe", n_seeds=3)
    parent_id = summaries[0]["parent_run_id"]

    for run_id in run_ids:
        # Fresh get_run() per child — a real read round-trip through mlflow's
        # own store, not the value this process just wrote in memory.
        refreshed = client.get_run(run_id)
        assert refreshed.data.tags["mlflow.parentRunId"] == parent_id
        # RunInfo has no parent_run_id attribute to fall back on — the tag
        # really is the only signal, confirming point 1 above inline.
        assert not hasattr(refreshed.info, "parent_run_id")


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


# ------------------------------------------------------- cmd_mlflow_aggregate (CLI)
def test_cli_mlflow_aggregate_handles_seed_group_growth_end_to_end(tmp_path: Path) -> None:
    """FIX 1, end to end: drive the ``mlflow-aggregate`` subcommand through
    :func:`entropic_substrate.cli.main` (not just the library function)
    against a real sqlite-backed store, with the seed group growing between
    the two invocations — the exact operator sequence the audit finding
    described. Must return 0 both times, never raise past ``main()``.
    """
    import mlflow.tracking

    from entropic_substrate.cli import main

    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    exp_id = client.create_experiment("cli-growth")
    _seed_run(client, exp_id, config_hash="h1", seed=1, npb=1.5)
    _seed_run(client, exp_id, config_hash="h1", seed=2, npb=1.6)
    _seed_run(client, exp_id, config_hash="h1", seed=3, npb=1.4)

    argv = ["mlflow-aggregate", "--tracking-uri", tracking_uri, "--experiment", "cli-growth"]
    assert main(argv) == 0

    _seed_run(client, exp_id, config_hash="h1", seed=4, npb=1.45)  # operator adds a 4th seed
    assert main(argv) == 0  # must not raise past main()


def test_cli_mlflow_aggregate_reports_mlflow_exception_as_controlled_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defense in depth, independent of the specific n_seeds bug FIX 1 makes
    unreachable: ANY ``MlflowException`` surfacing from :func:`aggregate` at
    the CLI boundary must read as a printed refusal + exit 2 — this repo's
    gate idiom (same shape as ``CheckpointFormatError`` in ``main()``) —
    never a raw traceback. The next MlflowException-raising defect in this
    path must land here, not as an unhandled crash.
    """
    from mlflow.exceptions import MlflowException

    import entropic_substrate.cli as cli_module

    def _boom(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        msg = "simulated server-side rejection"
        raise MlflowException(msg)

    monkeypatch.setattr(cli_module, "aggregate", _boom)

    code = cli_module.main(
        ["mlflow-aggregate", "--tracking-uri", f"sqlite:///{tmp_path}/mlflow.db"]
    )
    assert code == 2
