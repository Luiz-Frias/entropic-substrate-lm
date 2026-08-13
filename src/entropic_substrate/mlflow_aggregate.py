"""Idempotent seed aggregation over MLflow child runs, grouped by ``config_hash``.

Seed runs are written FLAT — params ``seed``/``exp``, tag ``config_hash``
(see :func:`entropic_substrate.sinks.build_sink`). :func:`aggregate` finds
every LEAF run (one without the ``aggregate.is_parent`` tag) that shares a
``config_hash``, and creates/updates exactly ONE parent run per group — but
only once the group has at least ``n_seeds`` members.

NO PARENT WITHOUT A COMPLETE AGGREGATE: a parent's absence for an
in-progress ``config_hash`` IS the loud signal that the ladder is not done —
see the repo's SACRED INVARIANTS. Idempotent: running this twice against
unchanged children reuses the SAME parent run (found by its
``aggregate.is_parent`` + ``config_hash`` tags) and recomputes the same
mean/std/n_seeds, rather than creating a second parent.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from entropic_substrate.logging import logger
from entropic_substrate.sinks import CONFIG_HASH_TAG, EXP_TAG

__all__ = ["DEFAULT_METRICS", "PARENT_TAG", "aggregate"]

DEFAULT_METRICS: tuple[str, ...] = ("nats_per_byte", "loss", "bits_per_byte", "perplexity")
PARENT_TAG = "aggregate.is_parent"


def _leaf_runs(client: Any, experiment_id: str) -> list[Any]:
    """Every run in the experiment that carries ``config_hash`` and is not itself a parent."""
    runs = client.search_runs([experiment_id], max_results=5000)
    return [
        run
        for run in runs
        if run.data.tags.get(CONFIG_HASH_TAG) and run.data.tags.get(PARENT_TAG) != "1"
    ]


def _group_by_config_hash(runs: Sequence[Any]) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = {}
    for run in runs:
        groups.setdefault(run.data.tags[CONFIG_HASH_TAG], []).append(run)
    return groups


def _existing_parent(client: Any, experiment_id: str, chash: str) -> Any | None:
    found = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`{PARENT_TAG}` = '1' and tags.`{CONFIG_HASH_TAG}` = '{chash}'",
        max_results=1,
    )
    return found[0] if found else None


def _group_stats(runs: Sequence[Any], metrics: Sequence[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [run.data.metrics[metric] for run in runs if metric in run.data.metrics]
        if len(values) != len(runs):
            continue  # incomplete for this metric across the group — skip it, don't guess
        stats[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    return stats


def aggregate(
    client: Any,
    *,
    experiment_name: str,
    n_seeds: int = 3,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> list[dict[str, Any]]:
    """Idempotently aggregate seed runs by ``config_hash``. Returns one summary per group."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"mlflow-aggregate: no experiment named {experiment_name!r}")
        return []

    groups = _group_by_config_hash(_leaf_runs(client, experiment.experiment_id))

    summaries: list[dict[str, Any]] = []
    for chash, runs in groups.items():
        if len(runs) < n_seeds:
            logger.info(
                f"mlflow-aggregate: {chash} has {len(runs)}/{n_seeds} seeds — "
                "not aggregating (no parent without a complete set)."
            )
            continue

        stats = _group_stats(runs, metrics)
        if not stats:
            logger.warning(f"mlflow-aggregate: {chash} has no complete metrics to aggregate")
            continue

        exp_name = runs[0].data.tags.get(EXP_TAG, experiment_name)
        parent = _existing_parent(client, experiment.experiment_id, chash)
        if parent is None:
            parent = client.create_run(
                experiment.experiment_id,
                tags={PARENT_TAG: "1", CONFIG_HASH_TAG: chash, EXP_TAG: exp_name},
            )
            client.set_tag(parent.info.run_id, "mlflow.runName", f"{exp_name}-aggregate-{chash}")
        parent_id = parent.info.run_id

        client.log_param(parent_id, "n_seeds", len(runs))
        for metric, values in stats.items():
            client.log_metric(parent_id, f"{metric}_mean", values["mean"])
            client.log_metric(parent_id, f"{metric}_std", values["std"])
        client.set_terminated(parent_id, status="FINISHED")

        # Retroactive nesting: adopt every child under the now-complete parent.
        #
        # VERIFIED (tests/test_mlflow_aggregate.py::
        # test_retroactive_parent_tag_nests_in_the_real_mlflow_ui), not
        # assumed: against a real (sqlite-backed, local) mlflow.tracking
        # store, (a) mlflow.entities.RunInfo carries no separate
        # creation-time-only parent_run_id field to fall back on in this
        # mlflow version, and (b) the MLflow UI's own shipped JS bundle
        # (mlflow/server/js/build/static/js/main.*.js) determines run-tree
        # nesting purely by reading the "mlflow.parentRunId" key out of
        # run.data.tags at render time. A tag set after run creation reads
        # back identically to one set at create_run() time, so this DOES
        # nest retroactively in the UI.
        for run in runs:
            client.set_tag(run.info.run_id, "mlflow.parentRunId", parent_id)

        summary = {"config_hash": chash, "parent_run_id": parent_id, "n_seeds": len(runs), **stats}
        summaries.append(summary)
        logger.info(f"mlflow-aggregate: {chash} ({exp_name}) -> parent {parent_id} {summary}")

    return summaries
