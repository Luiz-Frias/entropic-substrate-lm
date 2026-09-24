"""Retroactive MLflow attachment — Lane A's write side, best-effort by design.

MLflow writes here are best-effort-style client calls by run_id: never
authoritative, never required for a chart to exist (repo SACRED INVARIANTS).
Every figure this module attaches was already rendered from validated
telemetry (``entropic_substrate.analysis.charts``/``series``) BEFORE this
module is ever called — a network hiccup, a stale run_id, or a not-yet-
existing aggregate parent must never take a rendered chart down with it, so
every failure here is caught, logged, and skipped, exactly the
``except Exception as exc: logger.warning(...)`` idiom
:class:`entropic_substrate.sinks.MlflowSink` already uses for its own
best-effort writes (dataset lineage, system metrics).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.capture.reader.mlflow_aggregate import find_parent_run

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["attach_figures_to_run", "attach_seed_band_to_aggregate"]


def attach_figures_to_run(client: Any, run_id: str, figures: dict[str, Figure]) -> list[str]:
    """Best-effort ``MlflowClient.log_figure`` for each figure, under ``charts/``.

    Returns the ``artifact_file`` paths actually attached; a failure on ONE
    figure (bad ``run_id``, connection dropped mid-loop) is logged and
    skipped — it must never take down figures that would otherwise have
    succeeded, and never raises past this function.
    """
    attached: list[str] = []
    for name, figure in figures.items():
        artifact_file = f"charts/{name}.png"
        try:
            client.log_figure(run_id, figure, artifact_file)
        except Exception as exc:  # best-effort by design — see module docstring
            logger.warning(f"charts: could not attach {artifact_file!r} to run {run_id}: {exc!r}")
            continue
        attached.append(artifact_file)
    return attached


def attach_seed_band_to_aggregate(
    client: Any,
    *,
    experiment_name: str,
    config_hash: str,
    figures: dict[str, Figure],
) -> list[str]:
    """Attach seed-band chart variants to the mlflow-aggregate PARENT run.

    Returns ``[]`` (with a warning, never an exception) when the experiment
    or the parent doesn't exist yet: NO PARENT WITHOUT A COMPLETE AGGREGATE
    is :mod:`entropic_substrate.mlflow_aggregate`'s own invariant, and this
    function must not race ahead of it by inventing one.
    """
    try:
        experiment = client.get_experiment_by_name(experiment_name)
    except Exception as exc:  # best-effort by design — see module docstring
        logger.warning(f"charts: could not reach the tracking server: {exc!r}")
        return []
    if experiment is None:
        logger.warning(
            f"charts: experiment {experiment_name!r} not found — skipping seed-band attach"
        )
        return []
    try:
        parent = find_parent_run(client, experiment.experiment_id, config_hash)
    except Exception as exc:  # best-effort by design — see module docstring
        logger.warning(f"charts: could not look up the aggregate parent: {exc!r}")
        return []
    if parent is None:
        logger.warning(
            f"charts: no aggregate parent for config_hash={config_hash!r} yet "
            "(mlflow-aggregate has not completed for this group) — skipping seed-band attach"
        )
        return []
    return attach_figures_to_run(client, parent.info.run_id, figures)
