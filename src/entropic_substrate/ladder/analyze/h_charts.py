"""H chart set: a pure transform from the aggregate document to matplotlib figures.

Input is the dict :func:`~entropic_substrate.ladder.analyze.h_aggregate.aggregate_h_runs`
returns; output is figures. No file I/O, no MLflow. Series styles come from the public
palette API (Okabe-Ito plus a linestyle, a marker and a direct label per arm, because the
operator is red-green colorblind). No log or symlog axis is used anywhere: the captured
``future_nll_per_dimension`` is negative on most prediction rows and a log axis would mask
those points without an error.

A chart whose data the capture does not hold is named in the returned ``absent`` mapping
with its reason, never rendered empty. An aggregate with no arms raises.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from entropic_substrate.ladder.analyze.palette import OKABE_ITO, SeriesStyle, head_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["build_h_figures"]

_EVAL_K = "eval/log_marginal_k"
_ESS_K = "eval/ess_k"


def _figure(figsize: tuple[float, float]) -> Figure:
    from matplotlib.figure import Figure as _Figure

    return _Figure(figsize=figsize, layout="constrained")


def _styles(labels: Sequence[str]) -> dict[str, SeriesStyle]:
    return {label: head_style(i, label=label) for i, label in enumerate(labels)}


def _plot(
    ax: Axes,
    style: SeriesStyle,
    steps: Sequence[int],
    values: Sequence[float],
    *,
    markers_on_every_point: bool = False,
) -> None:
    ax.plot(
        list(steps),
        list(values),
        color=style.color,
        linestyle=style.linestyle,
        marker=style.marker,
        markevery=1 if markers_on_every_point else max(len(steps) // 8, 1),
        markersize=4,
        linewidth=1.3,
        label=style.label,
    )


def _series(rows: Sequence[Mapping[str, Any]], label: str) -> tuple[list[int], list[float]]:
    """(steps, values) for one arm, nulls skipped (a null is plotted as a gap)."""
    steps: list[int] = []
    values: list[float] = []
    for row in rows:
        value = row["values"][label]
        if value is None:
            continue
        steps.append(int(row["step"]))
        values.append(float(value))
    return steps, values


def _boundary(ax: Axes, aggregate: Mapping[str, Any]) -> None:
    firsts = {
        a["first_prediction_step"]
        for a in aggregate["arms"].values()
        if a["first_prediction_step"] is not None
    }
    for first in sorted(firsts):
        ax.axvline(first, color=OKABE_ITO["black"], linestyle=":", linewidth=1.0)
        ax.text(
            first,
            1.0,
            " prediction starts",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            va="top",
        )


def _training_curves(aggregate: Mapping[str, Any], styles: Mapping[str, SeriesStyle]) -> Figure:
    fig = _figure((9.0, 6.0))
    metrics = aggregate["training"]["metrics"]
    for ax, metric in zip(
        fig.subplots(2, 1, sharex=True), ("objective", "decoder_ce"), strict=True
    ):
        for label, style in styles.items():
            steps, values = _series(metrics[metric], label)
            _plot(ax, style, steps, values)
        _boundary(ax, aggregate)
        ax.set_ylabel(metric)
        ax.legend(loc="upper right", fontsize=8)
    fig.axes[-1].set_xlabel("global optimizer step")
    fig.suptitle("H training curves per arm (committed rows, global step axis)")
    return fig


def _future_nll(aggregate: Mapping[str, Any], styles: Mapping[str, SeriesStyle]) -> Figure | None:
    rows = aggregate["training"]["metrics"]["future_nll_per_dimension"]
    series = {label: _series(rows, label) for label in styles}
    if not any(steps for steps, _ in series.values()):
        return None
    fig = _figure((9.0, 4.5))
    ax = fig.subplots()
    for label, style in styles.items():
        steps, values = series[label]
        _plot(ax, style, steps, values)
    ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=1.0)
    ax.text(
        0.01,
        0.02,
        "y = 0 reference; values drawn as captured (linear axis)",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.set_xlabel("global optimizer step (prediction rows only; null on codec warm-up)")
    ax.set_ylabel("future_nll_per_dimension")
    ax.set_title("future_nll_per_dimension per arm")
    ax.legend(loc="best", fontsize=8)
    return fig


def _evaluation_rows(
    aggregate: Mapping[str, Any], styles: Mapping[str, SeriesStyle]
) -> Figure | None:
    metrics = aggregate["evaluation"]["metrics"]
    counts = sorted(
        {name[len(_EVAL_K) :] for name in metrics if name.startswith(_EVAL_K)},
        key=lambda k: (len(k), k),
    )
    if not any(metrics[f"{_EVAL_K}{k}"] for k in counts):
        return None
    fig = _figure((4.5 * len(counts), 6.0))
    axes = fig.subplots(2, len(counts), squeeze=False)
    for column, k in enumerate(counts):
        for row_index, (prefix, ylabel) in enumerate(((_EVAL_K, "log marginal"), (_ESS_K, "ESS"))):
            ax = axes[row_index][column]
            rows = metrics.get(f"{prefix}{k}", [])
            for label, style in styles.items():
                steps, values = _series(rows, label)
                _plot(ax, style, steps, values, markers_on_every_point=True)
            ax.set_title(f"{ylabel}, k={k}", fontsize=9)
            ax.set_xlabel("global optimizer step")
            ax.legend(loc="best", fontsize=7)
    fig.suptitle("H evaluation rows per arm (mean over examples, as projected)")
    return fig


def _publication(aggregate: Mapping[str, Any], styles: Mapping[str, SeriesStyle]) -> Figure | None:
    rows = aggregate["publication_seconds"]["values"]
    if not rows:
        return None
    fig = _figure((7.5, 4.5))
    ax = fig.subplots()
    for label, style in styles.items():
        steps, values = _series(rows, label)
        _plot(ax, style, steps, values, markers_on_every_point=True)
    ax.set_xlabel("checkpoint (global optimizer step)")
    ax.set_ylabel("publication_seconds")
    ax.set_title("Checkpoint publication timing per arm")
    ax.legend(loc="best", fontsize=8)
    return fig


def build_h_figures(aggregate: Mapping[str, Any]) -> tuple[dict[str, Figure], dict[str, str]]:
    """``(figures, absent)``: the chart set the aggregate's data supports, and the name
    and reason of every chart it does not. Raises ``ValueError`` on an empty aggregate."""
    arms = aggregate.get("arms")
    if not arms or not aggregate["training"]["paired_steps"]:
        msg = "no arms or no paired training steps in the aggregate -- nothing to chart"
        raise ValueError(msg)
    styles = _styles(sorted(arms))
    figures: dict[str, Figure] = {"h_training_curves": _training_curves(aggregate, styles)}
    absent: dict[str, str] = {}
    optional = (
        ("h_future_nll_per_dimension", _future_nll, "no non-null future_nll_per_dimension row"),
        ("h_evaluation_rows", _evaluation_rows, "no evaluation row on the shared step axis"),
        ("h_publication_timing", _publication, "no checkpoint publication timing on every arm"),
    )
    for name, build, reason in optional:
        figure = build(aggregate, styles)
        if figure is None:
            absent[name] = reason
        else:
            figures[name] = figure
    return figures, absent
