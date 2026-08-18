"""The six E1/E2/E3 interpretation charts — matplotlib, Lane A (retroactive
MLflow attachment via ``entropic_substrate.analysis.mlflow_attach``).

Every function here is a pure transform: ``entropic_substrate.analysis.
series`` objects in, a ``matplotlib.figure.Figure`` out. No file I/O, no
MLflow client — that keeps the SAME rendering code reachable from Lane B's
plotly notebooks' data-prep helpers without dragging a JSONL path through
this module.

Charts RENDER predicates visually; they never decide. Every threshold or CI
this module draws is also LABELED on the figure (text, not just a line) —
"drawn" and "computed but hidden" are not the same thing, and the latter is
exactly what this repo's SACRED INVARIANTS forbid. Colorblind rule: Okabe-Ito
hex (``entropic_substrate.analysis.palette``) plus a redundant non-color
channel (linestyle/marker/direct label) on every semantic distinction —
matplotlib is imported LAZILY (module-function-local), matching the
lazy-import idiom ``entropic_substrate.cli.cmd_capture_results`` already uses
for its own dev-only dependency (mlflow), so importing this module never
forces the analysis dependency group onto a caller that only wants the
dataclasses in ``series``/``palette``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from entropic_substrate.analysis.palette import (
    OKABE_ITO,
    SeriesStyle,
    control_style,
    e2_style,
    e3_style,
    fisher_style,
    head_style,
)
from entropic_substrate.analysis.series import T_0_9875, EvalSeriesRun, TrainSeriesRun, Verdict
from entropic_substrate.geometry.kappa import KAPPA_MAX, KAPPA_MIN
from entropic_substrate.telemetry import max_kappa_spread

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "fig_e1_log_steps_washout",
    "fig_e1_paired_loss_curves",
    "fig_e2_spike_variance_texture",
    "fig_e3_kappa_divergence_vs_loss_gap",
    "fig_e3_kappa_trajectories",
    "fig_e3_loss_separation_vs_e2",
]


def _figure(figsize: tuple[float, float]) -> Figure:
    from matplotlib.figure import Figure as _Figure

    return _Figure(figsize=figsize, layout="constrained")


def _series_xy(run: EvalSeriesRun | TrainSeriesRun, metric: str) -> tuple[list[int], list[float]]:
    steps = [r.step for r in run.records]
    values = [getattr(r, metric) for r in run.records]
    return steps, values


def _plot_seed_band(
    ax: Axes, runs: Sequence[EvalSeriesRun], style: SeriesStyle, *, metric: str
) -> None:
    """Thin per-seed lines (one legend entry per arm) + a shaded min-max band
    across whichever steps every seed in ``runs`` shares."""
    if not runs:
        return
    for i, run in enumerate(runs):
        steps, values = _series_xy(run, metric)
        ax.plot(
            steps,
            values,
            color=style.color,
            linestyle=style.linestyle,
            marker=style.marker,
            markevery=max(len(steps) // 5, 1) if steps else 1,
            linewidth=1.3,
            alpha=0.65,
            markersize=4,
            label=f"{style.label} (n={len(runs)})" if i == 0 else None,
        )
    if len(runs) > 1:
        step_sets = [{r.step for r in run.records} for run in runs]
        common = sorted(set.intersection(*step_sets))
        if common:
            import numpy as np

            by_run = [{r.step: getattr(r, metric) for r in run.records} for run in runs]
            arr = np.array([[by_step[s] for s in common] for by_step in by_run])
            ax.fill_between(
                common, arr.min(axis=0), arr.max(axis=0), color=style.color, alpha=0.15, linewidth=0
            )


def _draw_verdict_strip(ax: Axes, verdict: Verdict, *, epsilon: float | None) -> None:
    """The prereg's paired estimator, drawn — a readout strip, never a gate."""
    ax.axvline(0.0, color=OKABE_ITO["black"], linewidth=1.0, linestyle="-")
    if verdict.ci_low is not None and verdict.ci_high is not None:
        ax.errorbar(
            [verdict.mean_d],
            [0.0],
            xerr=[[verdict.mean_d - verdict.ci_low], [verdict.ci_high - verdict.mean_d]],
            fmt="o",
            color=OKABE_ITO["blue"],
            ecolor=OKABE_ITO["blue"],
            capsize=6,
            markersize=7,
        )
        readout = (
            f"mean(d)={verdict.mean_d:.4f}  n={verdict.n}  "
            f"CI=[{verdict.ci_low:.4f}, {verdict.ci_high:.4f}]"
        )
    else:
        ax.plot([verdict.mean_d], [0.0], marker="o", color=OKABE_ITO["blue"], markersize=7)
        readout = (
            f"mean(d)={verdict.mean_d:.4f}  n={verdict.n}  "
            f"CI not preregistered for n={verdict.n} (df={max(verdict.n - 1, 0)})"
        )
    if epsilon is not None:
        ax.axvline(epsilon, color=OKABE_ITO["reddish_purple"], linestyle=":", linewidth=1.5)
        readout += f"  ε={epsilon:g} (practical-relevance threshold, reported not preregistered)"
    stage_text = f"Stage 1: n=5, t_0.9875,4={T_0_9875[4]}   Stage 2: n=9, t_0.9875,8={T_0_9875[8]}"
    if verdict.stage is not None:
        stage_text += f"   [this readout: Stage {verdict.stage}]"
    ax.text(0.01, 0.72, readout, transform=ax.transAxes, fontsize=9)
    ax.text(0.01, 0.28, stage_text, transform=ax.transAxes, fontsize=8, color="dimgray")
    ax.set_yticks([])
    ax.set_xlabel("d = nats/byte(control) − nats/byte(fisher)  (H1: mean(d) > 0)")
    ax.set_title("E1 verdict strip (docs/e1-preregistration.md estimator)")


def fig_e1_paired_loss_curves(
    fisher_runs: Sequence[EvalSeriesRun],
    control_runs: Sequence[EvalSeriesRun],
    verdict: Verdict,
    *,
    metric: str = "nats_per_byte",
    epsilon: float | None = None,
) -> Figure:
    """E1 paired fisher-vs-control loss curves + shaded seed band + verdict strip.

    ``epsilon``, if given, is reported alongside the CI exactly as the
    prereg's Verdict-rule section describes ("no practically relevant
    improvement... ε reported alongside") — it is NOT itself preregistered as
    a number, and the strip labels it that way.
    """
    if not fisher_runs or not control_runs:
        msg = "fig_e1_paired_loss_curves: need at least one fisher run and one control run"
        raise ValueError(msg)
    fig = _figure((9.5, 6.5))
    curve_ax, strip_ax = fig.subplots(nrows=2, ncols=1, height_ratios=[3, 1])
    _plot_seed_band(curve_ax, fisher_runs, fisher_style(), metric=metric)
    _plot_seed_band(curve_ax, control_runs, control_style(), metric=metric)
    curve_ax.set_xlabel("step")
    curve_ax.set_ylabel(metric)
    curve_ax.set_title("E1: fisher vs control — paired loss curves (shaded: seed min-max)")
    curve_ax.legend(loc="upper right", fontsize=8)
    _draw_verdict_strip(strip_ax, verdict, epsilon=epsilon)
    return fig


def fig_e1_log_steps_washout(
    fisher_runs: Sequence[EvalSeriesRun],
    control_runs: Sequence[EvalSeriesRun],
    *,
    metric: str = "nats_per_byte",
) -> Figure:
    """Same E1 curves on a log-step x-axis — does the effect wash out with scale?"""
    if not fisher_runs or not control_runs:
        msg = "fig_e1_log_steps_washout: need at least one fisher run and one control run"
        raise ValueError(msg)
    # A log x-axis cannot place step <= 0, and matplotlib never says so:
    # MEASURED 2026-08-18 (Agg) — plotting x=[0, 10, 20, 30] then
    # set_xscale("log") raised nothing and warned nothing; the step-0 vertex
    # transformed to display x ≈ -9.5e5, outside the axes clip box (it is not
    # a NaN mask), and autoscale excluded it entirely, xlim coming back
    # (9.47, 31.69). The point is simply absent from the rendered figure —
    # exactly the "computed but hidden" failure this module's docstring
    # forbids — so refuse the series instead of drawing a lie.
    for run in (*fisher_runs, *control_runs):
        bad = [record.step for record in run.records if record.step <= 0]
        if bad:
            msg = (
                f"fig_e1_log_steps_washout: {run.label} has non-positive step(s) {bad} — "
                "a log x-axis cannot draw them"
            )
            raise ValueError(msg)
    fig = _figure((9, 5.5))
    ax = fig.add_subplot(111)
    _plot_seed_band(ax, fisher_runs, fisher_style(), metric=metric)
    _plot_seed_band(ax, control_runs, control_style(), metric=metric)
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel(metric)
    ax.set_title("E1: log-step washout view")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def _rolling_mean(values: Sequence[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    out: list[float] = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def fig_e2_spike_variance_texture(
    runs: Sequence[TrainSeriesRun], *, metric: str = "loss"
) -> Figure:
    """E2 per-step train-loss noise texture: raw (thin) + rolling mean (bold),
    one Okabe-Ito color per run so seeds/configs stay distinguishable."""
    if not runs:
        msg = "fig_e2_spike_variance_texture: no runs given"
        raise ValueError(msg)
    fig = _figure((9.5, 5.5))
    ax = fig.add_subplot(111)
    for i, run in enumerate(runs):
        style = head_style(i, label=run.label)
        steps, values = _series_xy(run, metric)
        window = max(min(5, len(values) // 3), 1)
        ax.plot(
            steps,
            values,
            color=style.color,
            linestyle=style.linestyle,
            linewidth=1.0,
            alpha=0.55,
            label=f"{run.label} (raw)",
        )
        ax.plot(
            steps,
            _rolling_mean(values, window),
            color=style.color,
            linestyle="-",
            linewidth=2.5,
            label=f"{run.label} (rolling mean, w={window})",
        )
    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.set_title("E2: per-step train loss — spike/variance texture")
    ax.legend(loc="upper right", fontsize=7)
    return fig


def fig_e3_loss_separation_vs_e2(
    e2_run: EvalSeriesRun, e3_run: EvalSeriesRun, *, metric: str = "nats_per_byte"
) -> Figure:
    """E3 loss curve vs its E2 (κ=0) baseline, plus the per-step gap panel."""
    fig = _figure((9.5, 6.5))
    curve_ax, gap_ax = fig.subplots(nrows=2, ncols=1, height_ratios=[3, 1])
    e2s, e3s = e2_style(), e3_style()
    steps_e2, ys_e2 = _series_xy(e2_run, metric)
    steps_e3, ys_e3 = _series_xy(e3_run, metric)
    curve_ax.plot(
        steps_e2,
        ys_e2,
        color=e2s.color,
        linestyle=e2s.linestyle,
        marker=e2s.marker,
        label=e2s.label,
    )
    curve_ax.plot(
        steps_e3,
        ys_e3,
        color=e3s.color,
        linestyle=e3s.linestyle,
        marker=e3s.marker,
        label=e3s.label,
    )
    curve_ax.set_ylabel(metric)
    curve_ax.set_title("E3 vs E2: loss separation")
    curve_ax.legend(fontsize=8)

    common = sorted(set(steps_e2) & set(steps_e3))
    if not common:
        msg = "fig_e3_loss_separation_vs_e2: e2 and e3 series share no common eval step"
        raise ValueError(msg)
    by_e2 = dict(zip(steps_e2, ys_e2, strict=True))
    by_e3 = dict(zip(steps_e3, ys_e3, strict=True))
    gap = [by_e2[s] - by_e3[s] for s in common]
    gap_ax.plot(
        common, gap, color=OKABE_ITO["bluish_green"], linestyle="-", marker="D", label="e2 − e3 gap"
    )
    gap_ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=1.0, linestyle=":")
    gap_ax.set_xlabel("step")
    gap_ax.set_ylabel(f"Δ {metric}")
    gap_ax.legend(fontsize=8)
    return fig


def _first_kappa_rows(run: EvalSeriesRun) -> list[list[float]]:
    for record in run.records:
        if record.kappa is not None:
            return record.kappa
    msg = f"{run.label}: no kappa telemetry anywhere in this eval series"
    raise ValueError(msg)


def fig_e3_kappa_trajectories(run: EvalSeriesRun) -> Figure:
    """Per-head κ trajectories, one subplot per layer, with clamp bounds + zero line drawn."""
    template = _first_kappa_rows(run)
    n_layers = len(template)
    fig = _figure((9.5, max(3.0 * n_layers, 3.0)))
    axes_obj = fig.subplots(nrows=n_layers, ncols=1, sharex=True)
    axes: list[Axes] = list(axes_obj) if n_layers > 1 else [axes_obj]
    kappa_records = [r for r in run.records if r.kappa is not None]
    kappa_steps = [r.step for r in kappa_records]

    for layer_idx, ax in enumerate(axes):
        n_heads = len(template[layer_idx])
        for head_idx in range(n_heads):
            style = head_style(head_idx)
            ys = [r.kappa[layer_idx][head_idx] for r in kappa_records]  # type: ignore[index]
            ax.plot(
                kappa_steps,
                ys,
                color=style.color,
                linestyle=style.linestyle,
                marker=style.marker,
                markevery=max(len(kappa_steps) // 5, 1),
                markersize=4,
                label=style.label,
            )
        ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=1.0, linestyle="-")
        ax.axhline(KAPPA_MIN, color=OKABE_ITO["black"], linewidth=1.0, linestyle=":")
        ax.axhline(KAPPA_MAX, color=OKABE_ITO["black"], linewidth=1.0, linestyle=":")
        ax.set_ylabel(f"layer {layer_idx} κ")
        ax.legend(fontsize=7, ncol=min(n_heads, 4), loc="upper left")
    axes[-1].set_xlabel("step")
    fig.suptitle(f"{run.label}: per-head κ trajectories (clamp [{KAPPA_MIN}, {KAPPA_MAX}])")
    return fig


def fig_e3_kappa_divergence_vs_loss_gap(
    e3_run: EvalSeriesRun, e2_run: EvalSeriesRun, *, metric: str = "nats_per_byte"
) -> Figure:
    """Twin-axis: E3's max within-layer head-κ spread vs the E2−E3 loss gap,
    aligned on shared eval steps — does divergence timing track separation?"""
    kappa_by_step: dict[int, list[list[float]]] = {
        r.step: r.kappa for r in e3_run.records if r.kappa is not None
    }
    if not kappa_by_step:
        msg = f"{e3_run.label}: no kappa telemetry anywhere in this eval series"
        raise ValueError(msg)
    steps_e2, ys_e2 = _series_xy(e2_run, metric)
    steps_e3, ys_e3 = _series_xy(e3_run, metric)
    by_e2 = dict(zip(steps_e2, ys_e2, strict=True))
    by_e3 = dict(zip(steps_e3, ys_e3, strict=True))
    common = sorted(set(kappa_by_step) & set(by_e2) & set(by_e3))
    if not common:
        msg = (
            f"fig_e3_kappa_divergence_vs_loss_gap: {e3_run.label} kappa-bearing steps share no "
            "common eval step with e2/e3 loss series"
        )
        raise ValueError(msg)

    fig = _figure((9.5, 5.5))
    kappa_ax = fig.add_subplot(111)
    max_spreads = [max_kappa_spread(kappa_by_step[s]) for s in common]
    gap = [by_e2[s] - by_e3[s] for s in common]

    kappa_ax.plot(
        common,
        max_spreads,
        color=OKABE_ITO["orange"],
        linestyle="-",
        marker="^",
        label="max within-layer head κ spread (E3)",
    )
    kappa_ax.set_xlabel("step")
    kappa_ax.set_ylabel("κ spread", color=OKABE_ITO["orange"])
    kappa_ax.tick_params(axis="y", labelcolor=OKABE_ITO["orange"])

    gap_ax = kappa_ax.twinx()
    gap_ax.plot(
        common,
        gap,
        color=OKABE_ITO["blue"],
        linestyle="--",
        marker="s",
        label=f"e2 − e3 Δ{metric} (loss gap)",
    )
    gap_ax.set_ylabel(f"Δ {metric}", color=OKABE_ITO["blue"])
    gap_ax.tick_params(axis="y", labelcolor=OKABE_ITO["blue"])
    gap_ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=1.0, linestyle=":")

    # ``get_lines()`` returns EVERY Line2D on the axes, including the
    # unlabeled ``gap_ax.axhline(0.0, ...)`` zero-reference line above —
    # matplotlib auto-names an unlabeled line with a leading underscore
    # (e.g. ``'_child1'``) specifically to signal "exclude me from a
    # legend"; collecting every line unconditionally let that auto-name
    # reach the rendered legend as a real, garbage entry (MEASURED
    # regression, see tests/test_analysis_charts.py::
    # test_e3_kappa_divergence_vs_loss_gap_legend_excludes_the_unlabeled_zero_line).
    all_lines = [*kappa_ax.get_lines(), *gap_ax.get_lines()]
    labeled = [line for line in all_lines if not str(line.get_label()).startswith("_")]
    kappa_ax.legend(
        labeled, [str(line.get_label()) for line in labeled], loc="upper left", fontsize=7
    )
    kappa_ax.set_title(f"{e3_run.label}: κ-divergence timing vs {e2_run.label} loss-gap timing")
    return fig
