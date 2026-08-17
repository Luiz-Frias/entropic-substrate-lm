"""The six interpretation charts (matplotlib), built entirely from
:mod:`entropic_substrate.analysis.series` objects — no JSONL access here at
all, so these tests construct series objects directly rather than writing
files (that boundary is pinned separately in tests/test_analysis_series.py).

Colorblind rule under test throughout: Okabe-Ito hex + a redundant non-color
channel (linestyle/marker) on every semantic distinction, and thresholds/CIs
drawn ON the figure wherever a verdict is implied — never computed off-figure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.analysis.charts import (
    fig_e1_log_steps_washout,
    fig_e1_paired_loss_curves,
    fig_e2_spike_variance_texture,
    fig_e3_kappa_divergence_vs_loss_gap,
    fig_e3_kappa_trajectories,
    fig_e3_loss_separation_vs_e2,
)
from entropic_substrate.analysis.palette import OKABE_ITO
from entropic_substrate.analysis.series import (
    EvalSeriesRun,
    PairedFinalMetric,
    TrainSeriesRun,
    paired_verdict,
)
from entropic_substrate.geometry.kappa import KAPPA_MAX, KAPPA_MIN
from entropic_substrate.telemetry import EvalRecord, TrainRecord

pytestmark = pytest.mark.unit


def _eval_run(
    label: str,
    seed: int,
    steps: list[int],
    npbs: list[float],
    kappa_by_step: dict[int, list[list[float]]] | None = None,
) -> EvalSeriesRun:
    kappa_by_step = kappa_by_step or {}
    records = tuple(
        EvalRecord(step=s, loss=npb * 2, nats_per_byte=npb, kappa=kappa_by_step.get(s))
        for s, npb in zip(steps, npbs, strict=True)
    )
    return EvalSeriesRun(label=label, seed=seed, path=Path(f"{label}.jsonl"), records=records)


def _train_run(label: str, seed: int, steps: list[int], losses: list[float]) -> TrainSeriesRun:
    records = tuple(
        TrainRecord(step=s, loss=loss, nats_per_byte=loss / 2)
        for s, loss in zip(steps, losses, strict=True)
    )
    return TrainSeriesRun(label=label, seed=seed, path=Path(f"{label}.jsonl"), records=records)


def _fisher_control_runs() -> tuple[list[EvalSeriesRun], list[EvalSeriesRun]]:
    fisher = [
        _eval_run(f"e1-fisher-s{s}", s, [10, 20, 30], [2.0 - 0.01 * s, 1.6, 1.3])
        for s in range(1, 4)
    ]
    control = [
        _eval_run(f"e1-random-s{s}", s, [10, 20, 30], [2.1 - 0.01 * s, 1.7, 1.4])
        for s in range(1, 4)
    ]
    return fisher, control


def _lines(ax):
    return list(ax.get_lines())


def _all_text(fig) -> str:
    chunks: list[str] = []
    for ax in fig.get_axes():
        chunks.extend(t.get_text() for t in ax.texts)
        title = ax.get_title()
        if title:
            chunks.append(title)
    return " ".join(chunks)


# ------------------------------------------------------------ chart 1: E1 paired
def test_e1_paired_loss_curves_uses_okabe_ito_and_distinct_linestyles() -> None:
    fisher, control = _fisher_control_runs()
    pairs = [PairedFinalMetric(seed=s, fisher=1.3, control=1.4 + 0.01 * s) for s in range(1, 4)]
    verdict = paired_verdict(pairs)

    fig = fig_e1_paired_loss_curves(fisher, control, verdict)

    assert len(fig.get_axes()) >= 2
    curve_ax = fig.get_axes()[0]
    colors = {line.get_color() for line in _lines(curve_ax)}
    assert OKABE_ITO["blue"] in colors
    assert OKABE_ITO["vermillion"] in colors
    fisher_lines = [ln for ln in _lines(curve_ax) if ln.get_color() == OKABE_ITO["blue"]]
    control_lines = [ln for ln in _lines(curve_ax) if ln.get_color() == OKABE_ITO["vermillion"]]
    assert fisher_lines and control_lines
    assert {ln.get_linestyle() for ln in fisher_lines} != {
        ln.get_linestyle() for ln in control_lines
    }


def test_e1_verdict_strip_labels_both_stage_thresholds_and_the_ci() -> None:
    fisher, control = _fisher_control_runs()
    pairs = [
        PairedFinalMetric(seed=s, fisher=1.5, control=1.5 + d)
        for s, d in zip(range(1, 6), [0.10, 0.12, 0.08, 0.11, 0.09], strict=True)
    ]
    verdict = paired_verdict(pairs)
    assert verdict.stage == 1

    fig = fig_e1_paired_loss_curves(fisher, control, verdict, epsilon=0.02)
    text = _all_text(fig)

    assert "Stage 1" in text
    assert "3.4954" in text
    assert "Stage 2" in text
    assert "2.7515" in text
    assert f"{verdict.mean_d:.4f}" in text or f"{verdict.mean_d:.3f}" in text
    assert "0.02" in text  # epsilon reported alongside, per the prereg


def test_e1_verdict_strip_on_a_non_preregistered_n_says_so_rather_than_a_number() -> None:
    fisher, control = _fisher_control_runs()
    pairs = [PairedFinalMetric(seed=s, fisher=1.5, control=1.6) for s in range(1, 4)]  # n=3
    verdict = paired_verdict(pairs)
    assert verdict.t_value is None

    fig = fig_e1_paired_loss_curves(fisher, control, verdict)
    text = _all_text(fig).lower()
    assert "not preregistered" in text or "non-preregistered" in text


# --------------------------------------------------------- chart 2: washout
def test_e1_log_steps_washout_uses_a_log_x_axis() -> None:
    fisher, control = _fisher_control_runs()
    fig = fig_e1_log_steps_washout(fisher, control)
    ax = fig.get_axes()[0]
    assert ax.get_xscale() == "log"
    colors = {line.get_color() for line in _lines(ax)}
    assert OKABE_ITO["blue"] in colors
    assert OKABE_ITO["vermillion"] in colors


# ---------------------------------------------------- chart 3: E2 spike texture
def test_e2_spike_variance_texture_plots_each_run_distinctly() -> None:
    runs = [
        _train_run(
            "e2-s1", 1, list(range(0, 50, 5)), [3.0, 2.8, 3.1, 2.6, 2.9, 2.5, 2.7, 2.4, 2.6, 2.3]
        ),
        _train_run(
            "e2-s2", 2, list(range(0, 50, 5)), [3.1, 2.9, 2.7, 2.8, 2.5, 2.6, 2.4, 2.5, 2.3, 2.2]
        ),
    ]
    fig = fig_e2_spike_variance_texture(runs)
    ax = fig.get_axes()[0]
    lines = _lines(ax)
    assert len(lines) >= len(runs)
    colors = {line.get_color() for line in lines}
    assert len(colors) >= len(runs)


def test_e2_spike_variance_texture_refuses_an_empty_run_list() -> None:
    with pytest.raises(ValueError, match="no runs"):
        fig_e2_spike_variance_texture([])


# --------------------------------------------------- chart 4: E3 vs E2 separation
def test_e3_loss_separation_vs_e2_has_a_gap_panel_with_a_zero_line() -> None:
    e2_run = _eval_run("e2", 1, [10, 20, 30], [1.8, 1.6, 1.5])
    e3_run = _eval_run("e3", 1, [10, 20, 30], [1.75, 1.5, 1.3])

    fig = fig_e3_loss_separation_vs_e2(e2_run, e3_run)

    assert len(fig.get_axes()) >= 2
    curve_ax, gap_ax = fig.get_axes()[0], fig.get_axes()[1]
    colors = {line.get_color() for line in _lines(curve_ax)}
    assert OKABE_ITO["blue"] in colors  # e2
    assert OKABE_ITO["orange"] in colors  # e3
    zero_lines = [
        ln for ln in _lines(gap_ax) if ln.get_ydata() is not None and set(ln.get_ydata()) == {0.0}
    ]
    assert zero_lines, "gap panel must draw an explicit zero line"


# ------------------------------------------------------- chart 5: kappa trajectories
def _e3_kappa_run() -> EvalSeriesRun:
    return _eval_run(
        "e3",
        1,
        [10, 20, 30],
        [1.8, 1.5, 1.3],
        kappa_by_step={
            10: [[-0.5, 0.1, 0.9], [0.0, -0.2, 0.3]],
            20: [[-0.7, 0.2, 1.1], [0.1, -0.1, 0.4]],
            30: [[-0.9, 0.3, 1.3], [0.2, 0.0, 0.5]],
        },
    )


def test_e3_kappa_trajectories_one_axis_per_layer_with_clamp_and_zero_lines() -> None:
    fig = fig_e3_kappa_trajectories(_e3_kappa_run())

    axes = fig.get_axes()
    assert len(axes) == 2  # two layers in the fixture

    layer0 = axes[0]
    n_heads = 3
    head_lines = [
        ln for ln in _lines(layer0) if ln.get_label() and ln.get_label().startswith("head")
    ]
    assert len(head_lines) == n_heads
    assert len({ln.get_color() for ln in head_lines}) == n_heads
    assert len({ln.get_linestyle() for ln in head_lines}) >= 1  # redundant channel present

    y_values = {round(y, 6) for ln in _lines(layer0) for y in ln.get_ydata()}
    assert 0.0 in y_values  # zero line
    assert round(KAPPA_MIN, 6) in y_values  # clamp lower bound
    assert round(KAPPA_MAX, 6) in y_values  # clamp upper bound


def test_e3_kappa_trajectories_refuses_a_run_with_no_kappa_telemetry() -> None:
    run = _eval_run("e2-mislabeled", 1, [10, 20], [1.8, 1.5])
    with pytest.raises(ValueError, match="no kappa"):
        fig_e3_kappa_trajectories(run)


# --------------------------------------------- chart 6: divergence vs loss-gap timing
def test_e3_kappa_divergence_vs_loss_gap_uses_a_twin_axis_with_distinct_styles() -> None:
    e3_run = _e3_kappa_run()
    e2_run = _eval_run("e2", 1, [10, 20, 30], [1.9, 1.7, 1.6])

    fig = fig_e3_kappa_divergence_vs_loss_gap(e3_run, e2_run)

    axes = fig.get_axes()
    assert len(axes) == 2  # primary + twinx
    kappa_lines = _lines(axes[0])
    gap_lines = _lines(axes[1])
    assert kappa_lines and gap_lines
    assert kappa_lines[0].get_linestyle() != gap_lines[0].get_linestyle()
    assert kappa_lines[0].get_color() != gap_lines[0].get_color()
