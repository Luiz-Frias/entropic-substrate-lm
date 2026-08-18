"""Chart-ready series loaders — the ONLY analysis-side JSONL access, and it
goes exclusively through :mod:`entropic_substrate.telemetry`. These tests pin
that boundary (a series loader that read JSON itself would defeat the point)
and the preregistered paired estimator (docs/e1-preregistration.md).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from entropic_substrate.analysis.series import (
    load_eval_run,
    load_train_run,
    pair_final_metrics,
    paired_verdict,
)
from entropic_substrate.telemetry import TelemetryGap

pytestmark = pytest.mark.unit


def _write(path: Path, *records: dict[str, Any]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _run_start(seed: int) -> dict[str, Any]:
    return {"event": "run_start", "step": 0, "config": {"train": {"seed": seed}}}


def _eval(step: int, npb: float, **extra: Any) -> dict[str, Any]:
    record = {"event": "eval", "step": step, "loss": npb * 2, "nats_per_byte": npb}
    record.update(extra)
    return record


def _train(step: int, loss: float) -> dict[str, Any]:
    return {"event": "train", "step": step, "loss": loss, "nats_per_byte": loss / 2}


# --------------------------------------------------------------------- loaders
def test_load_eval_run_reads_seed_and_series_via_telemetry(tmp_path: Path) -> None:
    path = _write(tmp_path / "e1-fisher.jsonl", _run_start(3), _eval(10, 1.8), _eval(20, 1.5))
    run = load_eval_run(path, label="fisher")
    assert run.label == "fisher"
    assert run.seed == 3
    assert [r.step for r in run.records] == [10, 20]


def test_load_eval_run_propagates_telemetry_gap(tmp_path: Path) -> None:
    """The invariant this file exists to pin: an inadmissible JSONL raises
    TelemetryGap through the analysis layer, never a silently empty series."""
    path = _write(tmp_path / "e1-fisher.jsonl", _run_start(3), _eval(10, float("nan")))
    with pytest.raises(TelemetryGap, match="non-finite"):
        load_eval_run(path, label="fisher")


def test_load_train_run_reads_seed_and_series(tmp_path: Path) -> None:
    path = _write(tmp_path / "e2.jsonl", _run_start(1), _train(1, 3.0), _train(2, 2.8))
    run = load_train_run(path, label="e2")
    assert run.seed == 1
    assert [r.loss for r in run.records] == pytest.approx([3.0, 2.8])


# ------------------------------------------------------------------- pairing
def test_pair_final_metrics_pairs_by_seed_and_reports_unpaired(tmp_path: Path) -> None:
    fisher_1 = _write(tmp_path / "f1.jsonl", _run_start(1), _eval(100, 1.2))
    fisher_2 = _write(tmp_path / "f2.jsonl", _run_start(2), _eval(100, 1.3))
    control_1 = _write(tmp_path / "c1.jsonl", _run_start(1), _eval(100, 1.4))
    control_3 = _write(tmp_path / "c3.jsonl", _run_start(3), _eval(100, 1.1))  # unpaired

    pairs, excluded = pair_final_metrics([fisher_1, fisher_2], [control_1, control_3])

    assert excluded == [2, 3]  # seed 2 (fisher-only), seed 3 (control-only)
    assert len(pairs) == 1
    assert pairs[0].seed == 1
    assert pairs[0].fisher == pytest.approx(1.2)
    assert pairs[0].control == pytest.approx(1.4)
    assert pairs[0].difference == pytest.approx(1.4 - 1.2)


def test_pair_final_metrics_reads_the_metric_via_telemetry_final_eval_metric(
    tmp_path: Path,
) -> None:
    """Primary metric per docs/e1-preregistration.md: exclusively the LAST
    admissible eval record's value, not a mean or any other reduction."""
    fisher = _write(tmp_path / "f.jsonl", _run_start(5), _eval(10, 2.0), _eval(20, 1.5))
    control = _write(tmp_path / "c.jsonl", _run_start(5), _eval(10, 2.2), _eval(20, 1.9))
    pairs, excluded = pair_final_metrics([fisher], [control])
    assert not excluded
    assert pairs[0].fisher == pytest.approx(1.5)
    assert pairs[0].control == pytest.approx(1.9)


# --------------------------------------------------------------- verdict / CI
def test_paired_verdict_stage1_matches_the_preregistered_t_value() -> None:
    """docs/e1-preregistration.md: t_{0.9875,4} = 3.4954 (stage 1, n=5).

    Asserted against the LITERAL value from the prereg, not
    ``series.T_0_9875[4]`` — comparing a module's constant against itself is
    tautological (self-consistent regardless of correctness) and would not
    catch a drifted/typo'd table entry. The guard fabric's actual catch for
    that drift is tests/test_analysis_charts.py's rendered-figure text
    assertion (the verdict strip draws the literal), but pinning the number
    here too keeps this test meaningful in isolation.
    """
    from entropic_substrate.analysis.series import PairedFinalMetric

    pairs = [
        PairedFinalMetric(seed=s, fisher=1.5, control=1.5 + d)
        for s, d in enumerate([0.10, 0.12, 0.08, 0.11, 0.09], start=1)
    ]
    verdict = paired_verdict(pairs)
    assert verdict.n == 5
    assert verdict.stage == 1
    assert verdict.t_value == pytest.approx(3.4954)
    assert verdict.ci_low is not None
    assert verdict.ci_high is not None
    assert verdict.ci_low < verdict.mean_d < verdict.ci_high


def test_paired_verdict_stage2_matches_the_preregistered_t_value() -> None:
    """t_{0.9875,8} = 2.7515 (stage 2, cumulative n=9) — literal, see above."""
    from entropic_substrate.analysis.series import PairedFinalMetric

    pairs = [PairedFinalMetric(seed=s, fisher=1.5, control=1.6) for s in range(1, 10)]
    verdict = paired_verdict(pairs)
    assert verdict.n == 9
    assert verdict.stage == 2
    assert verdict.t_value == pytest.approx(2.7515)


def test_paired_verdict_on_a_non_preregistered_n_reports_no_ci_rather_than_guess() -> None:
    """n=3 (df=2) is not a preregistered look — no t-quantile is hardcoded for
    it, and computing one would be an unauthorized-CI violation of the
    binding prereg. The verdict must say so, not silently produce a number."""
    from entropic_substrate.analysis.series import PairedFinalMetric

    pairs = [PairedFinalMetric(seed=s, fisher=1.5, control=1.6) for s in range(1, 4)]
    verdict = paired_verdict(pairs)
    assert verdict.n == 3
    assert verdict.stage is None
    assert verdict.t_value is None
    assert verdict.ci_low is None
    assert verdict.ci_high is None


def test_paired_verdict_on_zero_pairs_does_not_crash() -> None:
    verdict = paired_verdict([])
    assert verdict.n == 0
    assert math.isnan(verdict.mean_d)
    assert verdict.ci_low is None
