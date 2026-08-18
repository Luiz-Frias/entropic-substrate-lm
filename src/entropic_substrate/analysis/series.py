"""Chart-ready series loaders — the ONLY analysis-side JSONL access.

Every function here reads through :mod:`entropic_substrate.telemetry`
(``read_eval_series`` / ``read_train_series`` / ``read_seed`` /
``final_eval_metric``), never a second ad-hoc ``json.loads`` of a results
file. An inadmissible run therefore raises :class:`~entropic_substrate.
telemetry.TelemetryGap` here exactly as it would from a direct telemetry
call — this module adds structure (pairing, the preregistered estimator), it
never relaxes admissibility.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from entropic_substrate.telemetry import (
    EvalRecord,
    TrainRecord,
    final_eval_metric,
    read_eval_series,
    read_seed,
    read_train_series,
)

__all__ = [
    "T_0_9875",
    "EvalSeriesRun",
    "PairedFinalMetric",
    "TrainSeriesRun",
    "Verdict",
    "load_eval_run",
    "load_train_run",
    "pair_final_metrics",
    "paired_verdict",
]


@dataclass(frozen=True, slots=True)
class EvalSeriesRun:
    """One run's admissible eval series, labeled for a chart legend."""

    label: str
    seed: int
    path: Path
    records: tuple[EvalRecord, ...]


@dataclass(frozen=True, slots=True)
class TrainSeriesRun:
    """One run's admissible train series, labeled for a chart legend."""

    label: str
    seed: int
    path: Path
    records: tuple[TrainRecord, ...]


def load_eval_run(path: str | Path, *, label: str) -> EvalSeriesRun:
    """Load one run's eval series + seed, both via ``entropic_substrate.telemetry``."""
    path = Path(path)
    return EvalSeriesRun(
        label=label, seed=read_seed(path), path=path, records=tuple(read_eval_series(path))
    )


def load_train_run(path: str | Path, *, label: str) -> TrainSeriesRun:
    """Load one run's train series + seed, both via ``entropic_substrate.telemetry``."""
    path = Path(path)
    return TrainSeriesRun(
        label=label, seed=read_seed(path), path=path, records=tuple(read_train_series(path))
    )


@dataclass(frozen=True, slots=True)
class PairedFinalMetric:
    """One seed's paired final metric — the E1 primary comparison unit."""

    seed: int
    fisher: float
    control: float

    @property
    def difference(self) -> float:
        """H1's ``d_s = y_random,s − y_fisher,s`` (docs/e1-preregistration.md)."""
        return self.control - self.fisher


def _by_seed(paths: Sequence[str | Path], *, arm: str) -> dict[int, Path]:
    """``{seed: path}`` for one arm — raises on a same-arm duplicate seed
    rather than silently keeping the last path (a plain dict comprehension's
    failure mode: MEASURED regression, see
    tests/test_analysis_series.py::test_pair_final_metrics_rejects_a_duplicate_seed_within_one_arm).
    """
    by_seed: dict[int, Path] = {}
    for raw in paths:
        path = Path(raw)
        seed = read_seed(path)
        if seed in by_seed:
            msg = (
                f"pair_final_metrics: {arm} arm gives seed {seed} twice "
                f"({by_seed[seed]} and {path}) — one run per seed per arm, "
                "or the earlier run is silently lost"
            )
            raise ValueError(msg)
        by_seed[seed] = path
    return by_seed


def pair_final_metrics(
    fisher_paths: Sequence[str | Path],
    control_paths: Sequence[str | Path],
    *,
    metric: str = "nats_per_byte",
) -> tuple[list[PairedFinalMetric], list[int]]:
    """Pair fisher/control runs by seed; a seed present on only one side is
    reported (returned in the second element), never silently dropped — and
    a seed given TWICE within the SAME side raises rather than silently
    keeping only the last path (see :func:`_by_seed`).

    The metric is read EXCLUSIVELY via :func:`entropic_substrate.telemetry.
    final_eval_metric`, matching docs/e1-preregistration.md's Design section
    verbatim ("the last admissible eval record's value"). Digest-based pair
    INVALIDATION (a mismatched ``data_order_digest``) is the confirmatory
    analysis session's job per the prereg's own Integrity chain section —
    this chart-rendering path pairs on seed only.
    """
    fisher_by_seed = _by_seed(fisher_paths, arm="fisher")
    control_by_seed = _by_seed(control_paths, arm="control")
    common = sorted(set(fisher_by_seed) & set(control_by_seed))
    excluded = sorted(set(fisher_by_seed) ^ set(control_by_seed))
    pairs = [
        PairedFinalMetric(
            seed=seed,
            fisher=final_eval_metric(fisher_by_seed[seed], metric),
            control=final_eval_metric(control_by_seed[seed], metric),
        )
        for seed in common
    ]
    return pairs, excluded


# docs/e1-preregistration.md, Estimator: "CI = mean(d) ± t_{0.9875, n−1} ·
# sd(d)/√n — the total 5% alpha is split across the two looks (Bonferroni,
# per-look 2.5%): t_{0.9875,4} ≈ 3.50 (stage 1), t_{0.9875,8} ≈ 2.75 (stage
# 2, cumulative)." Exact 4-decimal Student's-t 0.9875-quantile values keyed
# by df, hardcoded rather than computed — a general t-quantile needs scipy
# for two numbers this repo's own prereg already pins exactly. Any OTHER n is
# not a preregistered look: reporting a CI for it would be an ad-hoc,
# unauthorized estimator wearing the prereg's name, so paired_verdict()
# reports "no CI" instead of computing one.
T_0_9875: dict[int, float] = {4: 3.4954, 8: 2.7515}
_STAGE_BY_DF: dict[int, int] = {4: 1, 8: 2}


@dataclass(frozen=True, slots=True)
class Verdict:
    """The prereg's paired estimator on H1's ``d`` — a readout, never a decision.

    ``stage``/``t_value``/``ci_low``/``ci_high`` are ``None`` when ``n`` does
    not match a preregistered look (df not in {4, 8}); the chart labels this
    explicitly rather than silently omitting the CI.
    """

    n: int
    mean_d: float
    sd_d: float
    stage: int | None
    t_value: float | None
    ci_low: float | None
    ci_high: float | None


def paired_verdict(pairs: Sequence[PairedFinalMetric]) -> Verdict:
    """Apply docs/e1-preregistration.md's estimator to paired seed differences."""
    differences = [p.difference for p in pairs]
    n = len(differences)
    mean_d = statistics.fmean(differences) if differences else float("nan")
    sd_d = statistics.stdev(differences) if n > 1 else 0.0
    df = n - 1
    t_value = T_0_9875.get(df)
    stage = _STAGE_BY_DF.get(df)
    if t_value is None:
        return Verdict(
            n=n, mean_d=mean_d, sd_d=sd_d, stage=stage, t_value=None, ci_low=None, ci_high=None
        )
    half_width = t_value * sd_d / math.sqrt(n)
    return Verdict(
        n=n,
        mean_d=mean_d,
        sd_d=sd_d,
        stage=stage,
        t_value=t_value,
        ci_low=mean_d - half_width,
        ci_high=mean_d + half_width,
    )
