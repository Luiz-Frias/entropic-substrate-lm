"""Pure registered D1 factorial arithmetic over finite extracted telemetry.

This module performs no file I/O and has no capture fallback.  A later adapter
may construct :class:`D1AnalysisCell` values only from the public D1 telemetry
reader.  Each seed block contributes exactly one endpoint to every contrast;
evaluation rows are smoothing observations, never independent replicates.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Final, cast

from entropic_substrate.evidence.d1_contract import (
    D1DecisionContract,
    D1RegisteredContrast,
    D1RegisteredEndpoint,
)
from entropic_substrate.evidence.d1_types import D1ContrastTerm
from entropic_substrate.experiments.d1_primitives import D1Arm
from entropic_substrate.experiments.d1_registered_design import (
    D1RegisteredDesign,
    decision_contract_digest,
)

__all__ = [
    "D1AnalysisCell",
    "D1ContrastEstimate",
    "D1EndpointSummary",
    "D1FactorialAnalysis",
    "D1FactorialEffect",
    "D1MetricPoint",
    "compute_d1_factorial",
]

_EVIDENCE_CLASS: Final = "DERIVED_FROM_MEASURED"
_ALL_ARMS: Final = tuple(D1Arm)
_FACTORIAL_TERMS: Final[dict[str, tuple[tuple[D1Arm, float], ...]]] = {
    "sg_at_ac_off": ((D1Arm.SG, 1.0), (D1Arm.NONE, -1.0)),
    "sg_at_ac_on": ((D1Arm.BOTH, 1.0), (D1Arm.AC, -1.0)),
    "ac_at_sg_off": ((D1Arm.AC, 1.0), (D1Arm.NONE, -1.0)),
    "ac_at_sg_on": ((D1Arm.BOTH, 1.0), (D1Arm.SG, -1.0)),
    "sg_main": (
        (D1Arm.SG, 0.5),
        (D1Arm.BOTH, 0.5),
        (D1Arm.NONE, -0.5),
        (D1Arm.AC, -0.5),
    ),
    "ac_main": (
        (D1Arm.AC, 0.5),
        (D1Arm.BOTH, 0.5),
        (D1Arm.NONE, -0.5),
        (D1Arm.SG, -0.5),
    ),
    "interaction": (
        (D1Arm.NONE, 1.0),
        (D1Arm.SG, -1.0),
        (D1Arm.AC, -1.0),
        (D1Arm.BOTH, 1.0),
    ),
}


@dataclass(frozen=True, slots=True)
class D1MetricPoint:
    step: int
    eval_index: int
    value: float

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("D1 metric point step must be a nonnegative integer")
        if type(self.eval_index) is not int or self.eval_index < 0:
            raise ValueError("D1 metric point eval_index must be a nonnegative integer")
        if type(self.value) is not float or not math.isfinite(self.value):
            raise ValueError("D1 metric point value must be a finite float")


@dataclass(frozen=True, slots=True)
class D1AnalysisCell:
    """Finite metric series for one arm in one paired seed block."""

    arm: D1Arm
    seed: int
    pairing_id: str
    metrics: Mapping[str, tuple[D1MetricPoint, ...]]

    def __post_init__(self) -> None:
        if type(self.arm) is not D1Arm:
            raise TypeError("D1 analysis cell arm must be D1Arm")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("D1 analysis cell seed must be a nonnegative integer")
        if type(self.pairing_id) is not str or not self.pairing_id.strip():
            raise ValueError("D1 analysis cell pairing_id must be nonempty")
        frozen: dict[str, tuple[D1MetricPoint, ...]] = {}
        for metric, points in self.metrics.items():
            if type(metric) is not str or not metric or type(points) is not tuple or not points:
                raise ValueError("D1 analysis cell metrics require nonempty tuple series")
            if any(type(point) is not D1MetricPoint for point in points):
                raise TypeError("D1 metric series must contain D1MetricPoint values")
            if any(
                current.step <= previous.step or current.eval_index <= previous.eval_index
                for previous, current in pairwise(points)
            ):
                raise ValueError("D1 metric series steps and eval indexes must strictly increase")
            frozen[metric] = points
        if not frozen:
            raise ValueError("D1 analysis cell requires at least one metric")
        object.__setattr__(self, "metrics", MappingProxyType(frozen))


@dataclass(frozen=True, slots=True)
class D1EndpointSummary:
    metric: str
    estimator: str
    smoothing_window: int
    values_by_arm: Mapping[D1Arm, tuple[float, ...]]
    evidence_class: str = _EVIDENCE_CLASS


@dataclass(frozen=True, slots=True)
class D1ContrastEstimate:
    contrast_id: str
    paired_estimates: tuple[float, ...]
    estimate: float
    sample_sd: float | None
    standard_error: float | None
    interval_lower: float | None
    interval_upper: float | None
    confidence_level: float
    multiplicity: str
    n: int
    df: int | None
    t_critical: float | None
    evidence_class: str = _EVIDENCE_CLASS


@dataclass(frozen=True, slots=True)
class D1FactorialEffect:
    effect_id: str
    terms: tuple[tuple[D1Arm, float], ...]
    paired_estimates: tuple[float, ...]
    estimate: float
    n: int
    deciding: bool = False
    evidence_class: str = _EVIDENCE_CLASS


@dataclass(frozen=True, slots=True)
class D1FactorialAnalysis:
    endpoint: D1EndpointSummary
    registered_contrasts: Mapping[str, D1ContrastEstimate]
    factorial_effects: Mapping[str, D1FactorialEffect]


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Lentz continued fraction for the incomplete beta function."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / max(abs(d), 1e-300) * (1.0 if d >= 0.0 else -1.0)
    h = d
    for index in range(1, 201):
        twice = 2 * index
        numerator = index * (b - index) * x / ((qam + twice) * (a + twice))
        d = 1.0 + numerator * d
        d = 1.0 / max(abs(d), 1e-300) * (1.0 if d >= 0.0 else -1.0)
        c = 1.0 + numerator / c
        c = math.copysign(max(abs(c), 1e-300), c)
        h *= d * c
        numerator = -(a + index) * (qab + index) * x / ((a + twice) * (qap + twice))
        d = 1.0 + numerator * d
        d = 1.0 / max(abs(d), 1e-300) * (1.0 if d >= 0.0 else -1.0)
        c = 1.0 + numerator / c
        c = math.copysign(max(abs(c), 1e-300), c)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= 3e-14:
            return h
    raise ArithmeticError("D1 Student-t beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, df: int) -> float:
    x = df / (df + value * value)
    tail = 0.5 * _regularized_beta(x, df / 2.0, 0.5)
    return tail if value < 0.0 else 1.0 - tail


def _student_t_quantile(probability: float, df: int) -> float:
    if not 0.5 < probability < 1.0 or df < 1:
        raise ValueError("D1 Student-t quantile requires p in (0.5, 1) and df >= 1")
    lower, upper = 0.0, 1.0
    while _student_t_cdf(upper, df) < probability:
        upper *= 2.0
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if _student_t_cdf(middle, df) < probability:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def _endpoint_value(cell: D1AnalysisCell, endpoint: D1RegisteredEndpoint) -> float:
    if endpoint.estimator != "last_k_mean":
        raise ValueError(f"unsupported registered endpoint estimator {endpoint.estimator!r}")
    try:
        points = cell.metrics[endpoint.metric]
    except KeyError as error:
        raise ValueError(f"D1 analysis cell lacks registered metric {endpoint.metric!r}") from error
    if len(points) < endpoint.smoothing_window:
        raise ValueError("D1 metric series is shorter than the registered smoothing window")
    tail = points[-endpoint.smoothing_window :]
    return math.fsum(point.value for point in tail) / endpoint.smoothing_window


def _contrast_values(
    terms: tuple[D1ContrastTerm, ...] | tuple[tuple[D1Arm, float], ...],
    endpoints: Mapping[tuple[D1Arm, int], float],
    seeds: tuple[int, ...],
) -> tuple[float, ...]:
    normalized: list[tuple[D1Arm, float]] = []
    for term in terms:
        if type(term) is D1ContrastTerm:
            normalized.append((D1Arm(term.arm_id), term.coefficient))
        else:
            normalized.append(cast(tuple[D1Arm, float], term))
    return tuple(
        math.fsum(coefficient * endpoints[(arm, seed)] for arm, coefficient in normalized)
        for seed in seeds
    )


def _estimate(rule: D1RegisteredContrast, values: tuple[float, ...]) -> D1ContrastEstimate:
    n = len(values)
    if rule.n_rule == "EXACT" and rule.registered_n != n:
        raise ValueError(f"D1 contrast {rule.contrast_id!r} registered n differs from {n}")
    estimate = math.fsum(values) / n
    if rule.df_rule == "NOT_APPLICABLE":
        return D1ContrastEstimate(
            rule.contrast_id,
            values,
            estimate,
            None,
            None,
            None,
            None,
            rule.confidence_level,
            rule.multiplicity,
            n,
            None,
            None,
        )
    if n < 2:
        raise ValueError("D1 N_MINUS_ONE interval requires at least two seed blocks")
    df = n - 1
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(n)
    probability = (1.0 + rule.confidence_level) / 2.0
    critical = _student_t_quantile(probability, df)
    half_width = critical * standard_error
    return D1ContrastEstimate(
        rule.contrast_id,
        values,
        estimate,
        sample_sd,
        standard_error,
        estimate - half_width,
        estimate + half_width,
        rule.confidence_level,
        rule.multiplicity,
        n,
        df,
        critical,
    )


def _admit_cells(
    design: D1RegisteredDesign, cells: tuple[D1AnalysisCell, ...]
) -> Mapping[tuple[D1Arm, int], D1AnalysisCell]:
    if type(cells) is not tuple:
        raise TypeError("D1 analysis cells must be an immutable tuple")
    by_cell: dict[tuple[D1Arm, int], D1AnalysisCell] = {}
    for cell in cells:
        if type(cell) is not D1AnalysisCell:
            raise TypeError("D1 analysis cells must contain D1AnalysisCell values")
        key = (cell.arm, cell.seed)
        if key in by_cell:
            raise ValueError(f"D1 analysis contains duplicate cell {key}")
        by_cell[key] = cell
    expected = {(arm, seed) for seed in design.registered_seeds for arm in _ALL_ARMS}
    missing = expected - set(by_cell)
    if missing:
        raise ValueError(f"D1 analysis is missing registered cells: {sorted(missing)}")
    extras = set(by_cell) - expected
    if extras:
        raise ValueError(f"D1 analysis contains unregistered cells: {sorted(extras)}")
    for seed in design.registered_seeds:
        pairing_ids = {by_cell[(arm, seed)].pairing_id for arm in _ALL_ARMS}
        if len(pairing_ids) != 1:
            raise ValueError(f"D1 seed {seed} arms do not share one pairing_id")
    return MappingProxyType(by_cell)


def compute_d1_factorial(
    *,
    design: D1RegisteredDesign,
    decision_contract: D1DecisionContract,
    cells: tuple[D1AnalysisCell, ...],
) -> D1FactorialAnalysis:
    """Compute registered contrasts and report-only 2x2 factorial effects."""
    if type(design) is not D1RegisteredDesign or type(decision_contract) is not D1DecisionContract:
        raise TypeError("D1 factorial analysis requires registered design and decision contract")
    if decision_contract_digest(decision_contract) != design.decision_contract_digest:
        raise ValueError("D1 decision contract differs from the registered design")
    admitted = _admit_cells(design, cells)
    endpoints = tuple(rule.endpoint for rule in decision_contract.contrast_rules.values())
    if any(endpoint != endpoints[0] for endpoint in endpoints[1:]):
        raise ValueError("D1 factorial effects require one shared registered endpoint")
    endpoint = endpoints[0]
    values = {key: _endpoint_value(cell, endpoint) for key, cell in admitted.items()}
    by_arm = MappingProxyType(
        {arm: tuple(values[(arm, seed)] for seed in design.registered_seeds) for arm in _ALL_ARMS}
    )
    contrast_results = {
        contrast_id: _estimate(
            rule,
            _contrast_values(rule.terms, values, design.registered_seeds),
        )
        for contrast_id, rule in decision_contract.contrast_rules.items()
    }
    effects = {
        effect_id: D1FactorialEffect(
            effect_id=effect_id,
            terms=terms,
            paired_estimates=(paired := _contrast_values(terms, values, design.registered_seeds)),
            estimate=math.fsum(paired) / len(paired),
            n=len(paired),
        )
        for effect_id, terms in _FACTORIAL_TERMS.items()
    }
    return D1FactorialAnalysis(
        endpoint=D1EndpointSummary(
            metric=endpoint.metric,
            estimator=endpoint.estimator,
            smoothing_window=endpoint.smoothing_window,
            values_by_arm=by_arm,
        ),
        registered_contrasts=MappingProxyType(contrast_results),
        factorial_effects=MappingProxyType(effects),
    )
