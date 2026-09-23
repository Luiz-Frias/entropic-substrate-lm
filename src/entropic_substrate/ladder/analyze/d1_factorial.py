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
from dataclasses import InitVar, dataclass, field
from itertools import pairwise
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Final, cast

from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.kernel.verdict.d1_contract import (
    D1DecisionContract,
    D1RegisteredContrast,
    D1RegisteredEndpoint,
)
from entropic_substrate.kernel.verdict.d1_evaluation_values import D1CosineGap
from entropic_substrate.kernel.verdict.d1_statistical_gaps import (
    D1EndpointGap,
    D1EstimateGap,
    D1GapCell,
)
from entropic_substrate.kernel.verdict.d1_types import D1ContrastTerm
from entropic_substrate.ladder.register.d1_registered_design import (
    D1RegisteredDesign,
    decision_contract_digest,
)

__all__ = [
    "D1AnalysisCell",
    "D1AppliedExclusion",
    "D1ContrastEstimate",
    "D1EndpointSummary",
    "D1FactorialAnalysis",
    "D1FactorialEffect",
    "D1FactorialExclusionAuthority",
    "D1MetricPoint",
    "compute_d1_factorial",
    "d1_factorial_effect_terms",
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
_EXCLUSION_AUTHORITY_TOKEN: Final = object()


def d1_factorial_effect_terms() -> Mapping[str, tuple[tuple[D1Arm, float], ...]]:
    """Return the canonical ordered report-only 2x2 factorial effects."""
    return MappingProxyType(_FACTORIAL_TERMS)


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
class D1DiagnosticGapPoint:
    """An explicitly undefined diagnostic at a retained evaluation cursor."""

    step: int
    eval_index: int
    gap: D1CosineGap

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("D1 diagnostic gap step must be nonnegative")
        if type(self.eval_index) is not int or self.eval_index < 0:
            raise ValueError("D1 diagnostic gap eval_index must be nonnegative")
        if type(self.gap) is not D1CosineGap:
            raise TypeError("D1 diagnostic gap requires D1CosineGap")


@dataclass(frozen=True, slots=True)
class D1AnalysisCell:
    """Finite metric series for one arm in one paired seed block."""

    arm: D1Arm
    seed: int
    pairing_id: str
    metrics: Mapping[str, tuple[D1MetricPoint, ...]]
    diagnostic_gaps: Mapping[str, tuple[D1DiagnosticGapPoint, ...]] = field(default_factory=dict)

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
        gaps = dict(self.diagnostic_gaps)
        for metric, points in gaps.items():
            if metric != "mean_pairwise_cosine_distance":
                raise ValueError("D1 diagnostic gap must refer to cosine")
            if (
                type(points) is not tuple
                or not points
                or any(type(point) is not D1DiagnosticGapPoint for point in points)
            ):
                raise TypeError("D1 diagnostic gaps require a nonempty tuple of gap points")
            if any(
                current.step <= previous.step or current.eval_index <= previous.eval_index
                for previous, current in pairwise(points)
            ):
                raise ValueError("D1 diagnostic gap cursors must strictly increase")
            known = {(point.step, point.eval_index) for point in frozen.get(metric, ())}
            if known & {(point.step, point.eval_index) for point in points}:
                raise ValueError("D1 metric cursor cannot be both measured and undefined")
            combined = sorted((*frozen.get(metric, ()), *points), key=lambda point: point.step)
            if any(
                current.step <= previous.step or current.eval_index <= previous.eval_index
                for previous, current in pairwise(combined)
            ):
                raise ValueError("D1 combined measured/gap cursors must strictly increase")
        object.__setattr__(self, "diagnostic_gaps", MappingProxyType(gaps))


@dataclass(frozen=True, slots=True)
class D1AppliedExclusion:
    """One evidence-backed whole-seed-block exclusion applied before arithmetic."""

    seed: int
    pairing_id: str
    member_run_ids: tuple[str, ...]
    reason_id: str
    evidence_paths: tuple[str, ...]
    evidence_digest: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("D1 applied exclusion seed must be a nonnegative integer")
        if (
            type(self.pairing_id) is not str
            or not self.pairing_id
            or self.pairing_id.strip() != self.pairing_id
        ):
            raise ValueError("D1 applied exclusion pairing_id must be nonempty and trimmed")
        if (
            type(self.member_run_ids) is not tuple
            or len(self.member_run_ids) != len(D1Arm)
            or len(set(self.member_run_ids)) != len(D1Arm)
            or any(
                type(value) is not str or not value or value.strip() != value
                for value in self.member_run_ids
            )
        ):
            raise ValueError("D1 applied exclusion requires four unique member run IDs")
        if (
            type(self.reason_id) is not str
            or not self.reason_id
            or self.reason_id.strip() != self.reason_id
        ):
            raise ValueError("D1 applied exclusion reason_id must be nonempty and trimmed")
        if type(self.evidence_paths) is not tuple or not self.evidence_paths:
            raise ValueError("D1 applied exclusion requires at least one evidence path")
        if len(set(self.evidence_paths)) != len(self.evidence_paths):
            raise ValueError("D1 applied exclusion evidence paths must be unique")
        for value in self.evidence_paths:
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError("D1 applied exclusion evidence path must be nonempty and trimmed")
            path = PurePosixPath(value)
            if (
                path.is_absolute()
                or "\\" in value
                or path.as_posix() != value
                or ".." in path.parts
            ):
                raise ValueError("D1 applied exclusion evidence path must be repository-relative")
        if (
            type(self.evidence_digest) is not str
            or len(self.evidence_digest) != 64
            or self.evidence_digest != self.evidence_digest.lower()
            or any(value not in "0123456789abcdef" for value in self.evidence_digest)
        ):
            raise ValueError(
                "D1 applied exclusion evidence_digest must be 64 lowercase hexadecimal"
            )

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "evidence_digest": self.evidence_digest,
            "evidence_paths": list(self.evidence_paths),
            "member_run_ids": list(self.member_run_ids),
            "pairing_id": self.pairing_id,
            "reason_id": self.reason_id,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class D1FactorialExclusionAuthority:
    """Admission-owned capability binding exclusions to one capture and design."""

    design_digest: str
    decision_contract_digest: str
    capture_digest: str
    exclusions: tuple[D1AppliedExclusion, ...]
    _attestation: InitVar[object | None] = None

    def __post_init__(self, _attestation: object | None) -> None:
        if _attestation is not _EXCLUSION_AUTHORITY_TOKEN:
            raise TypeError("D1 factorial exclusion authority is factory-only")
        for label, digest in (
            ("design_digest", self.design_digest),
            ("decision_contract_digest", self.decision_contract_digest),
            ("capture_digest", self.capture_digest),
        ):
            if (
                type(digest) is not str
                or len(digest) != 64
                or digest != digest.lower()
                or any(value not in "0123456789abcdef" for value in digest)
            ):
                raise ValueError(f"D1 factorial authority {label} must be 64 lowercase hexadecimal")
        if type(self.exclusions) is not tuple or any(
            type(value) is not D1AppliedExclusion for value in self.exclusions
        ):
            raise TypeError("D1 factorial authority exclusions must be an immutable typed tuple")


def seal_d1_factorial_exclusion_authority(
    *,
    design_digest: str,
    decision_contract_digest: str,
    capture_digest: str,
    exclusions: tuple[D1AppliedExclusion, ...],
) -> D1FactorialExclusionAuthority:
    """Seal exclusions only after an admission boundary has validated them."""
    return D1FactorialExclusionAuthority(
        design_digest=design_digest,
        decision_contract_digest=decision_contract_digest,
        capture_digest=capture_digest,
        exclusions=exclusions,
        _attestation=_EXCLUSION_AUTHORITY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class D1EndpointSummary:
    metric: str
    estimator: str
    smoothing_window: int
    values_by_arm: Mapping[D1Arm, tuple[float | D1EndpointGap, ...]]
    evidence_class: str = _EVIDENCE_CLASS


@dataclass(frozen=True, slots=True)
class D1ContrastEstimate:
    contrast_id: str
    paired_estimates: tuple[float | None, ...]
    estimate: float | None
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
    unavailable: D1EstimateGap | None = None


@dataclass(frozen=True, slots=True)
class D1FactorialEffect:
    effect_id: str
    terms: tuple[tuple[D1Arm, float], ...]
    paired_estimates: tuple[float | None, ...]
    estimate: float | None
    n: int
    deciding: bool = False
    evidence_class: str = _EVIDENCE_CLASS
    unavailable: D1EstimateGap | None = None


@dataclass(frozen=True, slots=True)
class D1FactorialAnalysis:
    included_seeds: tuple[int, ...]
    excluded_seeds: tuple[int, ...]
    cell_commitment: Mapping[tuple[D1Arm, int], D1AnalysisCell] = field(repr=False)
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


def d1_cell_endpoint(cell: D1AnalysisCell, endpoint: D1RegisteredEndpoint) -> float | D1EndpointGap:
    if endpoint.estimator != "last_k_mean":
        raise ValueError(f"unsupported registered endpoint estimator {endpoint.estimator!r}")
    points = sorted(
        (*cell.metrics.get(endpoint.metric, ()), *cell.diagnostic_gaps.get(endpoint.metric, ())),
        key=lambda point: point.step,
    )
    if len(points) < endpoint.smoothing_window:
        raise ValueError("D1 metric series is shorter than the registered smoothing window")
    tail = points[-endpoint.smoothing_window :]
    gaps = tuple(
        (point.step, point.eval_index, point.gap)
        for point in tail
        if type(point) is D1DiagnosticGapPoint
    )
    if gaps:
        return D1EndpointGap(gaps)
    return (
        math.fsum(point.value for point in tail if type(point) is D1MetricPoint)
        / endpoint.smoothing_window
    )


def _contrast_values(
    terms: tuple[D1ContrastTerm, ...] | tuple[tuple[D1Arm, float], ...],
    endpoints: Mapping[tuple[D1Arm, int], float | D1EndpointGap],
    seeds: tuple[int, ...],
) -> tuple[float | None, ...]:
    normalized: list[tuple[D1Arm, float]] = []
    for term in terms:
        if type(term) is D1ContrastTerm:
            normalized.append((D1Arm(term.arm_id), term.coefficient))
        else:
            normalized.append(cast(tuple[D1Arm, float], term))
    return tuple(
        None
        if any(type(endpoints[(arm, seed)]) is D1EndpointGap for arm, _ in normalized)
        else math.fsum(
            coefficient * cast(float, endpoints[(arm, seed)]) for arm, coefficient in normalized
        )
        for seed in seeds
    )


def _missing_endpoints(
    arms: tuple[D1Arm, ...],
    values: Mapping[tuple[D1Arm, int], float | D1EndpointGap],
    seeds: tuple[int, ...],
) -> D1EstimateGap | None:
    cells = tuple(
        D1GapCell(arm.value, seed, value)
        for seed in seeds
        for arm in arms
        if type(value := values[(arm, seed)]) is D1EndpointGap
    )
    return D1EstimateGap(cells) if cells else None


def _registered_estimate(
    rule: D1RegisteredContrast,
    values: Mapping[tuple[D1Arm, int], float | D1EndpointGap],
    seeds: tuple[int, ...],
) -> D1ContrastEstimate:
    pairs = _contrast_values(rule.terms, values, seeds)
    missing = _missing_endpoints(tuple(D1Arm(term.arm_id) for term in rule.terms), values, seeds)
    if missing is None:
        return _estimate(rule, cast(tuple[float, ...], pairs))
    n = len(seeds)
    return D1ContrastEstimate(
        rule.contrast_id,
        pairs,
        None,
        None,
        None,
        None,
        None,
        rule.confidence_level,
        rule.multiplicity,
        n,
        None if rule.df_rule == "NOT_APPLICABLE" else n - 1,
        None,
        unavailable=missing,
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
    design: D1RegisteredDesign,
    cells: tuple[D1AnalysisCell, ...],
    exclusions: tuple[D1AppliedExclusion, ...],
) -> tuple[Mapping[tuple[D1Arm, int], D1AnalysisCell], tuple[int, ...], tuple[int, ...]]:
    if type(cells) is not tuple:
        raise TypeError("D1 analysis cells must be an immutable tuple")
    if type(exclusions) is not tuple:
        raise TypeError("D1 applied exclusions must be an immutable tuple")
    if any(type(item) is not D1AppliedExclusion for item in exclusions):
        raise TypeError("D1 applied exclusions must contain D1AppliedExclusion values")
    if len({item.seed for item in exclusions}) != len(exclusions) or len(
        {item.pairing_id for item in exclusions}
    ) != len(exclusions):
        raise ValueError("D1 seed or pairing block may be excluded at most once")
    by_cell: dict[tuple[D1Arm, int], D1AnalysisCell] = {}
    for cell in cells:
        if type(cell) is not D1AnalysisCell:
            raise TypeError("D1 analysis cells must contain D1AnalysisCell values")
        key = (cell.arm, cell.seed)
        if key in by_cell:
            raise ValueError(f"D1 analysis contains duplicate cell {key}")
        by_cell[key] = cell
    registered = set(design.registered_seeds)
    observed_seeds = {seed for _, seed in by_cell}
    excluded_seeds = {item.seed for item in exclusions}
    unregistered_exclusions = excluded_seeds - registered
    if unregistered_exclusions:
        raise ValueError(
            f"D1 applied exclusion contains unregistered seed: {sorted(unregistered_exclusions)}"
        )
    unregistered_seeds = observed_seeds - registered
    if unregistered_seeds:
        raise ValueError(f"D1 analysis contains unregistered seeds: {sorted(unregistered_seeds)}")
    overlap = observed_seeds & excluded_seeds
    if overlap:
        raise ValueError(f"D1 excluded seed also has observed cells: {sorted(overlap)}")
    unexplained_missing = registered - observed_seeds - excluded_seeds
    if unexplained_missing:
        raise ValueError(
            "D1 missing registered seed block lacks an applied exclusion: "
            f"{sorted(unexplained_missing)}"
        )
    included = tuple(seed for seed in design.registered_seeds if seed not in excluded_seeds)
    if not included:
        raise ValueError("D1 factorial analysis has no finite seed block")
    expected = {(arm, seed) for seed in included for arm in _ALL_ARMS}
    missing = expected - set(by_cell)
    if missing:
        raise ValueError(f"D1 analysis is missing registered cells: {sorted(missing)}")
    extras = set(by_cell) - expected
    if extras:
        raise ValueError(f"D1 analysis contains unregistered cells: {sorted(extras)}")
    for seed in included:
        pairing_ids = {by_cell[(arm, seed)].pairing_id for arm in _ALL_ARMS}
        if len(pairing_ids) != 1:
            raise ValueError(f"D1 seed {seed} arms do not share one pairing_id")
    excluded = tuple(seed for seed in design.registered_seeds if seed in excluded_seeds)
    return MappingProxyType(by_cell), included, excluded


def _validate_sample_rules(contract: D1DecisionContract, n: int) -> None:
    for rule in contract.contrast_rules.values():
        if rule.n_rule == "EXACT" and rule.registered_n != n:
            raise ValueError(f"D1 contrast {rule.contrast_id!r} registered n differs from {n}")
        if rule.df_rule == "N_MINUS_ONE" and n < 2:
            raise ValueError("D1 N_MINUS_ONE interval requires at least two seed blocks")


def compute_d1_factorial(
    *,
    design: D1RegisteredDesign,
    decision_contract: D1DecisionContract,
    cells: tuple[D1AnalysisCell, ...],
    exclusion_authority: D1FactorialExclusionAuthority,
) -> D1FactorialAnalysis:
    """Compute registered contrasts and report-only 2x2 factorial effects."""
    if type(design) is not D1RegisteredDesign or type(decision_contract) is not D1DecisionContract:
        raise TypeError("D1 factorial analysis requires registered design and decision contract")
    if decision_contract_digest(decision_contract) != design.decision_contract_digest:
        raise ValueError("D1 decision contract differs from the registered design")
    if type(exclusion_authority) is not D1FactorialExclusionAuthority:
        raise TypeError("D1 factorial analysis requires an exclusion authority")
    if (
        exclusion_authority.design_digest != design.design_digest
        or exclusion_authority.decision_contract_digest != design.decision_contract_digest
    ):
        raise ValueError("D1 factorial exclusion authority differs from the registered design")
    admitted, included_seeds, excluded_seeds = _admit_cells(
        design, cells, exclusion_authority.exclusions
    )
    _validate_sample_rules(decision_contract, len(included_seeds))
    endpoints = tuple(rule.endpoint for rule in decision_contract.contrast_rules.values())
    if any(endpoint != endpoints[0] for endpoint in endpoints[1:]):
        raise ValueError("D1 factorial effects require one shared registered endpoint")
    endpoint = endpoints[0]
    values = {key: d1_cell_endpoint(cell, endpoint) for key, cell in admitted.items()}
    by_arm = MappingProxyType(
        {arm: tuple(values[(arm, seed)] for seed in included_seeds) for arm in _ALL_ARMS}
    )
    contrast_results = {
        contrast_id: _registered_estimate(rule, values, included_seeds)
        for contrast_id, rule in decision_contract.contrast_rules.items()
    }
    effects = {
        effect_id: D1FactorialEffect(
            effect_id=effect_id,
            terms=terms,
            paired_estimates=(paired := _contrast_values(terms, values, included_seeds)),
            estimate=None
            if any(value is None for value in paired)
            else math.fsum(cast(tuple[float, ...], paired)) / len(paired),
            n=len(paired),
            unavailable=_missing_endpoints(tuple(arm for arm, _ in terms), values, included_seeds),
        )
        for effect_id, terms in _FACTORIAL_TERMS.items()
    }
    return D1FactorialAnalysis(
        included_seeds=included_seeds,
        excluded_seeds=excluded_seeds,
        cell_commitment=admitted,
        endpoint=D1EndpointSummary(
            metric=endpoint.metric,
            estimator=endpoint.estimator,
            smoothing_window=endpoint.smoothing_window,
            values_by_arm=by_arm,
        ),
        registered_contrasts=MappingProxyType(contrast_results),
        factorial_effects=MappingProxyType(effects),
    )
