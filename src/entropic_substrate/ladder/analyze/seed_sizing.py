"""Paired-seed planning under independent normal contrast errors.

Pilot variance bounds are marginal one-sided 95% bounds; confirmatory intervals
are simultaneous Bonferroni intervals. Power is conditional on the declared
effect and variance bound, never an empirical guarantee or optional-stop rule.
SciPy belongs in the analysis dependency group, not the training runtime.
"""

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean, variance
from typing import Literal

THREE_SIGMA_CONFIDENCE = 0.99730020393674


@dataclass(frozen=True, slots=True)
class PairedSeed:
    """One endpoint pair; callers construct endpoints from admitted captures."""

    seed: int
    left: float
    right: float
    left_exposure: int
    right_exposure: int


@dataclass(frozen=True, slots=True)
class PilotContrast:
    contrast_id: str
    units: str
    run_class: str
    regime_id: str
    expected_exposure: int
    pairs: tuple[PairedSeed, ...]


@dataclass(frozen=True, slots=True)
class ContrastHypothesis:
    """Left-minus-right orientation; higher effects cross the lower bound.

    Superiority uses an explicit lower bound (usually zero). Noninferiority
    requires a negative lower bound. Equivalence requires both finite bounds.
    The assumed effect must lie strictly inside the desired acceptance region.
    """

    contrast_id: str
    units: str
    kind: Literal["superiority", "noninferiority", "equivalence"]
    expected_effect: float
    lower_bound: float
    upper_bound: float | None


class PilotVarianceUnknown(ValueError):
    code = "zero_empirical_variance"

    def __init__(self, contrast_id: str) -> None:
        self.contrast_id = contrast_id
        super().__init__(f"{contrast_id}: zero pilot variance cannot establish a powered design")


@dataclass(frozen=True, slots=True)
class ContrastPower:
    contrast_id: str
    variance_upper: float
    power: float
    power_method: str


@dataclass(frozen=True, slots=True)
class SeedPlan:
    status: Literal["planned", "cap_insufficient"]
    required_seeds: int | None
    evaluated_seeds: int
    contrasts: tuple[ContrastPower, ...]
    target_power: float = 0.9
    familywise_alpha: float = 0.05
    variance_confidence: float = 0.95
    evidence_class: str = "pilot_based_power_projection"


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    lower: float
    upper: float
    confidence: float
    deciding: bool


@dataclass(frozen=True, slots=True)
class DualConfidenceReport:
    sealed_data_id: str
    seed_ids: tuple[int, ...]
    mean: float
    df: int
    family_size: int
    primary: ConfidenceInterval
    secondary: ConfidenceInterval


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonblank")


def _integer(value: object, minimum: int, label: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _finite(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{label} must be finite")


def _paired_values(pairs: tuple[PairedSeed, ...]) -> tuple[float, ...]:
    if type(pairs) is not tuple or len(pairs) < 2:
        raise ValueError("at least two paired seed blocks are required")
    seen: set[int] = set()
    exposures: set[int] = set()
    values: list[float] = []
    for row in pairs:
        _integer(row.seed, 0, "seed")
        if row.seed in seen:
            raise ValueError("duplicate seed block")
        seen.add(row.seed)
        _finite(row.left, "left endpoint")
        _finite(row.right, "right endpoint")
        _integer(row.left_exposure, 1, "left exposure")
        _integer(row.right_exposure, 1, "right exposure")
        if row.left_exposure != row.right_exposure:
            raise ValueError("paired exposure mismatch")
        exposures.add(row.left_exposure)
        value = row.left - row.right
        _finite(value, "paired difference")
        values.append(value)
    if len(exposures) != 1:
        raise ValueError("seed exposure mismatch")
    return tuple(values)


def validate_hypothesis(rule: ContrastHypothesis) -> None:
    """Validate explicit hypothesis margins without observing pilot outcomes."""
    _text(rule.contrast_id, "contrast ID")
    _text(rule.units, "units")
    _finite(rule.expected_effect, "expected effect")
    _finite(rule.lower_bound, "lower bound")
    if rule.kind not in {"superiority", "noninferiority", "equivalence"}:
        raise ValueError("unknown hypothesis kind")
    if rule.expected_effect <= rule.lower_bound:
        raise ValueError("expected effect must exceed the lower bound")
    if rule.kind == "noninferiority" and rule.lower_bound >= 0:
        raise ValueError("noninferiority requires an explicit negative lower bound")
    if rule.kind == "equivalence":
        if rule.upper_bound is None:
            raise ValueError("equivalence requires an upper bound")
        _finite(rule.upper_bound, "upper bound")
        if rule.expected_effect >= rule.upper_bound:
            raise ValueError("expected effect must be below the equivalence upper bound")
    elif rule.upper_bound is not None:
        raise ValueError("only equivalence accepts an upper bound")


def _power(rule: ContrastHypothesis, var_upper: float, n: int, family_size: int) -> float:
    from scipy.stats import nct, t

    critical = float(t.ppf(1 - 0.05 / (2 * family_size), n - 1))
    scale = sqrt(n / var_upper)
    lower_pass = float(nct.sf(critical, n - 1, (rule.expected_effect - rule.lower_bound) * scale))
    if rule.upper_bound is None:
        result = lower_pass
    else:
        upper_pass = float(
            nct.sf(critical, n - 1, (rule.upper_bound - rule.expected_effect) * scale)
        )
        # Both tails share the same sample. Their product is not valid power.
        result = max(0.0, lower_pass + upper_pass - 1.0)
    if not isfinite(result) or not 0 <= result <= 1:
        raise ArithmeticError("noncentral-t power was nonfinite or outside [0, 1]")
    return result


def plan_seed_count(
    pilots: tuple[PilotContrast, ...],
    hypotheses: tuple[ContrastHypothesis, ...],
    *,
    max_seeds: int,
    expected_pilot_blocks: int = 8,
) -> SeedPlan:
    """Choose smallest n meeting every primary claim, or return cap insufficiency.

    The complete deciding family must be passed together. The caller supplies
    sealed hypothesis definitions and admitted pilot endpoints; metadata here
    validates their consistency but does not authenticate upstream artifacts.
    Equivalence uses a conservative union lower bound, not an independence
    approximation. No zero-variance epsilon or empirical effect substitution.
    """
    from scipy.stats import chi2

    _integer(max_seeds, 2, "max seeds")
    _integer(expected_pilot_blocks, 2, "expected pilot blocks")
    if type(pilots) is not tuple or type(hypotheses) is not tuple or not hypotheses:
        raise ValueError("complete nonempty immutable deciding family required")
    rules = {rule.contrast_id: rule for rule in hypotheses}
    data = {item.contrast_id: item for item in pilots}
    if len(rules) != len(hypotheses) or len(data) != len(pilots) or data.keys() != rules.keys():
        raise ValueError("complete unique deciding family and matching pilots required")
    variances: list[float] = []
    family_identity: set[tuple[str, int, tuple[int, ...]]] = set()
    for rule in hypotheses:
        validate_hypothesis(rule)
        item = data[rule.contrast_id]
        _text(item.regime_id, "regime ID")
        _integer(item.expected_exposure, 1, "expected exposure")
        if item.run_class != "pilot" or item.units != rule.units:
            raise ValueError("excluded pilot run class and matching units required")
        values = _paired_values(item.pairs)
        if len(values) != expected_pilot_blocks:
            raise ValueError("pilot blocks differ from the declared pilot roster size")
        if item.pairs[0].left_exposure != item.expected_exposure:
            raise ValueError("pilot exposure differs from declared endpoint")
        family_identity.add(
            (item.regime_id, item.expected_exposure, tuple(sorted(r.seed for r in item.pairs)))
        )
        empirical = variance(values)
        if empirical == 0:
            raise PilotVarianceUnknown(item.contrast_id)
        upper = (len(values) - 1) * empirical / float(chi2.ppf(0.05, len(values) - 1))
        if not isfinite(upper) or upper <= 0:
            raise ArithmeticError("pilot variance upper bound must be positive finite")
        variances.append(upper)
    if len(family_identity) != 1:
        raise ValueError("pilot family must share regime, endpoint exposure and paired seed roster")
    powers: tuple[ContrastPower, ...] = ()
    for n in range(2, max_seeds + 1):
        powers = tuple(
            ContrastPower(
                rule.contrast_id,
                var_upper,
                _power(rule, var_upper, n, len(hypotheses)),
                "equivalence_union_lower_bound" if rule.kind == "equivalence" else "noncentral_t",
            )
            for rule, var_upper in zip(hypotheses, variances, strict=True)
        )
        if all(item.power >= 0.9 for item in powers):
            return SeedPlan("planned", n, n, powers)
    return SeedPlan("cap_insufficient", None, max_seeds, powers)


def dual_confidence_intervals(
    pairs: tuple[PairedSeed, ...], *, family_size: int, sealed_data_id: str
) -> DualConfidenceReport:
    """Report simultaneous 95% and marginal 3-sigma intervals from one seed set.

    The secondary interval is report-only and has no multiplicity adjustment.
    A zero observed variance is reported literally; it is not a power claim.
    """
    from scipy.stats import t

    _integer(family_size, 1, "family size")
    _text(sealed_data_id, "sealed data ID")
    values = _paired_values(pairs)
    center = mean(values)
    se = sqrt(variance(values) / len(values))
    _finite(se, "standard error")
    intervals: list[ConfidenceInterval] = []
    for confidence, deciding in ((1 - 0.05 / family_size, True), (THREE_SIGMA_CONFIDENCE, False)):
        width = float(t.ppf((1 + confidence) / 2, len(values) - 1)) * se
        _finite(width, "interval half width")
        intervals.append(ConfidenceInterval(center - width, center + width, confidence, deciding))
    return DualConfidenceReport(
        sealed_data_id,
        tuple(row.seed for row in pairs),
        center,
        len(values) - 1,
        family_size,
        intervals[0],
        intervals[1],
    )
