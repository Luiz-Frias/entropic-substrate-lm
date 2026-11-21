"""Seed-block power planning uses declared effects, never pilot means."""

from dataclasses import replace
from math import sqrt
from statistics import variance

import numpy as np
import pytest
from scipy.stats import chi2, nct, t

from entropic_substrate.ladder.analyze.seed_sizing import (
    ContrastHypothesis,
    PairedSeed,
    PilotContrast,
    PilotVarianceUnknown,
    dual_confidence_intervals,
    plan_seed_count,
)

pytestmark = pytest.mark.unit


def pilot(name: str = "gain") -> PilotContrast:
    return PilotContrast(
        name,
        "nats/byte",
        "pilot",
        "enwik8-full-endpoint-v1",
        1000,
        tuple(
            PairedSeed(i, value, 0.0, 1000, 1000)
            for i, value in enumerate((0.01, 0.02, 0.0, 0.03, -0.01, 0.02, 0.01, 0.0))
        ),
    )


def hypothesis(name: str = "gain") -> ContrastHypothesis:
    return ContrastHypothesis(name, "nats/byte", "superiority", 0.01, 0.0, None)


def test_smallest_n_matches_independent_noncentral_t_reference() -> None:
    data = pilot()
    result = plan_seed_count((data,), (hypothesis(),), max_seeds=200)
    var_upper = 7 * variance(row.left - row.right for row in data.pairs) / chi2.ppf(0.05, 7)
    expected = next(
        n
        for n in range(2, 201)
        if nct.sf(t.ppf(0.975, n - 1), n - 1, 0.01 * sqrt(n / var_upper)) >= 0.9
    )
    assert result.required_seeds == expected
    assert result.contrasts[0].variance_upper == pytest.approx(var_upper)
    assert result.contrasts[0].power >= 0.9
    assert result.status == "planned"


def test_pilot_mean_does_not_select_assumed_effect() -> None:
    data = pilot()
    shifted = replace(data, pairs=tuple(replace(row, left=row.left + 0.3) for row in data.pairs))
    a = plan_seed_count((data,), (hypothesis(),), max_seeds=200)
    b = plan_seed_count((shifted,), (hypothesis(),), max_seeds=200)
    assert a.required_seeds == b.required_seeds


def test_full_family_increases_n_and_missing_claim_is_rejected() -> None:
    one = plan_seed_count((pilot(),), (hypothesis(),), max_seeds=500)
    two = plan_seed_count(
        (pilot(), pilot("other")), (hypothesis(), hypothesis("other")), max_seeds=500
    )
    assert two.required_seeds > one.required_seeds
    with pytest.raises(ValueError, match="complete"):
        plan_seed_count((pilot(),), (hypothesis(), hypothesis("other")), max_seeds=500)


def test_noninferiority_and_equivalence_use_declared_margins() -> None:
    ni = ContrastHypothesis("gain", "nats/byte", "noninferiority", 0.0, -0.01, None)
    equiv = ContrastHypothesis("gain", "nats/byte", "equivalence", 0.0, -0.01, 0.01)
    a = plan_seed_count((pilot(),), (ni,), max_seeds=500)
    b = plan_seed_count((pilot(),), (equiv,), max_seeds=500)
    assert b.required_seeds >= a.required_seeds
    assert b.contrasts[0].power_method == "equivalence_union_lower_bound"
    assert b.contrasts[0].power >= 0.9


def test_cap_is_structured_insufficiency_not_a_clipped_recommendation() -> None:
    result = plan_seed_count((pilot(),), (hypothesis(),), max_seeds=3)
    assert result.required_seeds is None
    assert result.status == "cap_insufficient"
    assert result.evaluated_seeds == 3
    assert result.contrasts[0].power < 0.9


def test_zero_pilot_variance_does_not_claim_perfect_power() -> None:
    data = pilot()
    data = replace(data, pairs=tuple(replace(row, left=0.01) for row in data.pairs))
    with pytest.raises(PilotVarianceUnknown) as error:
        plan_seed_count((data,), (hypothesis(),), max_seeds=100)
    assert error.value.contrast_id == "gain"
    assert error.value.code == "zero_empirical_variance"


@pytest.mark.parametrize(
    "change",
    [
        {"run_class": "confirmatory"},
        {"run_class": "smoke"},
        {"regime_id": ""},
        {"units": "accuracy"},
        {"expected_exposure": 0},
        {"pairs": ()},
    ],
)
def test_invalid_pilot_metadata_is_rejected(change: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        plan_seed_count((replace(pilot(), **change),), (hypothesis(),), max_seeds=100)


@pytest.mark.parametrize(
    "bad",
    [
        PairedSeed(0, float("nan"), 0, 1000, 1000),
        PairedSeed(0, 0, float("inf"), 1000, 1000),
        PairedSeed(0, 0, 0, 999, 1000),
        PairedSeed(0, 0, 0, 1000, 999),
    ],
)
def test_nonfinite_or_mismatched_exposure_rejected(bad: PairedSeed) -> None:
    with pytest.raises(ValueError):
        plan_seed_count(
            (replace(pilot(), pairs=(bad, *pilot().pairs[1:])),), (hypothesis(),), max_seeds=100
        )


def test_duplicate_seed_and_incomplete_pilot_rejected() -> None:
    with pytest.raises(ValueError, match="seed"):
        plan_seed_count(
            (replace(pilot(), pairs=(pilot().pairs[1], *pilot().pairs[1:])),),
            (hypothesis(),),
            max_seeds=100,
        )
    with pytest.raises(ValueError, match="pilot"):
        plan_seed_count(
            (replace(pilot(), pairs=pilot().pairs[:2]),), (hypothesis(),), max_seeds=100
        )


@pytest.mark.parametrize(
    "rule",
    [
        replace(hypothesis(), expected_effect=0.0),
        replace(hypothesis(), expected_effect=float("nan")),
        replace(hypothesis(), kind="equivalence"),
        replace(hypothesis(), upper_bound=0.02),
        replace(hypothesis(), kind="noninferiority", lower_bound=0.0),
    ],
)
def test_no_effect_or_incomplete_margins_rejected(rule: ContrastHypothesis) -> None:
    with pytest.raises(ValueError):
        plan_seed_count((pilot(),), (rule,), max_seeds=100)


def test_dual_ci_uses_same_sealed_seed_population_and_df() -> None:
    pairs = pilot().pairs
    report = dual_confidence_intervals(pairs, family_size=3, sealed_data_id="sha256:fixture")
    values = [row.left - row.right for row in pairs]
    se = sqrt(variance(values) / len(values))
    assert report.seed_ids == tuple(row.seed for row in pairs)
    assert report.df == 7
    assert report.primary.lower == pytest.approx(report.mean - t.ppf(1 - 0.05 / 6, 7) * se)
    assert report.secondary.upper == pytest.approx(
        report.mean + t.ppf((1 + 0.99730020393674) / 2, 7) * se
    )
    assert report.secondary.deciding is False
    assert report.sealed_data_id == "sha256:fixture"


def test_dual_ci_rejects_empty_seal_and_single_seed() -> None:
    with pytest.raises(ValueError):
        dual_confidence_intervals(pilot().pairs, family_size=1, sealed_data_id="")
    with pytest.raises(ValueError):
        dual_confidence_intervals(pilot().pairs[:1], family_size=1, sealed_data_id="sealed")


def test_equivalence_bound_against_normal_seed_simulation() -> None:
    rule = ContrastHypothesis("gain", "nats/byte", "equivalence", 0.002, -0.01, 0.01)
    result = plan_seed_count((pilot(),), (rule,), max_seeds=500)
    n = result.required_seeds
    assert n is not None
    sigma = sqrt(result.contrasts[0].variance_upper)
    rng = np.random.default_rng(42)
    # Exact sampling laws of mean and variance under the declared normal model.
    sample_means = rng.normal(rule.expected_effect, sigma / sqrt(n), 100_000)
    sample_sds = sigma * np.sqrt(rng.chisquare(n - 1, 100_000) / (n - 1))
    widths = t.ppf(0.975, n - 1) * sample_sds / sqrt(n)
    accepted = np.mean((sample_means - widths > -0.01) & (sample_means + widths < 0.01))
    assert result.contrasts[0].power <= accepted + 0.005
    assert accepted >= 0.9 - 0.005


@pytest.mark.parametrize(
    "change",
    [
        {"regime_id": "other-regime"},
        {"pairs": tuple(replace(row, seed=row.seed + 8) for row in pilot().pairs)},
    ],
)
def test_family_regime_and_roster_must_match(change: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="family"):
        plan_seed_count(
            (pilot(), replace(pilot("other"), **change)),
            (hypothesis(), hypothesis("other")),
            max_seeds=200,
        )


@pytest.mark.parametrize("cap", [0, 1, True, 2.5])
def test_invalid_caps_rejected(cap: int) -> None:
    with pytest.raises(ValueError, match="max seeds"):
        plan_seed_count((pilot(),), (hypothesis(),), max_seeds=cap)


def test_explicit_two_block_pilot_permitted_but_one_never_is() -> None:
    small = replace(pilot(), pairs=pilot().pairs[:2])
    result = plan_seed_count((small,), (hypothesis(),), max_seeds=3, expected_pilot_blocks=2)
    assert result.status == "cap_insufficient"
    with pytest.raises(ValueError, match="pilot blocks"):
        plan_seed_count((small,), (hypothesis(),), max_seeds=3, expected_pilot_blocks=1)
