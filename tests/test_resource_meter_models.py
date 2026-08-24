"""Pure resource-meter models keep estimates distinct from measured counters."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from entropic_substrate.evidence.resource_meter_models import (
    PositionFlopModel,
    PowerSample,
    ResourcePhase,
    continuous_rate_cost,
    integrate_power_samples,
)
from entropic_substrate.evidence.resource_provenance import (
    BillingRoundingPolicy,
    BillingUnit,
    Money,
    QuotedRate,
    RateScope,
    RoundingMode,
)
from entropic_substrate.evidence.resource_work import ReadingMethod, ScalarReading
from entropic_substrate.experiments.d1_primitives import EvidenceClass

pytestmark = pytest.mark.unit


def _rate(
    amount: str = "1.2",
    *,
    unit: BillingUnit = BillingUnit.HOUR,
    scope: RateScope = RateScope.PER_ACCELERATOR,
) -> QuotedRate:
    return QuotedRate(
        amount=Money(Decimal(amount), "USD"),
        billing_unit=unit,
        rounding_policy=BillingRoundingPolicy(1, RoundingMode.CEILING, 1),
        rate_scope=scope,
        source="injected quote fixture",
        quote_id="quote-1",
        observed_at=datetime(2026, 8, 24, tzinfo=UTC),
        evidence_class=EvidenceClass.MEASURED,
    )


def test_position_flop_model_has_canonical_identity_and_phase_separation() -> None:
    model = PositionFlopModel(
        name="d1-operation-count-v1",
        train_online_flops_per_position=11,
        train_target_flops_per_position=7,
        eval_online_flops_per_position=5,
        eval_target_flops_per_position=3,
    )

    assert len(model.identity_digest) == 64
    assert model.estimate(ResourcePhase.TRAIN, online_positions=10, target_positions=4) == 138.0
    assert model.estimate(ResourcePhase.EVAL, online_positions=10, target_positions=4) == 62.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_online_flops_per_position", -1),
        ("train_target_flops_per_position", True),
        ("eval_online_flops_per_position", -1),
    ],
)
def test_flop_model_refuses_ambiguous_coefficients(field: str, value: object) -> None:
    values: dict[str, object] = {
        "name": "d1-operation-count-v1",
        "train_online_flops_per_position": 1,
        "train_target_flops_per_position": 1,
        "eval_online_flops_per_position": 1,
        "eval_target_flops_per_position": 1,
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        PositionFlopModel(**values)  # type: ignore[arg-type]


def test_trapezoidal_energy_integration_is_derived_from_measured_samples() -> None:
    result = integrate_power_samples(
        (
            PowerSample(monotonic_ns=0, watts=100.0),
            PowerSample(monotonic_ns=1_000_000_000, watts=200.0),
            PowerSample(monotonic_ns=3_000_000_000, watts=100.0),
        ),
        max_gap_seconds=2.0,
    )

    assert result.sample_count == 3
    assert type(result.joules) is ScalarReading
    assert result.joules.value == 450.0
    assert result.joules.evidence_class is EvidenceClass.DERIVED_FROM_MEASURED
    assert result.joules.method is ReadingMethod.ENERGY_INTEGRATED


def test_energy_integration_reports_typed_gaps_without_partial_joules() -> None:
    insufficient = integrate_power_samples(
        (PowerSample(monotonic_ns=0, watts=100.0),), max_gap_seconds=2.0
    )
    assert insufficient.sample_count == 1
    assert insufficient.joules.field == "energy_joules"  # type: ignore[union-attr]

    coverage_gap = integrate_power_samples(
        (
            PowerSample(monotonic_ns=0, watts=100.0),
            PowerSample(monotonic_ns=3_000_000_000, watts=100.0),
        ),
        max_gap_seconds=2.0,
    )
    assert coverage_gap.sample_count == 2
    assert coverage_gap.joules.field == "energy_joules"  # type: ignore[union-attr]
    assert "sampling gap" in coverage_gap.joules.reason  # type: ignore[union-attr]


def test_energy_samples_fail_loud_on_nonmonotonic_time_or_invalid_power() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        integrate_power_samples(
            (
                PowerSample(monotonic_ns=1, watts=100.0),
                PowerSample(monotonic_ns=1, watts=100.0),
            ),
            max_gap_seconds=1.0,
        )
    with pytest.raises(ValueError, match="watts"):
        PowerSample(monotonic_ns=0, watts=float("nan"))


@pytest.mark.parametrize(
    ("rate", "seconds", "accelerators", "expected"),
    [
        (_rate(), 1800.0, 2, "1.2"),
        (_rate(scope=RateScope.PER_INSTANCE), 1800.0, 8, "0.6"),
        (_rate("0.02", unit=BillingUnit.MINUTE), 90.0, 1, "0.03"),
        (_rate("0.001", unit=BillingUnit.SECOND), 10.0, 1, "0.01"),
    ],
)
def test_continuous_rate_cost_uses_exact_quote_scope_and_unit(
    rate: QuotedRate, seconds: float, accelerators: int, expected: str
) -> None:
    cost = continuous_rate_cost(rate, trainer_wall_seconds=seconds, accelerator_count=accelerators)
    assert cost.amount == Decimal(expected)
    assert cost.currency == "USD"


def test_rate_cost_refuses_invalid_wall_time_instead_of_emitting_zero_unknown() -> None:
    with pytest.raises(ValueError, match="trainer_wall_seconds"):
        continuous_rate_cost(_rate(), trainer_wall_seconds=-1.0, accelerator_count=1)
