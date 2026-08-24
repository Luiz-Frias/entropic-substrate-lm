"""Pure models used by production D1 resource-meter adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import StrEnum
from itertools import pairwise

from entropic_substrate.evidence.resource_provenance import (
    BillingUnit,
    EvidenceGap,
    Money,
    QuotedRate,
    RateScope,
)
from entropic_substrate.evidence.resource_work import ReadingMethod, ScalarReading
from entropic_substrate.experiments.d1_primitives import EvidenceClass, canonical_digest

__all__ = [
    "EnergyIntegrationResult",
    "PositionFlopModel",
    "PowerSample",
    "ResourcePhase",
    "continuous_rate_cost",
    "integrate_power_samples",
]


class ResourcePhase(StrEnum):
    TRAIN = "train"
    EVAL = "eval"


def _count(value: int, *, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")


def _positive(value: float, *, label: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a finite positive float")


@dataclass(frozen=True, slots=True)
class PositionFlopModel:
    """Registered integer operation-count coefficients, never a measured counter."""

    name: str
    train_online_flops_per_position: int
    train_target_flops_per_position: int
    eval_online_flops_per_position: int
    eval_target_flops_per_position: int
    schema_version: str = "position-flop-model/v1"

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("flop model name must be nonempty")
        if self.schema_version != "position-flop-model/v1":
            raise ValueError("unsupported position FLOP model schema")
        for field in (
            "train_online_flops_per_position",
            "train_target_flops_per_position",
            "eval_online_flops_per_position",
            "eval_target_flops_per_position",
        ):
            _count(getattr(self, field), label=field)
        if self.train_online_flops_per_position + self.train_target_flops_per_position == 0:
            raise ValueError("train FLOP model coefficients cannot both be zero")
        if self.eval_online_flops_per_position + self.eval_target_flops_per_position == 0:
            raise ValueError("eval FLOP model coefficients cannot both be zero")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "eval_online_flops_per_position": self.eval_online_flops_per_position,
            "eval_target_flops_per_position": self.eval_target_flops_per_position,
            "name": self.name,
            "schema_version": self.schema_version,
            "train_online_flops_per_position": self.train_online_flops_per_position,
            "train_target_flops_per_position": self.train_target_flops_per_position,
        }

    @property
    def identity_digest(self) -> str:
        return canonical_digest(self.manifest)

    def estimate(
        self,
        phase: ResourcePhase,
        *,
        online_positions: int,
        target_positions: int,
    ) -> float:
        if type(phase) is not ResourcePhase:
            raise TypeError("FLOP estimate phase must be ResourcePhase")
        _count(online_positions, label="online_positions")
        _count(target_positions, label="target_positions")
        if phase is ResourcePhase.TRAIN:
            online_rate = self.train_online_flops_per_position
            target_rate = self.train_target_flops_per_position
        else:
            online_rate = self.eval_online_flops_per_position
            target_rate = self.eval_target_flops_per_position
        estimate = float(online_positions * online_rate + target_positions * target_rate)
        if not math.isfinite(estimate):
            raise OverflowError("FLOP estimate exceeds the finite telemetry representation")
        return estimate


@dataclass(frozen=True, slots=True)
class PowerSample:
    monotonic_ns: int
    watts: float

    def __post_init__(self) -> None:
        _count(self.monotonic_ns, label="monotonic_ns")
        if type(self.watts) is not float or not math.isfinite(self.watts) or self.watts < 0.0:
            raise ValueError("watts must be a finite nonnegative float")


@dataclass(frozen=True, slots=True)
class EnergyIntegrationResult:
    joules: ScalarReading | EvidenceGap
    sample_count: int


def integrate_power_samples(
    samples: tuple[PowerSample, ...], *, max_gap_seconds: float
) -> EnergyIntegrationResult:
    """Integrate a complete timestamped sample window or return a gap, never a partial sum."""
    if type(samples) is not tuple or any(type(sample) is not PowerSample for sample in samples):
        raise TypeError("power samples must be an immutable tuple of PowerSample values")
    _positive(max_gap_seconds, label="max_gap_seconds")
    for previous, current in pairwise(samples):
        if current.monotonic_ns <= previous.monotonic_ns:
            raise ValueError("power sample timestamps must be strictly increasing")
    if len(samples) < 2:
        return EnergyIntegrationResult(
            EvidenceGap(
                "energy_joules",
                f"energy integration requires at least two samples; observed {len(samples)}",
            ),
            len(samples),
        )
    max_gap_ns = max_gap_seconds * 1_000_000_000
    gaps = [current.monotonic_ns - previous.monotonic_ns for previous, current in pairwise(samples)]
    if any(gap > max_gap_ns for gap in gaps):
        return EnergyIntegrationResult(
            EvidenceGap(
                "energy_joules",
                f"power sampling gap exceeded {max_gap_seconds} seconds",
            ),
            len(samples),
        )
    joules = sum(
        (previous.watts + current.watts)
        * 0.5
        * ((current.monotonic_ns - previous.monotonic_ns) / 1_000_000_000)
        for previous, current in pairwise(samples)
    )
    return EnergyIntegrationResult(
        ScalarReading(
            float(joules),
            EvidenceClass.DERIVED_FROM_MEASURED,
            ReadingMethod.ENERGY_INTEGRATED,
        ),
        len(samples),
    )


def continuous_rate_cost(
    rate: QuotedRate, *, trainer_wall_seconds: float, accelerator_count: int
) -> Money:
    """Apply a measured quote continuously; provider rounding belongs to terminal billing."""
    if type(rate) is not QuotedRate:
        raise TypeError("continuous rate cost requires QuotedRate")
    if (
        type(trainer_wall_seconds) is not float
        or not math.isfinite(trainer_wall_seconds)
        or trainer_wall_seconds < 0.0
    ):
        raise ValueError("trainer_wall_seconds must be a finite nonnegative float")
    if type(accelerator_count) is not int or accelerator_count < 1:
        raise ValueError("accelerator_count must be a positive integer")
    unit_seconds = {
        BillingUnit.SECOND: Decimal(1),
        BillingUnit.MINUTE: Decimal(60),
        BillingUnit.HOUR: Decimal(3600),
    }[rate.billing_unit]
    scope_multiplier = (
        Decimal(accelerator_count) if rate.rate_scope is RateScope.PER_ACCELERATOR else Decimal(1)
    )
    with localcontext() as context:
        context.prec = 50
        amount = (
            rate.amount.amount
            * Decimal(str(trainer_wall_seconds))
            / unit_seconds
            * scope_multiplier
        )
    spelling = "0" if amount.is_zero() else format(amount.normalize(), "f")
    return Money(Decimal(spelling), rate.amount.currency)
