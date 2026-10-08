"""Pure immutable values shared by D1 evaluation producers and evidence readers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from entropic_substrate.kernel.codec.strict_json import exact_keys

TARGET_PREDICTED_EXACT_MATCH_SEMANTIC: Final = "elementwise_fp32_equality_v1"


@dataclass(frozen=True, slots=True)
class D1CosineGap:
    """Undefined full-population cosine, retaining the measured zero-vector count."""

    affected_positions: int
    total_positions: int

    def __post_init__(self) -> None:
        if (
            type(self.affected_positions) is not int
            or type(self.total_positions) is not int
            or self.total_positions < 2
            or not 1 <= self.affected_positions <= self.total_positions
        ):
            raise ValueError("D1 cosine gap positions must satisfy 1 <= affected <= total >= 2")

    @property
    def reason(self) -> str:
        return "undefined_zero_norm"

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": "d1-cosine-gap-v1",
            "reason": self.reason,
            "affected_positions": self.affected_positions,
            "total_positions": self.total_positions,
        }


def parse_d1_cosine(value: object) -> float | D1CosineGap:
    """Dispatch historical scalar evidence or the exact versioned diagnostic gap."""
    if type(value) is float:
        if not math.isfinite(value) or not 0.0 <= value <= 2.0:
            raise ValueError("D1 cosine must be finite and in [0, 2]")
        return value
    if not isinstance(value, Mapping) or any(
        exact_keys(
            value,
            frozenset({"schema_version", "reason", "affected_positions", "total_positions"}),
        )
    ):
        raise ValueError("D1 cosine gap requires exact versioned fields")
    if value["schema_version"] != "d1-cosine-gap-v1" or value["reason"] != "undefined_zero_norm":
        raise ValueError("D1 cosine gap schema or reason is unknown")
    affected, total = value["affected_positions"], value["total_positions"]
    if type(affected) is not int or type(total) is not int:
        raise ValueError("D1 cosine gap positions must be integers")
    return D1CosineGap(affected, total)


def validate_d1_cosine(value: float | D1CosineGap, *, positions: int) -> None:
    """Bind a measured scalar/gap to its unchanged full evaluation population."""
    if type(value) is D1CosineGap:
        if value.total_positions != positions:
            raise ValueError("D1 cosine gap total positions differ from online positions")
    else:
        parse_d1_cosine(value)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-6)


@dataclass(frozen=True, slots=True)
class D1ChannelMoments:
    """Direct held-out moments for one latent channel, with no fitted estimator."""

    mean: tuple[float, ...]
    variance: tuple[float, ...]
    covariance_eigenvalues: tuple[float, ...]
    effective_rank: float
    numerical_rank: int
    rms_l2_norm: float

    def __post_init__(self) -> None:
        width = len(self.mean)
        if width < 1 or len(self.variance) != width or len(self.covariance_eigenvalues) != width:
            raise ValueError("D1 channel moment vectors must share one positive width")
        vectors = (self.mean, self.variance, self.covariance_eigenvalues)
        if any(type(vector) is not tuple for vector in vectors):
            raise TypeError("D1 channel moment vectors must be immutable tuples")
        if any(
            type(value) is not float or not math.isfinite(value)
            for vector in vectors
            for value in vector
        ):
            raise ValueError("D1 channel moments must contain only finite floats")
        if any(value < 0.0 for value in (*self.variance, *self.covariance_eigenvalues)):
            raise ValueError("D1 channel variance and eigenvalues must be nonnegative")
        if tuple(sorted(self.covariance_eigenvalues)) != self.covariance_eigenvalues:
            raise ValueError("D1 channel covariance eigenvalues must be ascending")
        if not _close(sum(self.variance), sum(self.covariance_eigenvalues)):
            raise ValueError("D1 channel variance trace differs from its covariance spectrum")
        mass = sum(self.covariance_eigenvalues)
        expected_effective_rank = 0.0
        if mass > 0.0:
            probabilities = tuple(
                value / mass for value in self.covariance_eigenvalues if value > 0
            )
            expected_effective_rank = math.exp(
                -sum(value * math.log(value) for value in probabilities)
            )
        if (
            type(self.effective_rank) is not float
            or not math.isfinite(self.effective_rank)
            or not 0.0 <= self.effective_rank <= float(width)
        ):
            raise ValueError("D1 channel effective rank is outside the channel width")
        if not _close(self.effective_rank, expected_effective_rank):
            raise ValueError("D1 channel effective rank differs from its covariance spectrum")
        tolerance = max(self.covariance_eigenvalues, default=0.0) * width * 2**-23
        expected_numerical_rank = sum(value > tolerance for value in self.covariance_eigenvalues)
        if type(self.numerical_rank) is not int or self.numerical_rank != expected_numerical_rank:
            raise ValueError("D1 channel numerical rank is outside the channel width")
        if (
            type(self.rms_l2_norm) is not float
            or not math.isfinite(self.rms_l2_norm)
            or self.rms_l2_norm < 0.0
        ):
            raise ValueError("D1 channel RMS L2 norm must be finite and nonnegative")

    @property
    def width(self) -> int:
        return len(self.mean)


class D1StateMomentAccessors:
    """State-channel views shared by every evaluation record carrying ``state_moments``.

    Slotted and fieldless: the annotation reserves no storage, so a frozen slotted
    dataclass inheriting it declares ``state_moments`` as its own field.
    """

    __slots__ = ()

    state_moments: D1ChannelMoments

    @property
    def state_mean(self) -> tuple[float, ...]:
        return self.state_moments.mean

    @property
    def state_variance(self) -> tuple[float, ...]:
        return self.state_moments.variance

    @property
    def state_covariance_eigenvalues(self) -> tuple[float, ...]:
        return self.state_moments.covariance_eigenvalues

    @property
    def effective_rank(self) -> float:
        return self.state_moments.effective_rank


__all__ = [
    "TARGET_PREDICTED_EXACT_MATCH_SEMANTIC",
    "D1ChannelMoments",
    "D1StateMomentAccessors",
]
