"""Pure immutable values shared by D1 evaluation producers and evidence readers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

TARGET_PREDICTED_EXACT_MATCH_SEMANTIC: Final = "elementwise_fp32_equality_v1"


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


__all__ = ["TARGET_PREDICTED_EXACT_MATCH_SEMANTIC", "D1ChannelMoments"]
