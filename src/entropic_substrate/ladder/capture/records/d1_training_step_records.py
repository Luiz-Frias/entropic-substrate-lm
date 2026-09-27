"""Record half of the D1 training-step split (torch half:
``__legacy/d1_training_step.py``).

RED-first split (ruling R-parallel; contract C17-p5): the immutable
evidence types plus the aggregate-gradient-norm invariant live here,
math-only, no torch/numpy import, so p8's later move of this half never
drags torch along for the three ``evidence.d1_telemetry_*`` readers that
only need these types. The torch half imports these names and RE-EXPORTS
them under its own literal ``__all__`` for the 11 out-of-envelope importers
still on the old dotted name (ruling C17-p5); p8/p9 rewrite those importers,
never this phase.

``_require_aggregate_gradient_norm`` lives here rather than the torch half
because it is called from BOTH sides of the split -- from
``D1TrainStepResult.__post_init__`` (a record symbol, here) and from
``apply_d1_training_step`` (the torch transaction) -- and the split's
import direction is one-directional: the torch half may import the record
half, never the reverse. Leaving the helper in the torch half would force
this module to import back from it to validate its own constructor,
closing a module-level cycle. It is pure math (``math.isclose``,
``math.hypot``) over a ``D1GradientReachability`` and a ``float``, so it
needs no torch import to live here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "D1ComponentGradientEvidence",
    "D1GradientEvidenceKind",
    "D1GradientReachability",
    "D1TrainStepResult",
]


class D1GradientEvidenceKind(StrEnum):
    """The only admissible interpretations of a D1 component gradient."""

    MEASURED_NONZERO = "measured_nonzero"
    MEASURED_ZERO = "measured_zero"
    STRUCTURALLY_ABSENT_STOP_GRADIENT = "structurally_absent_stop_gradient"


@dataclass(frozen=True, slots=True)
class D1ComponentGradientEvidence:
    """Pre-clip gradient norm or the registered stop-gradient absence."""

    kind: D1GradientEvidenceKind
    pre_clip_l2_norm: float | None

    def __post_init__(self) -> None:
        if type(self.kind) is not D1GradientEvidenceKind:
            raise TypeError("D1 component gradient kind must be D1GradientEvidenceKind")
        if self.kind is D1GradientEvidenceKind.MEASURED_NONZERO:
            if (
                type(self.pre_clip_l2_norm) is not float
                or not math.isfinite(self.pre_clip_l2_norm)
                or self.pre_clip_l2_norm <= 0.0
            ):
                raise ValueError("D1 measured component gradient must be a finite nonzero float")
        elif self.kind is D1GradientEvidenceKind.MEASURED_ZERO:
            if type(self.pre_clip_l2_norm) is not float or self.pre_clip_l2_norm != 0.0:
                raise ValueError("D1 measured zero gradient must have a zero float norm")
        elif self.pre_clip_l2_norm is not None:
            raise ValueError("D1 structurally absent gradient must have a null norm")

    @property
    def manifest(self) -> dict[str, object]:
        return {"kind": self.kind.value, "pre_clip_l2_norm": self.pre_clip_l2_norm}


@dataclass(frozen=True, slots=True)
class D1GradientReachability:
    """Closed four-component gradient evidence for one D1 backward pass."""

    online_branch: D1ComponentGradientEvidence
    target_branch: D1ComponentGradientEvidence
    language_readout: D1ComponentGradientEvidence
    future_law: D1ComponentGradientEvidence

    def __post_init__(self) -> None:
        components = (
            self.online_branch,
            self.target_branch,
            self.language_readout,
            self.future_law,
        )
        if any(type(item) is not D1ComponentGradientEvidence for item in components):
            raise TypeError("D1 gradient reachability requires four component evidence values")
        required = (self.online_branch, self.language_readout, self.future_law)
        measured = {D1GradientEvidenceKind.MEASURED_NONZERO, D1GradientEvidenceKind.MEASURED_ZERO}
        if any(item.kind not in measured for item in required):
            raise ValueError("D1 non-target component gradients must be measured")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "future_law": self.future_law.manifest,
            "language_readout": self.language_readout.manifest,
            "online_branch": self.online_branch.manifest,
            "target_branch": self.target_branch.manifest,
        }


@dataclass(frozen=True, slots=True)
class D1TrainStepResult:
    """Immutable, unrounded scalar evidence from one committed optimizer step."""

    successful_step: int
    learning_rate: float
    objective: float
    language_loss: float
    future_nll: float
    variance_penalty: float
    covariance_penalty: float
    raw_variance_penalty: float
    raw_covariance_penalty: float
    zero_displacement_nll: float
    mean_displacement_nll: float
    residual_mean: float
    residual_rms: float
    residual_variance: float
    coverage_50: float
    coverage_80: float
    coverage_95: float
    effective_rank: float
    numerical_rank: int
    gradient_norm: float
    gradient_reachability: D1GradientReachability

    def __post_init__(self) -> None:
        if type(self.successful_step) is not int or self.successful_step < 1:
            msg = "D1 train result successful_step must be a positive integer"
            raise ValueError(msg)
        float_fields = (
            "learning_rate",
            "objective",
            "language_loss",
            "future_nll",
            "variance_penalty",
            "covariance_penalty",
            "raw_variance_penalty",
            "raw_covariance_penalty",
            "zero_displacement_nll",
            "mean_displacement_nll",
            "residual_mean",
            "residual_rms",
            "residual_variance",
            "coverage_50",
            "coverage_80",
            "coverage_95",
            "effective_rank",
            "gradient_norm",
        )
        for name in float_fields:
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                msg = f"D1 train result {name} must be a finite float"
                raise ValueError(msg)
        if self.learning_rate <= 0.0 or self.gradient_norm < 0.0:
            msg = "D1 train result learning rate must be positive and gradient norm nonnegative"
            raise ValueError(msg)
        if any(not 0.0 <= value <= 1.0 for value in self.coverages):
            msg = "D1 train result coverage must be in [0, 1]"
            raise ValueError(msg)
        if type(self.numerical_rank) is not int or self.numerical_rank < 0:
            msg = "D1 train result numerical_rank must be a nonnegative integer"
            raise ValueError(msg)
        if type(self.gradient_reachability) is not D1GradientReachability:
            raise TypeError("D1 train result requires typed gradient reachability")
        _require_aggregate_gradient_norm(self.gradient_reachability, self.gradient_norm)

    @property
    def coverages(self) -> tuple[float, float, float]:
        return self.coverage_50, self.coverage_80, self.coverage_95

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "covariance_penalty": self.covariance_penalty,
            "coverage_50": self.coverage_50,
            "coverage_80": self.coverage_80,
            "coverage_95": self.coverage_95,
            "effective_rank": self.effective_rank,
            "future_nll": self.future_nll,
            "gradient_norm": self.gradient_norm,
            "gradient_reachability": self.gradient_reachability.manifest,
            "language_loss": self.language_loss,
            "learning_rate": self.learning_rate,
            "mean_displacement_nll": self.mean_displacement_nll,
            "numerical_rank": self.numerical_rank,
            "objective": self.objective,
            "raw_covariance_penalty": self.raw_covariance_penalty,
            "raw_variance_penalty": self.raw_variance_penalty,
            "residual_mean": self.residual_mean,
            "residual_rms": self.residual_rms,
            "residual_variance": self.residual_variance,
            "successful_step": self.successful_step,
            "variance_penalty": self.variance_penalty,
            "zero_displacement_nll": self.zero_displacement_nll,
        }


def _require_aggregate_gradient_norm(
    reachability: D1GradientReachability,
    aggregate_norm: float,
) -> None:
    """Bind component norms to the pre-clip total returned by ``clip_grad_norm_``."""
    component_norms = tuple(
        item.pre_clip_l2_norm
        for item in (
            reachability.online_branch,
            reachability.target_branch,
            reachability.language_readout,
            reachability.future_law,
        )
        if item.pre_clip_l2_norm is not None
    )
    if not math.isclose(
        aggregate_norm,
        math.hypot(*component_norms),
        rel_tol=1e-5,
        abs_tol=1e-12,
    ):
        raise ValueError("D1 component gradients do not reconstruct the aggregate norm")
