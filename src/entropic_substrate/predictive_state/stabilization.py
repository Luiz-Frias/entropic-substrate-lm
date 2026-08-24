"""Online-latent collapse penalties and diagnostics for D1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from entropic_substrate.predictive_state.state import LatentTrajectory

__all__ = [
    "CollapseDiagnostics",
    "CollapseRegularizer",
    "CollapseResult",
    "MomentCollapseRegularizer",
    "NoCollapseRegularizer",
]


@dataclass(frozen=True, slots=True)
class CollapseDiagnostics:
    """FP32 covariance/rank statistics over online latent positions only."""

    variance: torch.Tensor
    std: torch.Tensor
    covariance: torch.Tensor
    covariance_eigenvalues: torch.Tensor
    effective_rank: float
    numerical_rank: int

    def __post_init__(self) -> None:
        if (
            self.variance.dtype != torch.float32
            or self.std.dtype != torch.float32
            or self.covariance.dtype != torch.float32
        ):
            msg = "collapse variance, std, and covariance diagnostics must be FP32"
            raise ValueError(msg)
        if self.covariance_eigenvalues.dtype != torch.float32:
            msg = "collapse covariance eigenvalue diagnostics must be FP32"
            raise ValueError(msg)
        if self.variance.ndim != 1 or self.std.ndim != 1 or self.covariance.ndim != 2:
            msg = "collapse diagnostics require vector variance/std and matrix covariance"
            raise ValueError(msg)
        if self.std.shape != self.variance.shape:
            msg = "collapse std dimensions must match variance dimensions"
            raise ValueError(msg)
        if self.covariance.shape != (self.variance.numel(), self.variance.numel()):
            msg = "collapse covariance dimensions must match the variance width"
            raise ValueError(msg)
        if not math.isfinite(self.effective_rank) or self.effective_rank < 0.0:
            msg = f"effective rank must be finite and nonnegative, got {self.effective_rank!r}"
            raise ValueError(msg)
        if type(self.numerical_rank) is not int or self.numerical_rank < 0:
            msg = f"numerical rank must be a nonnegative integer, got {self.numerical_rank!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CollapseResult:
    """Raw and weighted VICReg terms kept unambiguous for the D1 objective."""

    raw_variance_penalty: torch.Tensor
    weighted_variance_penalty: torch.Tensor
    raw_covariance_penalty: torch.Tensor
    weighted_covariance_penalty: torch.Tensor
    total: torch.Tensor
    diagnostics: CollapseDiagnostics | None

    def __post_init__(self) -> None:
        for name in (
            "raw_variance_penalty",
            "weighted_variance_penalty",
            "raw_covariance_penalty",
            "weighted_covariance_penalty",
            "total",
        ):
            value = getattr(self, name)
            if type(value) is not torch.Tensor or value.ndim != 0 or value.dtype != torch.float32:
                msg = f"{name} must be an FP32 scalar tensor"
                raise ValueError(msg)


@runtime_checkable
class CollapseRegularizer(Protocol):
    """Online-state-only strategy that retains separate variance/covariance terms."""

    def __call__(self, online_states: LatentTrajectory) -> CollapseResult: ...


def _positive_float(name: str, value: float) -> float:
    if not isinstance(value, float) or not math.isfinite(value) or value <= 0.0:
        msg = f"{name} must be a finite positive float, got {value!r}"
        raise ValueError(msg)
    return value


class NoCollapseRegularizer:
    """Explicit off control with exact zero terms and report-only diagnostics."""

    def __init__(self, *, variance_eps: float = 1e-4) -> None:
        self.variance_eps = _positive_float("variance_eps", variance_eps)

    def __call__(self, online_states: LatentTrajectory) -> CollapseResult:
        if type(online_states) is not LatentTrajectory:
            msg = "collapse regularizer requires a LatentTrajectory"
            raise TypeError(msg)
        n_positions = online_states.values.shape[0] * online_states.values.shape[1]
        if n_positions < 2:
            msg = "collapse regularizer requires at least two online latent positions"
            raise ValueError(msg)
        # NONE/SG arms permit collapse as an outcome, but they must still
        # measure it.  Detaching makes this a report-only calculation while
        # preserving exact zero objective terms below.
        with torch.autocast(device_type=online_states.values.device.type, enabled=False):
            values32 = (
                online_states.values.detach().float().reshape(n_positions, online_states.width)
            )
            diagnostics = _collapse_diagnostics(values32=values32, variance_eps=self.variance_eps)
        zero = torch.zeros((), dtype=torch.float32, device=online_states.values.device)
        return CollapseResult(
            raw_variance_penalty=zero,
            weighted_variance_penalty=zero,
            raw_covariance_penalty=zero,
            weighted_covariance_penalty=zero,
            total=zero,
            diagnostics=diagnostics,
        )


def _collapse_diagnostics(*, values32: torch.Tensor, variance_eps: float) -> CollapseDiagnostics:
    """Calculate the shared FP32 sample-covariance and VICReg-std report."""
    if values32.shape[0] < 2:
        msg = "collapse diagnostics require at least two positions"
        raise ValueError(msg)
    centered = values32 - values32.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(values32.shape[0] - 1)
    variance = covariance.diagonal()
    std = (variance + variance_eps).sqrt()
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    mass = eigenvalues.sum()
    if float(mass.detach()) == 0.0:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues / mass
        positive = probabilities > 0.0
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        effective_rank = float(entropy.exp().detach())
    numerical_rank = int(torch.linalg.matrix_rank(covariance).item())
    return CollapseDiagnostics(
        variance=variance,
        std=std,
        covariance=covariance,
        covariance_eigenvalues=eigenvalues,
        effective_rank=effective_rank,
        numerical_rank=numerical_rank,
    )


class MomentCollapseRegularizer:
    """FP32 variance-floor and off-diagonal covariance penalty.

    The loss is calculated only from the online latent trajectory.  Target
    states are deliberately not accepted by this interface, which makes a
    target-collapse penalty impossible to add accidentally at this seam.
    """

    def __init__(
        self,
        *,
        gamma: float,
        variance_weight: float,
        covariance_weight: float,
        variance_eps: float,
    ) -> None:
        self.gamma = _positive_float("gamma", gamma)
        if (
            not isinstance(variance_weight, float)
            or not math.isfinite(variance_weight)
            or variance_weight < 0.0
        ):
            msg = f"variance_weight must be a finite nonnegative float, got {variance_weight!r}"
            raise ValueError(msg)
        if (
            not isinstance(covariance_weight, float)
            or not math.isfinite(covariance_weight)
            or covariance_weight < 0.0
        ):
            msg = f"covariance_weight must be a finite nonnegative float, got {covariance_weight!r}"
            raise ValueError(msg)
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_eps = _positive_float("variance_eps", variance_eps)

    def __call__(self, online_states: LatentTrajectory) -> CollapseResult:
        if type(online_states) is not LatentTrajectory:
            msg = "collapse regularizer requires a LatentTrajectory"
            raise TypeError(msg)
        n_positions = online_states.values.shape[0] * online_states.values.shape[1]
        if n_positions < 2:
            msg = "collapse regularizer requires at least two online latent positions"
            raise ValueError(msg)
        with torch.autocast(device_type=online_states.values.device.type, enabled=False):
            values32 = online_states.values.float().reshape(n_positions, online_states.width)
            diagnostics = _collapse_diagnostics(values32=values32, variance_eps=self.variance_eps)
            covariance = diagnostics.covariance
            raw_variance_penalty = torch.relu(self.gamma - diagnostics.std).mean()
            off_diagonal = covariance - torch.diag_embed(diagnostics.variance)
            raw_covariance_penalty = off_diagonal.square().sum() / float(online_states.width)
            weighted_variance_penalty = raw_variance_penalty * self.variance_weight
            weighted_covariance_penalty = raw_covariance_penalty * self.covariance_weight
            total = weighted_variance_penalty + weighted_covariance_penalty
        return CollapseResult(
            raw_variance_penalty=raw_variance_penalty,
            weighted_variance_penalty=weighted_variance_penalty,
            raw_covariance_penalty=raw_covariance_penalty,
            weighted_covariance_penalty=weighted_covariance_penalty,
            total=total,
            diagnostics=diagnostics,
        )
