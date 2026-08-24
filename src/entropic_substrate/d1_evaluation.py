"""Deterministic, streaming evaluation for the D1 predictive-state tracer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import torch

from entropic_substrate.data.batching import SequenceSampler
from entropic_substrate.experiments.d1_execution import (
    PAIRWISE_COSINE_SEMANTIC,
    D1AmpPolicy,
)
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem
from entropic_substrate.predictive_state.runtime_components import D1PredictiveResult

__all__ = [
    "PAIRWISE_COSINE_SEMANTIC",
    "D1EvalResult",
    "D1Evaluator",
]

_COVERAGE_Z: Final = (0.6744897501960817, 1.2815515655446004, 1.959963984540054)


@dataclass(frozen=True, slots=True)
class D1EvalResult:
    """Immutable full-precision scalar and latent-state evidence from one eval."""

    language_loss: float
    nats_per_byte: float
    bits_per_byte: float
    perplexity: float
    future_nll: float
    zero_displacement_nll: float
    mean_displacement_nll: float
    zero_baseline_improvement: float
    mean_baseline_improvement: float
    residual_mean: float
    residual_rms: float
    residual_variance: float
    coverage_50: float
    coverage_80: float
    coverage_95: float
    state_sample_covariance: tuple[tuple[float, ...], ...]
    state_variance: tuple[float, ...]
    state_covariance_eigenvalues: tuple[float, ...]
    effective_rank: float
    numerical_rank: int
    mean_pairwise_cosine_distance: float
    raw_variance_penalty: float
    weighted_variance_penalty: float
    raw_covariance_penalty: float
    weighted_covariance_penalty: float
    batch_count: int
    online_positions: int
    residual_elements: int

    def __post_init__(self) -> None:
        scalar_names = (
            "language_loss",
            "nats_per_byte",
            "bits_per_byte",
            "perplexity",
            "future_nll",
            "zero_displacement_nll",
            "mean_displacement_nll",
            "zero_baseline_improvement",
            "mean_baseline_improvement",
            "residual_mean",
            "residual_rms",
            "residual_variance",
            "coverage_50",
            "coverage_80",
            "coverage_95",
            "effective_rank",
            "mean_pairwise_cosine_distance",
            "raw_variance_penalty",
            "weighted_variance_penalty",
            "raw_covariance_penalty",
            "weighted_covariance_penalty",
        )
        for name in scalar_names:
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                msg = f"D1 evaluation {name} must be a finite float"
                raise ValueError(msg)
        if any(not 0.0 <= value <= 1.0 for value in self.coverages):
            msg = "D1 evaluation coverage must be in [0, 1]"
            raise ValueError(msg)
        width = len(self.state_variance)
        if width < 1 or len(self.state_covariance_eigenvalues) != width:
            msg = "D1 evaluation state vectors must share one positive width"
            raise ValueError(msg)
        if len(self.state_sample_covariance) != width or any(
            len(row) != width for row in self.state_sample_covariance
        ):
            msg = "D1 evaluation covariance must be square at the state width"
            raise ValueError(msg)
        state_values = (
            *self.state_variance,
            *self.state_covariance_eigenvalues,
            *(value for row in self.state_sample_covariance for value in row),
        )
        if any(type(value) is not float or not math.isfinite(value) for value in state_values):
            msg = "D1 evaluation state statistics must be finite floats"
            raise ValueError(msg)
        if type(self.numerical_rank) is not int or not 0 <= self.numerical_rank <= width:
            msg = "D1 evaluation numerical rank is outside the state width"
            raise ValueError(msg)
        for name in ("batch_count", "online_positions", "residual_elements"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                msg = f"D1 evaluation {name} must be a positive integer"
                raise ValueError(msg)

    @property
    def coverages(self) -> tuple[float, float, float]:
        return self.coverage_50, self.coverage_80, self.coverage_95


@dataclass(slots=True)
class _StateMoments:
    width: int
    device: torch.device
    count: int = 0
    mean: torch.Tensor = field(init=False)
    centered_outer_sum: torch.Tensor = field(init=False)
    normalized_sum: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.mean = torch.zeros(self.width, dtype=torch.float32, device=self.device)
        self.centered_outer_sum = torch.zeros(
            (self.width, self.width), dtype=torch.float32, device=self.device
        )
        self.normalized_sum = torch.zeros(self.width, dtype=torch.float32, device=self.device)

    def add(self, states: torch.Tensor) -> None:
        values = states.detach().float().reshape(-1, self.width)
        if not bool(torch.isfinite(values).all().item()):
            msg = "D1 evaluation online states must be finite"
            raise ValueError(msg)
        norms = torch.linalg.vector_norm(values, dim=1)
        if bool((norms == 0.0).any().item()):
            msg = "D1 pairwise cosine evidence forbids zero-norm online states"
            raise ValueError(msg)
        batch_count = values.shape[0]
        batch_mean = values.mean(dim=0)
        centered = values - batch_mean
        batch_outer = centered.T @ centered
        if self.count == 0:
            self.mean.copy_(batch_mean)
            self.centered_outer_sum.copy_(batch_outer)
        else:
            total = self.count + batch_count
            delta = batch_mean - self.mean
            self.centered_outer_sum.add_(batch_outer)
            self.centered_outer_sum.add_(
                torch.outer(delta, delta) * (self.count * batch_count / total)
            )
            self.mean.add_(delta * (batch_count / total))
        self.normalized_sum.add_((values / norms.unsqueeze(1)).sum(dim=0))
        self.count += batch_count

    def finish(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int, float]:
        if self.count < 2:
            msg = "D1 evaluation requires at least two online state positions"
            raise ValueError(msg)
        covariance = self.centered_outer_sum / float(self.count - 1)
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        mass = eigenvalues.sum()
        effective_rank = 0.0
        if float(mass.item()) > 0.0:
            probabilities = eigenvalues / mass
            positive = probabilities > 0.0
            effective_rank = float(
                (-(probabilities[positive] * probabilities[positive].log()).sum()).exp().item()
            )
        numerical_rank = int(torch.linalg.matrix_rank(covariance).item())
        cosine_mean = (self.normalized_sum.square().sum() - self.count) / (
            self.count * (self.count - 1)
        )
        return (
            covariance,
            covariance.diagonal(),
            eigenvalues,
            effective_rank,
            numerical_rank,
            float((1.0 - cosine_mean).item()),
        )


class D1Evaluator:
    """Apply the registered D1 eval schedule without retaining trajectories."""

    def evaluate(self, *, system: D1PredictiveSystem, sampler: SequenceSampler) -> D1EvalResult:
        self._validate_inputs(system=system, sampler=sampler)
        spec = system.draft.execution_protocol.evaluation
        amp = system.draft.execution_protocol.optimization.amp
        device = next(system.parameters()).device
        sampler.reset_eval()
        was_training = system.training
        system.eval()
        state_moments = _StateMoments(system.draft.encoder_pair.latent_width, device)
        sums = {name: torch.zeros((), dtype=torch.float32, device=device) for name in _SUM_NAMES}
        residual_sum = torch.zeros((), dtype=torch.float32, device=device)
        residual_square_sum = torch.zeros((), dtype=torch.float32, device=device)
        coverage_counts = [0, 0, 0]
        residual_elements = 0
        positions = 0
        try:
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp is D1AmpPolicy.BF16,
                ),
            ):
                for _ in range(spec.num_batches):
                    batch = sampler.predictive_batch(horizon=1)
                    output = system(batch)
                    if output.collapse.diagnostics is None:
                        msg = "D1 evaluation requires complete collapse diagnostics"
                        raise ValueError(msg)
                    batch_positions = (
                        output.online_states.values.shape[0] * output.online_states.values.shape[1]
                    )
                    residual = output.target.values.float() - output.future_law_loss.mean.float()
                    if not bool(torch.isfinite(residual).all().item()):
                        msg = "D1 evaluation future residuals must be finite"
                        raise ValueError(msg)
                    batch_residuals = residual.numel()
                    state_moments.add(output.online_states.values)
                    _add_weighted(sums, output, batch_positions, batch_residuals)
                    residual_sum.add_(residual.sum())
                    residual_square_sum.add_(residual.square().sum())
                    for index, threshold in enumerate(_COVERAGE_Z):
                        coverage_counts[index] += int((residual.abs() <= threshold).sum().item())
                    positions += batch_positions
                    residual_elements += batch_residuals
        finally:
            system.train(was_training)
        return _finish_result(
            sums=sums,
            states=state_moments,
            residual_sum=residual_sum,
            residual_square_sum=residual_square_sum,
            coverage_counts=coverage_counts,
            batch_count=spec.num_batches,
            positions=positions,
            residual_elements=residual_elements,
        )

    @staticmethod
    def _validate_inputs(*, system: D1PredictiveSystem, sampler: SequenceSampler) -> None:
        if type(system) is not D1PredictiveSystem:
            msg = "D1 evaluator requires a D1PredictiveSystem"
            raise TypeError(msg)
        if type(sampler) is not SequenceSampler or sampler.train:
            msg = "D1 evaluator requires a deterministic evaluation sampler"
            raise ValueError(msg)
        protocol = system.draft.execution_protocol.batching
        if sampler.batch_size != protocol.batch_size or sampler.seq_len != protocol.sequence_length:
            msg = "D1 evaluation sampler shape must match the registered batching protocol"
            raise ValueError(msg)
        if sampler.device != next(system.parameters()).device:
            msg = "D1 evaluation sampler and system must share one device"
            raise ValueError(msg)
        if not system.mean_displacement_baseline.is_fitted:
            msg = "D1 evaluation requires the frozen training-fitted mean baseline"
            raise ValueError(msg)


_SUM_NAMES: Final = (
    "language",
    "future",
    "zero",
    "mean",
    "raw_variance",
    "weighted_variance",
    "raw_covariance",
    "weighted_covariance",
)


def _add_weighted(
    sums: dict[str, torch.Tensor],
    output: D1PredictiveResult,
    positions: int,
    residuals: int,
) -> None:
    law = output.future_law_loss
    collapse = output.collapse
    sums["language"].add_(output.objective.language.float() * positions)
    sums["future"].add_(law.nll.float() * residuals)
    sums["zero"].add_(law.zero_displacement_nll.float() * residuals)
    sums["mean"].add_(law.mean_displacement_nll.float() * residuals)
    sums["raw_variance"].add_(collapse.raw_variance_penalty.float() * positions)
    sums["weighted_variance"].add_(collapse.weighted_variance_penalty.float() * positions)
    sums["raw_covariance"].add_(collapse.raw_covariance_penalty.float() * positions)
    sums["weighted_covariance"].add_(collapse.weighted_covariance_penalty.float() * positions)


def _finish_result(
    *,
    sums: dict[str, torch.Tensor],
    states: _StateMoments,
    residual_sum: torch.Tensor,
    residual_square_sum: torch.Tensor,
    coverage_counts: list[int],
    batch_count: int,
    positions: int,
    residual_elements: int,
) -> D1EvalResult:
    covariance, variance, eigenvalues, effective_rank, numerical_rank, pairwise = states.finish()
    language = float((sums["language"] / positions).item())
    future = float((sums["future"] / residual_elements).item())
    zero = float((sums["zero"] / residual_elements).item())
    mean = float((sums["mean"] / residual_elements).item())
    residual_mean = residual_sum / residual_elements
    residual_second = residual_square_sum / residual_elements
    return D1EvalResult(
        language_loss=language,
        nats_per_byte=language,
        bits_per_byte=language / math.log(2.0),
        perplexity=math.exp(language),
        future_nll=future,
        zero_displacement_nll=zero,
        mean_displacement_nll=mean,
        zero_baseline_improvement=zero - future,
        mean_baseline_improvement=mean - future,
        residual_mean=float(residual_mean.item()),
        residual_rms=float(residual_second.sqrt().item()),
        residual_variance=float((residual_second - residual_mean.square()).clamp_min(0.0).item()),
        coverage_50=coverage_counts[0] / residual_elements,
        coverage_80=coverage_counts[1] / residual_elements,
        coverage_95=coverage_counts[2] / residual_elements,
        state_sample_covariance=tuple(
            tuple(float(value) for value in row) for row in covariance.tolist()
        ),
        state_variance=tuple(float(value) for value in variance.tolist()),
        state_covariance_eigenvalues=tuple(float(value) for value in eigenvalues.tolist()),
        effective_rank=effective_rank,
        numerical_rank=numerical_rank,
        mean_pairwise_cosine_distance=pairwise,
        raw_variance_penalty=float((sums["raw_variance"] / positions).item()),
        weighted_variance_penalty=float((sums["weighted_variance"] / positions).item()),
        raw_covariance_penalty=float((sums["raw_covariance"] / positions).item()),
        weighted_covariance_penalty=float((sums["weighted_covariance"] / positions).item()),
        batch_count=batch_count,
        online_positions=positions,
        residual_elements=residual_elements,
    )
