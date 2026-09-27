"""Deterministic, streaming evaluation for the D1 predictive-state tracer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import torch

from entropic_substrate.core.corpus.batching import SequenceSampler
from entropic_substrate.kernel.verdict.d1_evaluation_values import (
    TARGET_PREDICTED_EXACT_MATCH_SEMANTIC,
    D1ChannelMoments,
    D1CosineGap,
    validate_d1_cosine,
)
from entropic_substrate.ladder.capture.reader.d1_numerical_defect import d1_nonfinite_metric_error
from entropic_substrate.ladder.register.d1_execution import (
    PAIRWISE_COSINE_SEMANTIC,
    D1AmpPolicy,
)
from entropic_substrate.ladder.run.construction.runtime import D1PredictiveSystem
from entropic_substrate.ladder.run.construction.runtime_components import D1PredictiveResult

__all__ = [
    "PAIRWISE_COSINE_SEMANTIC",
    "TARGET_PREDICTED_EXACT_MATCH_SEMANTIC",
    "D1ChannelMoments",
    "D1EvalResult",
    "D1Evaluator",
]

_COVERAGE_Z: Final = (0.6744897501960817, 1.2815515655446004, 1.959963984540054)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-6)


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
    target_predicted_exact_match_count: int
    coverage_50: float
    coverage_80: float
    coverage_95: float
    state_sample_covariance: tuple[tuple[float, ...], ...]
    state_moments: D1ChannelMoments
    target_moments: D1ChannelMoments
    predicted_moments: D1ChannelMoments
    mean_pairwise_cosine_distance: float | D1CosineGap
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
        validate_d1_cosine(self.mean_pairwise_cosine_distance, positions=self.online_positions)
        if any(not 0.0 <= value <= 1.0 for value in self.coverages):
            msg = "D1 evaluation coverage must be in [0, 1]"
            raise ValueError(msg)
        if self.residual_rms < 0.0 or self.residual_variance < 0.0:
            raise ValueError("D1 evaluation residual RMS and variance must be nonnegative")
        if not self.coverage_50 <= self.coverage_80 <= self.coverage_95:
            raise ValueError("D1 evaluation coverage intervals must be nested")
        if any(
            type(channel) is not D1ChannelMoments
            for channel in (self.state_moments, self.target_moments, self.predicted_moments)
        ):
            raise TypeError("D1 evaluation channels must be D1ChannelMoments")
        width = self.state_moments.width
        if self.target_moments.width != width or self.predicted_moments.width != width:
            raise ValueError("D1 evaluation channels must share one latent width")
        if type(self.state_sample_covariance) is not tuple or any(
            type(row) is not tuple for row in self.state_sample_covariance
        ):
            raise TypeError("D1 evaluation covariance must use immutable tuple rows")
        if len(self.state_sample_covariance) != width or any(
            len(row) != width for row in self.state_sample_covariance
        ):
            msg = "D1 evaluation covariance must be square at the state width"
            raise ValueError(msg)
        state_values = (*(value for row in self.state_sample_covariance for value in row),)
        if any(type(value) is not float or not math.isfinite(value) for value in state_values):
            msg = "D1 evaluation state statistics must be finite floats"
            raise ValueError(msg)
        if any(
            not _close(row[index], self.state_variance[index])
            for index, row in enumerate(self.state_sample_covariance)
        ):
            raise ValueError("D1 evaluation state covariance diagonal differs from state variance")
        covariance = torch.tensor(self.state_sample_covariance, dtype=torch.float32)
        if not torch.allclose(covariance, covariance.T, rtol=1e-6, atol=1e-7):
            raise ValueError("D1 evaluation state covariance must be symmetric")
        spectrum = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        if not torch.allclose(
            spectrum,
            torch.tensor(self.state_covariance_eigenvalues),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("D1 evaluation state covariance differs from its spectrum")
        for name in (
            "batch_count",
            "online_positions",
            "residual_elements",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                msg = f"D1 evaluation {name} must be a positive integer"
                raise ValueError(msg)
        if self.residual_elements != self.online_positions * width:
            raise ValueError("D1 evaluation residual count must equal positions times latent width")
        if (
            type(self.target_predicted_exact_match_count) is not int
            or not 0 <= self.target_predicted_exact_match_count <= self.residual_elements
        ):
            raise ValueError("D1 evaluation exact-match count is outside residual elements")
        for channel in (self.state_moments, self.target_moments, self.predicted_moments):
            expected_square_norm = sum(value * value for value in channel.mean) + (
                (self.online_positions - 1) / self.online_positions
            ) * sum(channel.variance)
            if not _close(channel.rms_l2_norm * channel.rms_l2_norm, expected_square_norm):
                raise ValueError("D1 evaluation channel RMS differs from its moments")
        if not _close(self.residual_mean, sum(self.residual_mean_vector) / width):
            raise ValueError("D1 evaluation residual scalar mean differs from its vector")
        if not _close(
            self.residual_variance,
            self.residual_rms * self.residual_rms - self.residual_mean * self.residual_mean,
        ):
            raise ValueError("D1 evaluation residual variance differs from RMS and mean")

    @property
    def coverages(self) -> tuple[float, float, float]:
        return self.coverage_50, self.coverage_80, self.coverage_95

    @property
    def residual_mean_vector(self) -> tuple[float, ...]:
        return tuple(
            target - predicted
            for target, predicted in zip(
                self.target_moments.mean,
                self.predicted_moments.mean,
                strict=True,
            )
        )

    @property
    def residual_mean_l2_norm(self) -> float:
        return math.sqrt(sum(value * value for value in self.residual_mean_vector))

    @property
    def target_predicted_exact_match_fraction(self) -> float:
        return self.target_predicted_exact_match_count / self.residual_elements

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

    @property
    def numerical_rank(self) -> int:
        return self.state_moments.numerical_rank

    @property
    def state_rms_l2_norm(self) -> float:
        return self.state_moments.rms_l2_norm

    @property
    def target_predicted_exact_match_semantic(self) -> str:
        return TARGET_PREDICTED_EXACT_MATCH_SEMANTIC

    @property
    def telemetry_manifest(self) -> dict[str, object]:
        """Authoritative eval payload without the redundant dense covariance matrix."""
        return {
            "batch_count": self.batch_count,
            "bits_per_byte": self.bits_per_byte,
            "coverage_50": self.coverage_50,
            "coverage_80": self.coverage_80,
            "coverage_95": self.coverage_95,
            "effective_rank": self.effective_rank,
            "future_nll": self.future_nll,
            "language_loss": self.language_loss,
            "mean_baseline_improvement": self.mean_baseline_improvement,
            "mean_displacement_nll": self.mean_displacement_nll,
            "mean_pairwise_cosine_distance": (
                self.mean_pairwise_cosine_distance.manifest
                if type(self.mean_pairwise_cosine_distance) is D1CosineGap
                else self.mean_pairwise_cosine_distance
            ),
            "nats_per_byte": self.nats_per_byte,
            "numerical_rank": self.numerical_rank,
            "online_positions": self.online_positions,
            "pairwise_cosine_semantic": PAIRWISE_COSINE_SEMANTIC,
            "perplexity": self.perplexity,
            "predicted_covariance_eigenvalues": list(self.predicted_moments.covariance_eigenvalues),
            "predicted_effective_rank": self.predicted_moments.effective_rank,
            "predicted_mean": list(self.predicted_moments.mean),
            "predicted_numerical_rank": self.predicted_moments.numerical_rank,
            "predicted_rms_l2_norm": self.predicted_moments.rms_l2_norm,
            "predicted_variance": list(self.predicted_moments.variance),
            "raw_covariance_penalty": self.raw_covariance_penalty,
            "raw_variance_penalty": self.raw_variance_penalty,
            "residual_elements": self.residual_elements,
            "residual_mean": self.residual_mean,
            "residual_mean_l2_norm": self.residual_mean_l2_norm,
            "residual_mean_vector": list(self.residual_mean_vector),
            "residual_rms": self.residual_rms,
            "residual_variance": self.residual_variance,
            "state_covariance_eigenvalues": list(self.state_covariance_eigenvalues),
            "state_mean": list(self.state_mean),
            "state_rms_l2_norm": self.state_rms_l2_norm,
            "state_variance": list(self.state_variance),
            "target_covariance_eigenvalues": list(self.target_moments.covariance_eigenvalues),
            "target_effective_rank": self.target_moments.effective_rank,
            "target_mean": list(self.target_moments.mean),
            "target_numerical_rank": self.target_moments.numerical_rank,
            "target_predicted_exact_match_count": self.target_predicted_exact_match_count,
            "target_predicted_exact_match_fraction": self.target_predicted_exact_match_fraction,
            "target_predicted_exact_match_semantic": self.target_predicted_exact_match_semantic,
            "target_rms_l2_norm": self.target_moments.rms_l2_norm,
            "target_variance": list(self.target_moments.variance),
            "weighted_covariance_penalty": self.weighted_covariance_penalty,
            "weighted_variance_penalty": self.weighted_variance_penalty,
            "zero_baseline_improvement": self.zero_baseline_improvement,
            "zero_displacement_nll": self.zero_displacement_nll,
        }


@dataclass(slots=True)
class _ChannelAccumulator:
    width: int
    device: torch.device
    label: str
    pairwise_cosine: bool = False
    count: int = 0
    zero_norm_count: int = 0
    mean: torch.Tensor = field(init=False)
    centered_outer_sum: torch.Tensor = field(init=False)
    normalized_sum: torch.Tensor = field(init=False)
    squared_norm_sum: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.mean = torch.zeros(self.width, dtype=torch.float32, device=self.device)
        self.centered_outer_sum = torch.zeros(
            (self.width, self.width), dtype=torch.float32, device=self.device
        )
        self.normalized_sum = torch.zeros(self.width, dtype=torch.float32, device=self.device)
        self.squared_norm_sum = torch.zeros((), dtype=torch.float32, device=self.device)

    def add(self, channel: torch.Tensor) -> None:
        if (
            type(channel) is not torch.Tensor
            or channel.ndim != 3
            or channel.shape[-1] != self.width
        ):
            raise ValueError(f"D1 evaluation {self.label} must be rank-3 at the latent width")
        values = channel.detach().float().reshape(-1, self.width)
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError(f"D1 evaluation {self.label} must be finite")
        norms = torch.linalg.vector_norm(values, dim=1)
        if self.pairwise_cosine:
            self.zero_norm_count += int((norms == 0.0).sum().item())
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
        if self.pairwise_cosine and self.zero_norm_count == 0:
            self.normalized_sum.add_((values / norms.unsqueeze(1)).sum(dim=0))
        self.squared_norm_sum.add_(values.square().sum())
        self.count += batch_count

    def finish(self) -> tuple[D1ChannelMoments, torch.Tensor, float | D1CosineGap | None]:
        if self.count < 2:
            raise ValueError(f"D1 evaluation requires at least two {self.label} positions")
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
        pairwise: float | D1CosineGap | None = None
        if self.pairwise_cosine and self.zero_norm_count:
            pairwise = D1CosineGap(self.zero_norm_count, self.count)
        elif self.pairwise_cosine:
            cosine_mean = (self.normalized_sum.square().sum() - self.count) / (
                self.count * (self.count - 1)
            )
            pairwise = float((1.0 - cosine_mean).item())
        moments = D1ChannelMoments(
            mean=tuple(float(value) for value in self.mean.tolist()),
            variance=tuple(float(value) for value in covariance.diagonal().tolist()),
            covariance_eigenvalues=tuple(float(value) for value in eigenvalues.tolist()),
            effective_rank=effective_rank,
            numerical_rank=numerical_rank,
            rms_l2_norm=float((self.squared_norm_sum / self.count).sqrt().item()),
        )
        return moments, covariance, pairwise


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
        width = system.draft.encoder_pair.latent_width
        state_moments = _ChannelAccumulator(width, device, "online states", pairwise_cosine=True)
        target_moments = _ChannelAccumulator(width, device, "actual target displacements")
        predicted_moments = _ChannelAccumulator(width, device, "predicted displacements")
        sums = {name: torch.zeros((), dtype=torch.float32, device=device) for name in _SUM_NAMES}
        residual_sum = torch.zeros((), dtype=torch.float32, device=device)
        residual_square_sum = torch.zeros((), dtype=torch.float32, device=device)
        coverage_counts = [0, 0, 0]
        exact_match_count = 0
        residual_elements = 0
        positions = 0
        try:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                for _ in range(spec.num_batches):
                    batch = sampler.predictive_batch(horizon=1)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp is D1AmpPolicy.BF16,
                    ):
                        output = system(batch)
                    if output.collapse.diagnostics is None:
                        msg = "D1 evaluation requires complete collapse diagnostics"
                        raise ValueError(msg)
                    batch_positions = (
                        output.online_states.values.shape[0] * output.online_states.values.shape[1]
                    )
                    target = output.target.values.float()
                    predicted = output.future_law_loss.mean.float()
                    residual = target - predicted
                    if not bool(torch.isfinite(residual).all().item()):
                        msg = "D1 evaluation future residuals must be finite"
                        raise ValueError(msg)
                    batch_residuals = residual.numel()
                    state_moments.add(output.online_states.values)
                    target_moments.add(target)
                    predicted_moments.add(predicted)
                    _add_weighted(sums, output, batch_positions, batch_residuals)
                    residual_sum.add_(residual.sum())
                    residual_square_sum.add_(residual.square().sum())
                    exact_match_count += int((target == predicted).sum().item())
                    for index, threshold in enumerate(_COVERAGE_Z):
                        coverage_counts[index] += int((residual.abs() <= threshold).sum().item())
                    positions += batch_positions
                    residual_elements += batch_residuals
                result = _finish_result(
                    sums=sums,
                    states=state_moments,
                    targets=target_moments,
                    predictions=predicted_moments,
                    residual_sum=residual_sum,
                    residual_square_sum=residual_square_sum,
                    coverage_counts=coverage_counts,
                    exact_match_count=exact_match_count,
                    batch_count=spec.num_batches,
                    positions=positions,
                    residual_elements=residual_elements,
                )
        finally:
            system.train(was_training)
        return result

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
    states: _ChannelAccumulator,
    targets: _ChannelAccumulator,
    predictions: _ChannelAccumulator,
    residual_sum: torch.Tensor,
    residual_square_sum: torch.Tensor,
    coverage_counts: list[int],
    exact_match_count: int,
    batch_count: int,
    positions: int,
    residual_elements: int,
) -> D1EvalResult:
    state_moments, covariance, pairwise = states.finish()
    target_moments, _, target_pairwise = targets.finish()
    predicted_moments, _, predicted_pairwise = predictions.finish()
    if (
        target_pairwise is not None
        or predicted_pairwise is not None
        or pairwise is None
        or states.count != targets.count
        or states.count != predictions.count
        or states.count != positions
    ):
        raise ValueError("D1 evaluation channel accumulation invariants differ")
    language = float((sums["language"] / positions).item())
    if not math.isfinite(language):
        raise d1_nonfinite_metric_error(
            metric_name="nats_per_byte",
            value=language,
            source_exception_code="d1.eval.nats_per_byte.nonfinite.v1",
        )
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
        target_predicted_exact_match_count=exact_match_count,
        coverage_50=coverage_counts[0] / residual_elements,
        coverage_80=coverage_counts[1] / residual_elements,
        coverage_95=coverage_counts[2] / residual_elements,
        state_sample_covariance=tuple(
            tuple(float(value) for value in row) for row in covariance.tolist()
        ),
        state_moments=state_moments,
        target_moments=target_moments,
        predicted_moments=predicted_moments,
        mean_pairwise_cosine_distance=pairwise,
        raw_variance_penalty=float((sums["raw_variance"] / positions).item()),
        weighted_variance_penalty=float((sums["weighted_variance"] / positions).item()),
        raw_covariance_penalty=float((sums["raw_covariance"] / positions).item()),
        weighted_covariance_penalty=float((sums["weighted_covariance"] / positions).item()),
        batch_count=batch_count,
        online_positions=positions,
        residual_elements=residual_elements,
    )
