"""The fixed-variance diagonal Gaussian future law used by the D1 tracer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from entropic_substrate.core.latent.state import LatentTrajectory
from entropic_substrate.core.latent.target_values import FutureTarget, StabilizedTargets
from entropic_substrate.core.observation.contracts import CodecIdentity

__all__ = [
    "DiagonalGaussianFutureLaw",
    "FutureLaw",
    "FutureLawLoss",
    "MeanDisplacementBaseline",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)
_MARGINAL_GAUSSIAN_Z = {
    "50": 0.6744897501960817,
    "80": 1.2815515655446004,
    "95": 1.959963984540054,
}


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        msg = f"{name} must be a positive integer, got {value!r}"
        raise ValueError(msg)
    return value


def _validate_scalar(name: str, value: torch.Tensor) -> None:
    if type(value) is not torch.Tensor or value.ndim != 0 or not value.is_floating_point():
        msg = f"{name} must be a floating scalar tensor"
        raise ValueError(msg)
    if not bool(torch.isfinite(value).item()):
        msg = f"{name} must be finite"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class FutureLaw:
    """Conditional mean of D1's fixed ``Normal(mu(h), I)`` law."""

    mean: torch.Tensor
    codec_identity: CodecIdentity

    def __post_init__(self) -> None:
        if type(self.mean) is not torch.Tensor or self.mean.ndim != 3:
            msg = "future-law mean must be a rank-3 tensor"
            raise ValueError(msg)
        if self.mean.dtype != torch.float32:
            msg = f"future-law mean must be FP32, got {self.mean.dtype}"
            raise ValueError(msg)
        if not bool(torch.isfinite(self.mean).all().item()):
            msg = "future-law mean must be finite"
            raise ValueError(msg)
        if type(self.codec_identity) is not CodecIdentity:
            msg = "future law requires a CodecIdentity"
            raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class FutureLawLoss:
    """Future-law loss and the declared trivial grounding comparators."""

    nll: torch.Tensor
    zero_displacement_nll: torch.Tensor
    mean_displacement_nll: torch.Tensor
    residual_mean: torch.Tensor
    residual_rms: torch.Tensor
    residual_variance: torch.Tensor
    coverage_50: torch.Tensor
    coverage_80: torch.Tensor
    coverage_95: torch.Tensor
    mean: torch.Tensor

    def __post_init__(self) -> None:
        for name in (
            "nll",
            "zero_displacement_nll",
            "mean_displacement_nll",
            "residual_mean",
            "residual_rms",
            "residual_variance",
            "coverage_50",
            "coverage_80",
            "coverage_95",
        ):
            _validate_scalar(name, getattr(self, name))
        if type(self.mean) is not torch.Tensor or self.mean.ndim != 3:
            msg = "future-law loss mean must be rank-3"
            raise ValueError(msg)
        if not bool(torch.isfinite(self.mean).all().item()):
            msg = "future-law loss mean must be finite"
            raise ValueError(msg)


class MeanDisplacementBaseline(nn.Module):
    """Frozen training-fitted mean-displacement comparator for D1 evaluation.

    The baseline is intentionally an ``nn.Module`` so its vector, fitted
    marker, and codec provenance live in a normal ``state_dict``.  ``fit`` is
    single-use: calling it at evaluation time would turn a comparator into a
    post-treatment estimate of the target it scores.
    """

    # Registered buffers are annotated so static analysis does not widen
    # nn.Module's dynamic attribute lookup to ``Tensor | Module``.
    _mean: torch.Tensor
    _fitted: torch.Tensor

    def __init__(self, *, latent_width: int) -> None:
        super().__init__()
        self.latent_width = _positive_int("latent_width", latent_width)
        self.register_buffer("_mean", torch.zeros(self.latent_width, dtype=torch.float32))
        self.register_buffer("_fitted", torch.tensor(False, dtype=torch.bool))
        self._codec_identity: CodecIdentity | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether this baseline has been fitted from a declared training target."""
        return bool(self._fitted.item())

    @property
    def mean(self) -> torch.Tensor:
        """The frozen FP32 training mean; unavailable before fitting."""
        if not self.is_fitted:
            msg = "mean-displacement baseline must be fitted before its mean is read"
            raise ValueError(msg)
        return self._mean

    @property
    def codec_identity(self) -> CodecIdentity:
        """Provenance of the training target used to fit the frozen baseline."""
        if not self.is_fitted or self._codec_identity is None:
            msg = "mean-displacement baseline must be fitted before codec identity is read"
            raise ValueError(msg)
        return self._codec_identity

    def fit(self, training_target: FutureTarget | StabilizedTargets) -> None:
        """Fit exactly once from training targets, then freeze the vector."""
        if type(training_target) not in (FutureTarget, StabilizedTargets):
            msg = "mean-displacement baseline fitting requires FutureTarget or StabilizedTargets"
            raise TypeError(msg)
        if self.is_fitted:
            msg = "mean-displacement baseline is already fitted and cannot be recomputed"
            raise RuntimeError(msg)
        if training_target.width != self.latent_width:
            msg = (
                f"training target width {training_target.width} does not match baseline width "
                f"{self.latent_width}"
            )
            raise ValueError(msg)
        if training_target.values.device != self._mean.device:
            msg = "training target and mean-displacement baseline must share one device"
            raise ValueError(msg)
        with (
            torch.no_grad(),
            torch.autocast(device_type=training_target.values.device.type, enabled=False),
        ):
            self._mean.copy_(training_target.values.float().mean(dim=(0, 1)))
            self._fitted.fill_(True)
        self._codec_identity = training_target.codec_identity

    def validate_for(self, target: FutureTarget | StabilizedTargets) -> None:
        """Reject an unfitted, mismatched, or nonlocal frozen comparator."""
        if not self.is_fitted:
            msg = "mean-displacement baseline must be fitted before future-law scoring"
            raise ValueError(msg)
        if target.width != self.latent_width:
            msg = f"future target width {target.width} does not match baseline width {self.latent_width}"
            raise ValueError(msg)
        if target.codec_identity != self.codec_identity:
            msg = "future target and mean-displacement baseline codec identities must match"
            raise ValueError(msg)
        if target.values.device != self._mean.device:
            msg = "future target and mean-displacement baseline must share one device"
            raise ValueError(msg)

    def get_extra_state(self) -> dict[str, object]:
        """Serialise codec provenance alongside the registered tensor buffers."""
        if self._codec_identity is None:
            return {"codec_identity": None}
        return {
            "codec_identity": {
                "name": self._codec_identity.name,
                "construction_digest": self._codec_identity.construction_digest,
                "cardinality": self._codec_identity.cardinality,
            }
        }

    def set_extra_state(self, state: object) -> None:
        """Restore a validated codec identity from a module state dictionary."""
        if not isinstance(state, dict):
            msg = "mean-displacement baseline extra state must be a dictionary"
            raise TypeError(msg)
        codec_state = state.get("codec_identity")
        if codec_state is None:
            self._codec_identity = None
            return
        if not isinstance(codec_state, dict):
            msg = "mean-displacement baseline codec state must be a dictionary or null"
            raise ValueError(msg)
        name = codec_state.get("name")
        digest = codec_state.get("construction_digest")
        cardinality = codec_state.get("cardinality")
        if not isinstance(name, str) or not isinstance(digest, str) or type(cardinality) is not int:
            msg = "mean-displacement baseline codec state is malformed"
            raise ValueError(msg)
        self._codec_identity = CodecIdentity(
            name=name,
            construction_digest=digest,
            cardinality=cardinality,
        )


class DiagonalGaussianFutureLaw(nn.Module):
    """Nonlinear mean predictor for ``q(Y | h) = Normal(mu(h), I)``.

    There is deliberately no learned variance or categorical output.  The
    fixed unit diagonal covariance is part of D1's registered tracer and
    makes the full per-dimension NLL directly comparable to its zero and
    frozen-training-mean displacement baselines.
    """

    def __init__(self, *, latent_width: int, hidden_width: int) -> None:
        super().__init__()
        self.latent_width = _positive_int("latent_width", latent_width)
        self.hidden_width = _positive_int("hidden_width", hidden_width)
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_width, self.hidden_width),
            nn.SiLU(),
            nn.Linear(self.hidden_width, self.latent_width),
        )

    def forward(self, states: LatentTrajectory) -> FutureLaw:
        """Predict an FP32 latent displacement mean from causal online states."""
        if type(states) is not LatentTrajectory:
            msg = "future law requires a LatentTrajectory"
            raise TypeError(msg)
        if states.width != self.latent_width:
            msg = f"latent state width {states.width} does not match future-law width {self.latent_width}"
            raise ValueError(msg)
        with torch.autocast(device_type=states.values.device.type, enabled=False):
            mean = self.predictor(states.values.float())
        return FutureLaw(mean=mean, codec_identity=states.codec_identity)

    def loss(
        self,
        states: LatentTrajectory,
        target: FutureTarget | StabilizedTargets,
        *,
        mean_baseline: MeanDisplacementBaseline,
    ) -> FutureLawLoss:
        """Score a target with full ``0.5 * ((Y-mu)^2 + log(2*pi))`` NLL."""
        if type(target) not in (FutureTarget, StabilizedTargets):
            msg = "future-law loss requires FutureTarget or StabilizedTargets"
            raise TypeError(msg)
        if type(mean_baseline) is not MeanDisplacementBaseline:
            msg = "future-law loss requires a MeanDisplacementBaseline"
            raise TypeError(msg)
        law = self(states)
        if law.codec_identity != target.codec_identity:
            msg = "future-law state and target codec identities must match"
            raise ValueError(msg)
        if target.width != self.latent_width:
            msg = f"future target width {target.width} does not match future-law width {self.latent_width}"
            raise ValueError(msg)
        if law.mean.shape != target.values.shape:
            msg = (
                f"future-law mean shape {tuple(law.mean.shape)} does not match target shape "
                f"{tuple(target.values.shape)}"
            )
            raise ValueError(msg)
        mean_baseline.validate_for(target)
        with torch.autocast(device_type=law.mean.device.type, enabled=False):
            target32 = target.values.float()
            residual = target32 - law.mean
            nll = 0.5 * (residual.square() + _LOG_TWO_PI).mean()
            zero_nll = 0.5 * (target32.square() + _LOG_TWO_PI).mean()
            baseline_residual = target32 - mean_baseline.mean.view(1, 1, -1)
            mean_nll = 0.5 * (baseline_residual.square() + _LOG_TWO_PI).mean()
            residual_mean = residual.mean()
            residual_rms = residual.square().mean().sqrt()
            residual_variance = residual.var(unbiased=False)
            coverage_50 = (residual.abs() <= _MARGINAL_GAUSSIAN_Z["50"]).float().mean()
            coverage_80 = (residual.abs() <= _MARGINAL_GAUSSIAN_Z["80"]).float().mean()
            coverage_95 = (residual.abs() <= _MARGINAL_GAUSSIAN_Z["95"]).float().mean()
        return FutureLawLoss(
            nll=nll,
            zero_displacement_nll=zero_nll,
            mean_displacement_nll=mean_nll,
            residual_mean=residual_mean,
            residual_rms=residual_rms,
            residual_variance=residual_variance,
            coverage_50=coverage_50,
            coverage_80=coverage_80,
            coverage_95=coverage_95,
            mean=law.mean,
        )
