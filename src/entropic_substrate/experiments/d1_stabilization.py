"""Future-law, geometry-control, and stabilization contracts for D1."""

from __future__ import annotations

import math
from dataclasses import dataclass

from entropic_substrate.experiments.d1_primitives import (
    CovarianceNormalization,
    EMAKind,
    FutureActivation,
    FutureLawFamily,
    GeometryConsumer,
    PredictiveMetric,
)


@dataclass(frozen=True, slots=True)
class D1TargetSpec:
    """Fixed one-position RMS-normalized latent displacement target."""

    horizon: int = 1
    latent_displacement: bool = True
    rms_normalized: bool = True

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or self.horizon != 1:
            msg = "D1 target requires horizon=1"
            raise ValueError(msg)
        if self.latent_displacement is not True:
            msg = "D1 target requires a latent displacement"
            raise ValueError(msg)
        if self.rms_normalized is not True:
            msg = "D1 target requires RMS-normalized target states"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "horizon": self.horizon,
            "latent_displacement": self.latent_displacement,
            "rms_normalized": self.rms_normalized,
        }


@dataclass(frozen=True, slots=True)
class FutureLawSpec:
    """D1's nonlinear diagonal Gaussian future law with fixed variance."""

    predictor_hidden_width: int = 64
    activation: FutureActivation = FutureActivation.SILU
    family: FutureLawFamily = FutureLawFamily.NONLINEAR_FIXED_VARIANCE_GAUSSIAN
    variance: float = 1.0
    future_loss_weight: float = 1.0
    reduce_per_latent_dimension: bool = True
    mean_baseline_calibration_batches: int = 32
    mean_baseline_sampling: str = "deterministic_train_grid_v1"

    def __post_init__(self) -> None:
        if type(self.predictor_hidden_width) is not int or self.predictor_hidden_width < 1:
            msg = "predictor_hidden_width must be a positive integer"
            raise ValueError(msg)
        if self.activation is not FutureActivation.SILU:
            msg = "D1 future law requires SiLU activation"
            raise ValueError(msg)
        if self.family is not FutureLawFamily.NONLINEAR_FIXED_VARIANCE_GAUSSIAN:
            msg = "D1 requires the nonlinear fixed-variance Gaussian future law"
            raise ValueError(msg)
        if type(self.variance) not in (int, float) or float(self.variance) != 1.0:
            msg = "D1 future law requires fixed variance=1 (sigma^2=1)"
            raise ValueError(msg)
        if (
            type(self.future_loss_weight) not in (int, float)
            or float(self.future_loss_weight) != 1.0
        ):
            msg = "D1 future law requires future_loss_weight=1"
            raise ValueError(msg)
        if self.reduce_per_latent_dimension is not True:
            msg = "D1 future law requires per-latent-dimension reduction"
            raise ValueError(msg)
        if (
            type(self.mean_baseline_calibration_batches) is not int
            or self.mean_baseline_calibration_batches != 32
        ):
            msg = "D1 mean-baseline calibration batches are fixed at 32"
            raise ValueError(msg)
        if self.mean_baseline_sampling != "deterministic_train_grid_v1":
            msg = "D1 mean-baseline calibration requires deterministic train-grid sampling"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "activation": self.activation.value,
            "family": self.family.value,
            "future_loss_weight": float(self.future_loss_weight),
            "mean_baseline_calibration_batches": self.mean_baseline_calibration_batches,
            "mean_baseline_sampling": self.mean_baseline_sampling,
            "predictor_hidden_width": self.predictor_hidden_width,
            "reduce_per_latent_dimension": self.reduce_per_latent_dimension,
            "variance": float(self.variance),
        }


@dataclass(frozen=True, slots=True)
class D1GeometrySpec:
    """Metric/consumer declaration validated as Euclidean/off by the D1 draft."""

    metric: PredictiveMetric = PredictiveMetric.EUCLIDEAN
    consumer: GeometryConsumer = GeometryConsumer.OFF

    @property
    def manifest(self) -> dict[str, object]:
        return {"consumer": self.consumer.value, "metric": self.metric.value}


@dataclass(frozen=True, slots=True)
class EMAParameters:
    """Registered target-teacher schedule constants for SG-containing arms."""

    kind: EMAKind = EMAKind.FIXED
    decay: float = 0.99
    final_decay: float | None = None
    schedule_steps: int | None = None
    updates_after_successful_optimizer_step: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.decay) not in (int, float)
            or not math.isfinite(float(self.decay))
            or not 0.0 < float(self.decay) < 1.0
        ):
            msg = "EMA decay must be strictly between zero and one"
            raise ValueError(msg)
        if self.updates_after_successful_optimizer_step is not True:
            msg = "D1 EMA must update only after a successful optimizer step"
            raise ValueError(msg)
        if self.kind is EMAKind.FIXED:
            if self.final_decay is not None or self.schedule_steps is not None:
                msg = "fixed EMA forbids final_decay and schedule_steps"
                raise ValueError(msg)
        elif self.kind is EMAKind.COSINE_SCHEDULE:
            if self.final_decay is None or self.schedule_steps is None:
                msg = "scheduled EMA requires final_decay and schedule_steps"
                raise ValueError(msg)
            if (
                type(self.final_decay) not in (int, float)
                or not math.isfinite(float(self.final_decay))
                or not 0.0 < float(self.final_decay) < 1.0
            ):
                msg = "EMA final_decay must be strictly between zero and one"
                raise ValueError(msg)
            if type(self.schedule_steps) is not int or self.schedule_steps < 1:
                msg = "EMA schedule_steps must be a positive integer"
                raise ValueError(msg)
        else:
            msg = "EMA kind must be fixed or cosine_schedule"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "decay": float(self.decay),
            "final_decay": None if self.final_decay is None else float(self.final_decay),
            "kind": self.kind.value,
            "schedule_steps": self.schedule_steps,
            "updates_after_successful_optimizer_step": self.updates_after_successful_optimizer_step,
        }


@dataclass(frozen=True, slots=True)
class CollapseParameters:
    """Moment anti-collapse constants using the implemented VICReg conventions."""

    std_floor: float = 1.0
    variance_weight: float = 1.0
    covariance_weight: float = 0.04
    variance_epsilon: float = 1e-4
    covariance_normalization: CovarianceNormalization = (
        CovarianceNormalization.OFF_DIAGONAL_SUM_PER_DIMENSION
    )

    def __post_init__(self) -> None:
        values = {
            "std_floor": self.std_floor,
            "variance_weight": self.variance_weight,
            "covariance_weight": self.covariance_weight,
            "variance_epsilon": self.variance_epsilon,
        }
        if any(
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in values.values()
        ):
            msg = "D1 collapse parameters must be strictly positive finite scalars"
            raise ValueError(msg)
        if (
            self.covariance_normalization
            is not CovarianceNormalization.OFF_DIAGONAL_SUM_PER_DIMENSION
        ):
            msg = "D1 requires off-diagonal sum-squared per-dimension covariance normalization"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "covariance_normalization": self.covariance_normalization.value,
            "covariance_weight": float(self.covariance_weight),
            "variance_epsilon": float(self.variance_epsilon),
            "std_floor": float(self.std_floor),
            "variance_weight": float(self.variance_weight),
        }
