"""Registered training and evidence protocol shared by every D1 arm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from entropic_substrate.experiments.d1_primitives import EvidenceClass, canonical_digest

__all__ = [
    "BaselineSampling",
    "CalibrationSplit",
    "D1AmpPolicy",
    "D1BaselineCalibrationSpec",
    "D1BatchingSpec",
    "D1EvaluationSpec",
    "D1ExecutionProtocol",
    "D1OptimizationSpec",
    "D1ResourceProtocol",
    "OptimizerKind",
    "TrainerWallOrigin",
    "WeightDecayPolicy",
    "default_d1_execution_protocol",
]


class OptimizerKind(StrEnum):
    ADAMW = "adamw"


class WeightDecayPolicy(StrEnum):
    MATRIX_ONLY = "matrix_only"


class D1AmpPolicy(StrEnum):
    BF16 = "bf16"
    OFF = "off"


class CalibrationSplit(StrEnum):
    TRAIN = "train"


class BaselineSampling(StrEnum):
    DETERMINISTIC_TRAIN_GRID_V1 = "deterministic_train_grid_v1"


class TrainerWallOrigin(StrEnum):
    AFTER_PREFLIGHT = "after_preflight_before_construction"


def _positive_float(value: float, *, label: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) <= 0.0:
        msg = f"{label} must be a finite positive scalar"
        raise ValueError(msg)


def _nonnegative_float(value: float, *, label: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0.0:
        msg = f"{label} must be a finite nonnegative scalar"
        raise ValueError(msg)


def _positive_int(value: int, *, label: str) -> None:
    if type(value) is not int or value < 1:
        msg = f"{label} must be a positive integer"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class D1OptimizationSpec:
    """One exact AdamW, clipping, schedule, and precision construction."""

    learning_rate: float
    min_learning_rate_ratio: float
    warmup_steps: int
    max_steps: int
    weight_decay: float
    betas: tuple[float, float]
    epsilon: float
    grad_clip_norm: float
    amp: D1AmpPolicy
    compile: bool
    optimizer: OptimizerKind = OptimizerKind.ADAMW
    weight_decay_policy: WeightDecayPolicy = WeightDecayPolicy.MATRIX_ONLY

    def __post_init__(self) -> None:
        _positive_float(self.learning_rate, label="learning_rate")
        if (
            type(self.min_learning_rate_ratio) not in (int, float)
            or not math.isfinite(float(self.min_learning_rate_ratio))
            or not 0.0 < float(self.min_learning_rate_ratio) <= 1.0
        ):
            msg = "min_learning_rate_ratio must be finite and in (0, 1]"
            raise ValueError(msg)
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            msg = "warmup_steps must be a nonnegative integer"
            raise ValueError(msg)
        _positive_int(self.max_steps, label="max_steps")
        if self.warmup_steps > self.max_steps:
            msg = "warmup_steps cannot exceed max_steps"
            raise ValueError(msg)
        _nonnegative_float(self.weight_decay, label="weight_decay")
        if (
            type(self.betas) is not tuple
            or len(self.betas) != 2
            or any(
                type(beta) not in (int, float)
                or not math.isfinite(float(beta))
                or not 0.0 <= float(beta) < 1.0
                for beta in self.betas
            )
        ):
            msg = "betas must be two finite scalars in [0, 1)"
            raise ValueError(msg)
        _positive_float(self.epsilon, label="epsilon")
        _positive_float(self.grad_clip_norm, label="grad_clip_norm")
        if type(self.amp) is not D1AmpPolicy:
            msg = "D1 amp must be BF16 or off"
            raise ValueError(msg)
        if type(self.compile) is not bool:
            msg = "compile must be a bool"
            raise ValueError(msg)
        if self.optimizer is not OptimizerKind.ADAMW:
            msg = "D1 optimizer must be AdamW"
            raise ValueError(msg)
        if self.weight_decay_policy is not WeightDecayPolicy.MATRIX_ONLY:
            msg = "D1 weight decay must use the matrix-only policy"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "amp": self.amp.value,
            "betas": list(self.betas),
            "compile": self.compile,
            "epsilon": float(self.epsilon),
            "grad_clip_norm": float(self.grad_clip_norm),
            "learning_rate": float(self.learning_rate),
            "max_steps": self.max_steps,
            "min_learning_rate_ratio": float(self.min_learning_rate_ratio),
            "optimizer": self.optimizer.value,
            "warmup_steps": self.warmup_steps,
            "weight_decay": float(self.weight_decay),
            "weight_decay_policy": self.weight_decay_policy.value,
        }


@dataclass(frozen=True, slots=True)
class D1BatchingSpec:
    """Training exposure and data-order accounting used for quality curves."""

    batch_size: int
    sequence_length: int
    horizon: int = 1
    data_order_digest_batches: int = 32
    quality_exposure: str = "committed_online_positions_v1"

    def __post_init__(self) -> None:
        _positive_int(self.batch_size, label="batch_size")
        _positive_int(self.sequence_length, label="sequence_length")
        if type(self.horizon) is not int or self.horizon != 1:
            msg = "D1 batching requires horizon=1"
            raise ValueError(msg)
        if self.data_order_digest_batches != 32:
            msg = "D1 data-order digest window is fixed at 32 batches"
            raise ValueError(msg)
        if self.quality_exposure != "committed_online_positions_v1":
            msg = "D1 quality exposure uses committed online positions"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "data_order_digest_batches": self.data_order_digest_batches,
            "horizon": self.horizon,
            "quality_exposure": self.quality_exposure,
            "sequence_length": self.sequence_length,
        }


@dataclass(frozen=True, slots=True)
class D1BaselineCalibrationSpec:
    """Single-use pre-optimization mean comparator calibration."""

    split: CalibrationSplit = CalibrationSplit.TRAIN
    num_batches: int = 32
    sampling: BaselineSampling = BaselineSampling.DETERMINISTIC_TRAIN_GRID_V1
    fit_once_before_optimization: bool = True

    def __post_init__(self) -> None:
        if self.split is not CalibrationSplit.TRAIN:
            msg = "D1 baseline calibration requires the training split"
            raise ValueError(msg)
        if self.num_batches != 32:
            msg = "D1 baseline calibration requires 32 batches"
            raise ValueError(msg)
        if self.sampling is not BaselineSampling.DETERMINISTIC_TRAIN_GRID_V1:
            msg = "D1 baseline calibration requires deterministic train-grid sampling"
            raise ValueError(msg)
        if self.fit_once_before_optimization is not True:
            msg = "D1 baseline calibration must fit once before optimization"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "fit_once_before_optimization": self.fit_once_before_optimization,
            "num_batches": self.num_batches,
            "sampling": self.sampling.value,
            "split": self.split.value,
        }


@dataclass(frozen=True, slots=True)
class D1EvaluationSpec:
    """Deterministic evaluation, logging, checkpoint, and terminal cadence."""

    num_batches: int
    eval_interval: int
    log_interval: int
    save_interval: int
    evaluate_at_step_zero: bool = True
    terminal_eval_required: bool = True

    def __post_init__(self) -> None:
        _positive_int(self.num_batches, label="num_batches")
        _positive_int(self.eval_interval, label="eval_interval")
        _positive_int(self.log_interval, label="log_interval")
        if type(self.save_interval) is not int or self.save_interval < 0:
            msg = "save_interval must be a nonnegative integer"
            raise ValueError(msg)
        if self.evaluate_at_step_zero is not True:
            msg = "D1 requires a step-zero evaluation"
            raise ValueError(msg)
        if self.terminal_eval_required is not True:
            msg = "D1 requires a terminal evaluation"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "eval_interval": self.eval_interval,
            "evaluate_at_step_zero": self.evaluate_at_step_zero,
            "log_interval": self.log_interval,
            "num_batches": self.num_batches,
            "save_interval": self.save_interval,
            "terminal_eval_required": self.terminal_eval_required,
        }


@dataclass(frozen=True, slots=True)
class D1ResourceProtocol:
    """Evidence semantics, deliberately separate from meter availability."""

    trainer_wall_origin: TrainerWallOrigin = TrainerWallOrigin.AFTER_PREFLIGHT
    quality_exposure: str = "committed_online_positions_v1"
    total_work: str = "include_target_calibration_eval_v1"
    optional_meter_absence: str = "typed_evidence_gap_v1"
    required_evidence: frozenset[EvidenceClass] = frozenset({EvidenceClass.MEASURED})

    def __post_init__(self) -> None:
        if self.trainer_wall_origin is not TrainerWallOrigin.AFTER_PREFLIGHT:
            msg = "D1 trainer wall time starts after preflight and before construction"
            raise ValueError(msg)
        if self.quality_exposure != "committed_online_positions_v1":
            msg = "D1 resource quality exposure semantics changed"
            raise ValueError(msg)
        if self.total_work != "include_target_calibration_eval_v1":
            msg = "D1 total-work accounting semantics changed"
            raise ValueError(msg)
        if self.optional_meter_absence != "typed_evidence_gap_v1":
            msg = "D1 unavailable resource meters require typed evidence gaps"
            raise ValueError(msg)
        if self.required_evidence != frozenset({EvidenceClass.MEASURED}):
            msg = "D1 requires measured trainer wall, source-byte, and token counters"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "optional_meter_absence": self.optional_meter_absence,
            "quality_exposure": self.quality_exposure,
            "required_evidence": sorted(item.value for item in self.required_evidence),
            "total_work": self.total_work,
            "trainer_wall_origin": self.trainer_wall_origin.value,
        }


@dataclass(frozen=True, slots=True)
class D1ExecutionProtocol:
    """The non-arm protocol that makes a D1 family a poolable study."""

    optimization: D1OptimizationSpec
    batching: D1BatchingSpec
    baseline: D1BaselineCalibrationSpec
    evaluation: D1EvaluationSpec
    resources: D1ResourceProtocol
    schema_version: str = "d1-execution-protocol-v1"

    def __post_init__(self) -> None:
        expected = (
            (self.optimization, D1OptimizationSpec),
            (self.batching, D1BatchingSpec),
            (self.baseline, D1BaselineCalibrationSpec),
            (self.evaluation, D1EvaluationSpec),
            (self.resources, D1ResourceProtocol),
        )
        if any(type(value) is not value_type for value, value_type in expected):
            msg = "D1 execution protocol requires validated component specs"
            raise TypeError(msg)
        if self.schema_version != "d1-execution-protocol-v1":
            msg = "unsupported D1 execution protocol schema"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.manifest,
            "batching": self.batching.manifest,
            "evaluation": self.evaluation.manifest,
            "optimization": self.optimization.manifest,
            "resources": self.resources.manifest,
            "schema_version": self.schema_version,
        }

    @property
    def protocol_digest(self) -> str:
        return canonical_digest(self.manifest)


def default_d1_execution_protocol() -> D1ExecutionProtocol:
    """Full-scale D1 protocol used when a caller accepts the registered defaults."""
    return D1ExecutionProtocol(
        optimization=D1OptimizationSpec(
            learning_rate=6e-4,
            min_learning_rate_ratio=0.1,
            warmup_steps=500,
            max_steps=20_000,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            epsilon=1e-8,
            grad_clip_norm=1.0,
            amp=D1AmpPolicy.BF16,
            compile=False,
        ),
        batching=D1BatchingSpec(batch_size=16, sequence_length=1024),
        baseline=D1BaselineCalibrationSpec(),
        evaluation=D1EvaluationSpec(
            num_batches=20,
            eval_interval=500,
            log_interval=20,
            save_interval=1000,
        ),
        resources=D1ResourceProtocol(),
    )
