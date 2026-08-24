"""Registered D1 optimization, sampling, evaluation, and resource protocol."""

from __future__ import annotations

from dataclasses import replace

import pytest

from entropic_substrate.experiments.d1_execution import (
    PAIRWISE_COSINE_SEMANTIC,
    BaselineSampling,
    CalibrationSplit,
    D1AmpPolicy,
    D1BaselineCalibrationSpec,
    D1BatchingSpec,
    D1EvaluationSpec,
    D1ExecutionProtocol,
    D1OptimizationSpec,
    D1ResourceProtocol,
    OptimizerKind,
    TrainerWallOrigin,
    WeightDecayPolicy,
)
from entropic_substrate.experiments.d1_primitives import EvidenceClass, canonical_digest

pytestmark = pytest.mark.unit


def _protocol() -> D1ExecutionProtocol:
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


def test_complete_execution_protocol_has_a_canonical_registered_digest() -> None:
    protocol = _protocol()

    assert protocol.optimization.optimizer is OptimizerKind.ADAMW
    assert protocol.optimization.weight_decay_policy is WeightDecayPolicy.MATRIX_ONLY
    assert protocol.batching.horizon == 1
    assert protocol.baseline == D1BaselineCalibrationSpec(
        split=CalibrationSplit.TRAIN,
        num_batches=32,
        sampling=BaselineSampling.DETERMINISTIC_TRAIN_GRID_V1,
        fit_once_before_optimization=True,
    )
    assert protocol.evaluation.evaluate_at_step_zero is True
    assert protocol.evaluation.terminal_eval_required is True
    assert protocol.evaluation.pairwise_cosine_semantic == PAIRWISE_COSINE_SEMANTIC
    assert protocol.resources.trainer_wall_origin is TrainerWallOrigin.AFTER_PREFLIGHT
    assert protocol.resources.required_evidence == frozenset({EvidenceClass.MEASURED})
    assert protocol.protocol_digest == canonical_digest(protocol.manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("learning_rate", 0.0, "learning_rate"),
        ("min_learning_rate_ratio", 1.1, "min_learning_rate_ratio"),
        ("warmup_steps", -1, "warmup_steps"),
        ("max_steps", 0, "max_steps"),
        ("weight_decay", -0.1, "weight_decay"),
        ("betas", (0.9, 1.0), "betas"),
        ("epsilon", float("nan"), "epsilon"),
        ("grad_clip_norm", float("inf"), "grad_clip_norm"),
    ],
)
def test_optimization_spec_refuses_invalid_numerics(
    field: str, value: object, message: str
) -> None:
    kwargs = {
        "learning_rate": 6e-4,
        "min_learning_rate_ratio": 0.1,
        "warmup_steps": 500,
        "max_steps": 20_000,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "epsilon": 1e-8,
        "grad_clip_norm": 1.0,
        "amp": D1AmpPolicy.BF16,
        "compile": False,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        D1OptimizationSpec(**kwargs)  # type: ignore[arg-type]


def test_fp16_and_unregistered_optimizer_policies_are_not_representable() -> None:
    assert set(D1AmpPolicy) == {D1AmpPolicy.BF16, D1AmpPolicy.OFF}
    assert set(OptimizerKind) == {OptimizerKind.ADAMW}
    assert set(WeightDecayPolicy) == {WeightDecayPolicy.MATRIX_ONLY}


@pytest.mark.parametrize(
    "spec",
    [
        lambda: D1BatchingSpec(batch_size=0, sequence_length=8),
        lambda: D1BatchingSpec(batch_size=2, sequence_length=0),
        lambda: D1BatchingSpec(batch_size=2, sequence_length=8, horizon=2),
        lambda: D1BatchingSpec(batch_size=2, sequence_length=8, data_order_digest_batches=31),
        lambda: D1BaselineCalibrationSpec(num_batches=31),
        lambda: D1BaselineCalibrationSpec(fit_once_before_optimization=False),
        lambda: D1EvaluationSpec(num_batches=0, eval_interval=5, log_interval=1, save_interval=0),
        lambda: D1EvaluationSpec(
            num_batches=1,
            eval_interval=5,
            log_interval=1,
            save_interval=0,
            evaluate_at_step_zero=False,
        ),
        lambda: D1EvaluationSpec(
            num_batches=1,
            eval_interval=5,
            log_interval=1,
            save_interval=0,
            terminal_eval_required=False,
        ),
        lambda: D1EvaluationSpec(
            num_batches=1,
            eval_interval=5,
            log_interval=1,
            save_interval=0,
            pairwise_cosine_semantic="batch_mean_cosine",
        ),
    ],
)
def test_fixed_execution_contracts_reject_axis_drift(spec) -> None:
    with pytest.raises(ValueError):
        spec()


def test_every_execution_change_changes_the_protocol_digest() -> None:
    protocol = _protocol()

    changed = replace(
        protocol,
        optimization=replace(protocol.optimization, learning_rate=3e-4),
    )
    assert changed.protocol_digest != protocol.protocol_digest


def test_quality_exposure_and_total_work_are_distinct_named_semantics() -> None:
    resources = D1ResourceProtocol()

    assert resources.quality_exposure == "committed_online_positions_v1"
    assert resources.total_work == "include_target_calibration_eval_v1"
    assert resources.optional_meter_absence == "typed_evidence_gap_v1"
