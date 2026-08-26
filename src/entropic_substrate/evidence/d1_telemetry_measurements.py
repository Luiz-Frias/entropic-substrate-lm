"""Strict D1 TRAIN and EVAL payload admission."""

from __future__ import annotations

import math
from typing import Final, cast

from entropic_substrate.d1_training_step import (
    D1ComponentGradientEvidence,
    D1GradientEvidenceKind,
    D1GradientReachability,
    D1TrainStepResult,
)
from entropic_substrate.evidence.d1_journal_values import D1JournalEvent
from entropic_substrate.evidence.d1_telemetry_payloads import (
    exact_fields,
    positive_int,
    require_mapping,
    require_sha256,
)
from entropic_substrate.evidence.d1_telemetry_values import (
    D1_QUALITY_FIELDS,
    D1EvalEvidence,
    D1QualityEvidence,
    gap,
)
from entropic_substrate.evidence.resource_work_codec import (
    ResourceWorkCodecGap,
    resource_snapshot_from_mapping,
)
from entropic_substrate.experiments.d1_primitives import TargetUpdateStrategy

__all__ = ["parse_eval_event", "parse_quality", "parse_train_event"]

_TRAIN_KEYS: Final = {
    "covariance_penalty",
    "coverage_50",
    "coverage_80",
    "coverage_95",
    "effective_rank",
    "future_nll",
    "gradient_norm",
    "gradient_reachability",
    "language_loss",
    "learning_rate",
    "mean_displacement_nll",
    "numerical_rank",
    "objective",
    "raw_covariance_penalty",
    "raw_variance_penalty",
    "residual_mean",
    "residual_rms",
    "residual_variance",
    "successful_step",
    "variance_penalty",
    "zero_displacement_nll",
}

_GRADIENT_COMPONENTS: Final = {
    "future_law",
    "language_readout",
    "online_branch",
    "target_branch",
}


def parse_quality(value: object) -> D1QualityEvidence:
    item = require_mapping(value, path="eval.quality")
    exact_fields(item, D1_QUALITY_FIELDS, path="eval.quality")

    def number(field: str) -> float:
        value = item[field]
        if type(value) is not float or not math.isfinite(value):
            raise gap("invalid_metric", f"eval.quality.{field}", "must be a finite float")
        return value

    def integer(field: str) -> int:
        value = item[field]
        if type(value) is not int:
            raise gap("invalid_metric", f"eval.quality.{field}", "must be an integer")
        return value

    def vector(field: str) -> tuple[float, ...]:
        value = item[field]
        if type(value) is not list or any(type(entry) is not float for entry in value):
            raise gap("invalid_vector", f"eval.quality.{field}", "must be a float list")
        return tuple(cast(list[float], value))

    scalar_fields = D1_QUALITY_FIELDS - {
        "batch_count",
        "numerical_rank",
        "online_positions",
        "pairwise_cosine_semantic",
        "residual_elements",
        "state_covariance_eigenvalues",
        "state_variance",
    }
    values = {field: number(field) for field in scalar_fields}
    semantic = item["pairwise_cosine_semantic"]
    if type(semantic) is not str:
        raise gap("invalid_metric", "eval.quality.pairwise_cosine_semantic", "must be a string")
    return D1QualityEvidence(
        **values,  # type: ignore[arg-type]
        batch_count=integer("batch_count"),
        numerical_rank=integer("numerical_rank"),
        online_positions=integer("online_positions"),
        residual_elements=integer("residual_elements"),
        pairwise_cosine_semantic=semantic,
        state_covariance_eigenvalues=vector("state_covariance_eigenvalues"),
        state_variance=vector("state_variance"),
    )


def parse_eval_event(event: D1JournalEvent, expected_index: int) -> D1EvalEvidence:
    payload = require_mapping(event.manifest["payload"], path="eval.payload")
    exact_fields(payload, {"eval_index", "quality", "resource_work"}, path="eval.payload")
    index = payload["eval_index"]
    if type(index) is not int or index != expected_index:
        raise gap("eval_index", "eval.eval_index", "must be consecutive from zero")
    try:
        resource = resource_snapshot_from_mapping(payload["resource_work"])
    except ResourceWorkCodecGap as error:
        raise gap("invalid_resource", "eval.resource_work", str(error)) from error
    if (
        resource.eval_index != index
        or resource.training_step != event.successful_step
        or resource.logical_run_id != event.logical_run_id
    ):
        raise gap("resource_cursor", "eval.resource_work", "does not match enclosing event")
    return D1EvalEvidence(event.successful_step, index, parse_quality(payload["quality"]), resource)


def parse_train_event(
    event: D1JournalEvent,
    *,
    expected_target_update: TargetUpdateStrategy,
) -> tuple[str, int, D1TrainStepResult]:
    if type(expected_target_update) is not TargetUpdateStrategy:
        raise TypeError("D1 TRAIN admission requires a registered target update strategy")
    payload = require_mapping(event.manifest["payload"], path="train.payload")
    exact_fields(payload, {"data_order", "scheduled_log_boundary", "train"}, path="train.payload")
    if type(payload["scheduled_log_boundary"]) is not bool:
        raise gap("invalid_train", "train.scheduled_log_boundary", "must be a bool")
    train = require_mapping(payload["train"], path="train.train")
    exact_fields(train, _TRAIN_KEYS, path="train.train")
    reachability = require_mapping(
        train["gradient_reachability"], path="train.train.gradient_reachability"
    )
    exact_fields(reachability, _GRADIENT_COMPONENTS, path="train.train.gradient_reachability")

    def component(name: str) -> D1ComponentGradientEvidence:
        value = require_mapping(
            reachability[name], path=f"train.train.gradient_reachability.{name}"
        )
        exact_fields(
            value,
            {"kind", "pre_clip_l2_norm"},
            path=f"train.train.gradient_reachability.{name}",
        )
        raw_kind = value["kind"]
        raw_norm = value["pre_clip_l2_norm"]
        if type(raw_kind) is not str or (raw_norm is not None and type(raw_norm) is not float):
            raise gap(
                "invalid_train",
                f"train.train.gradient_reachability.{name}",
                "kind must be a string and pre_clip_l2_norm must be a float or null",
            )
        try:
            kind = D1GradientEvidenceKind(raw_kind)
            return D1ComponentGradientEvidence(kind, raw_norm)
        except (TypeError, ValueError) as error:
            raise gap(
                "invalid_train", f"train.train.gradient_reachability.{name}", str(error)
            ) from error

    parsed_components = {name: component(name) for name in _GRADIENT_COMPONENTS}

    def number(name: str) -> float:
        value = train[name]
        if type(value) is not float:
            raise gap("invalid_train", f"train.train.{name}", "must be a float")
        return value

    def nonnegative_integer(name: str) -> int:
        value = train[name]
        if type(value) is not int or value < 0:
            raise gap("invalid_train", f"train.train.{name}", "must be a nonnegative integer")
        return value

    try:
        typed_reachability = D1GradientReachability(
            online_branch=parsed_components["online_branch"],
            target_branch=parsed_components["target_branch"],
            language_readout=parsed_components["language_readout"],
            future_law=parsed_components["future_law"],
        )
        result = D1TrainStepResult(
            successful_step=positive_int(
                train["successful_step"], path="train.train.successful_step"
            ),
            learning_rate=number("learning_rate"),
            objective=number("objective"),
            language_loss=number("language_loss"),
            future_nll=number("future_nll"),
            variance_penalty=number("variance_penalty"),
            covariance_penalty=number("covariance_penalty"),
            raw_variance_penalty=number("raw_variance_penalty"),
            raw_covariance_penalty=number("raw_covariance_penalty"),
            zero_displacement_nll=number("zero_displacement_nll"),
            mean_displacement_nll=number("mean_displacement_nll"),
            residual_mean=number("residual_mean"),
            residual_rms=number("residual_rms"),
            residual_variance=number("residual_variance"),
            coverage_50=number("coverage_50"),
            coverage_80=number("coverage_80"),
            coverage_95=number("coverage_95"),
            effective_rank=number("effective_rank"),
            numerical_rank=nonnegative_integer("numerical_rank"),
            gradient_norm=number("gradient_norm"),
            gradient_reachability=typed_reachability,
        )
    except (TypeError, ValueError) as error:
        raise gap("invalid_train", "train.train", str(error)) from error
    if result.manifest != train or result.successful_step != event.successful_step:
        raise gap("train_step", "train.train", "does not match the enclosing successful step")
    target_kind = result.gradient_reachability.target_branch.kind
    expected_kind = (
        D1GradientEvidenceKind.STRUCTURALLY_ABSENT_STOP_GRADIENT
        if expected_target_update is TargetUpdateStrategy.STOP_GRADIENT_EMA
        else D1GradientEvidenceKind.MEASURED_NONZERO
    )
    if target_kind is not expected_kind:
        raise gap(
            "target_gradient_posture",
            "train.train.gradient_reachability.target_branch.kind",
            "target gradient posture differs from the RUN_START arm",
        )
    order = require_mapping(payload["data_order"], path="train.data_order")
    exact_fields(order, {"batches_attested", "sha256"}, path="train.data_order")
    return (
        require_sha256(order["sha256"], path="train.data_order.sha256"),
        positive_int(order["batches_attested"], path="train.data_order.batches_attested"),
        result,
    )
