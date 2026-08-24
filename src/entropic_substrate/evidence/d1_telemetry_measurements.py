"""Strict D1 TRAIN and EVAL payload admission."""

from __future__ import annotations

import math
from typing import Final, cast

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

__all__ = ["parse_eval_event", "parse_quality", "parse_train_event"]

_TRAIN_KEYS: Final = {
    "covariance_penalty",
    "coverage_50",
    "coverage_80",
    "coverage_95",
    "effective_rank",
    "future_nll",
    "gradient_norm",
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


def parse_train_event(event: D1JournalEvent) -> tuple[str, int]:
    from entropic_substrate.d1_training_step import D1TrainStepResult

    payload = require_mapping(event.manifest["payload"], path="train.payload")
    exact_fields(payload, {"data_order", "scheduled_log_boundary", "train"}, path="train.payload")
    if type(payload["scheduled_log_boundary"]) is not bool:
        raise gap("invalid_train", "train.scheduled_log_boundary", "must be a bool")
    train = require_mapping(payload["train"], path="train.train")
    exact_fields(train, _TRAIN_KEYS, path="train.train")
    try:
        result = D1TrainStepResult(**train)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise gap("invalid_train", "train.train", str(error)) from error
    if result.manifest != train or result.successful_step != event.successful_step:
        raise gap("train_step", "train.train", "does not match the enclosing successful step")
    order = require_mapping(payload["data_order"], path="train.data_order")
    exact_fields(order, {"batches_attested", "sha256"}, path="train.data_order")
    return require_sha256(order["sha256"], path="train.data_order.sha256"), positive_int(
        order["batches_attested"], path="train.data_order.batches_attested"
    )
