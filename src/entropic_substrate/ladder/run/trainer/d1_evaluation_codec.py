"""Canonical process-boundary codec for the complete D1 evaluation value."""

from __future__ import annotations

import json
from typing import NoReturn, cast

from entropic_substrate.kernel.codec.strict_json import require_mapping, strict_loads
from entropic_substrate.kernel.verdict.d1_evaluation_values import (
    D1ChannelMoments,
    D1CosineGap,
    parse_d1_cosine,
)
from entropic_substrate.ladder.run.trainer.d1_evaluation import D1EvalResult

__all__ = [
    "D1EvaluationCodecError",
    "d1_evaluation_from_mapping",
    "d1_evaluation_to_mapping",
    "parse_d1_evaluation",
    "serialize_d1_evaluation",
]

_SCHEMA = "d1-evaluation-wire-v2"
_GAP_SCHEMA = "d1-evaluation-wire-v3"
_FLOAT_FIELDS = (
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
    "residual_mean_l2_norm",
    "target_predicted_exact_match_fraction",
    "coverage_50",
    "coverage_80",
    "coverage_95",
    "raw_variance_penalty",
    "weighted_variance_penalty",
    "raw_covariance_penalty",
    "weighted_covariance_penalty",
)
_INT_FIELDS = (
    "batch_count",
    "online_positions",
    "residual_elements",
    "target_predicted_exact_match_count",
)
_VECTOR_FIELDS = ("residual_mean_vector",)
_CHANNEL_VECTOR_FIELDS = ("mean", "variance", "covariance_eigenvalues")
_CHANNEL_FLOAT_FIELDS = ("effective_rank", "rms_l2_norm")
_CHANNEL_INT_FIELDS = ("numerical_rank",)
_CHANNEL_KEYS = frozenset({*_CHANNEL_VECTOR_FIELDS, *_CHANNEL_FLOAT_FIELDS, *_CHANNEL_INT_FIELDS})
_KEYS = frozenset(
    {
        "schema_version",
        "mean_pairwise_cosine_distance",
        "channels",
        "state_sample_covariance",
        *_FLOAT_FIELDS,
        *_INT_FIELDS,
        *_VECTOR_FIELDS,
    }
)


class D1EvaluationCodecError(ValueError):
    """A D1 evaluation wire value is incomplete, permissive, or noncanonical."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1EvaluationCodecError(message)
    raise D1EvaluationCodecError(message) from error


def _mapping(value: object) -> dict[str, object]:
    return require_mapping(
        value, fail=lambda _reason: _fail("D1 evaluation must be a JSON object with string keys")
    )


def _floating(value: object, *, field: str) -> float:
    if type(value) is not float:
        _fail(f"D1 evaluation {field} must be a JSON float")
    return value


def _integer(value: object, *, field: str) -> int:
    if type(value) is not int:
        _fail(f"D1 evaluation {field} must be a JSON integer")
    return value


def _vector(value: object, *, field: str) -> tuple[float, ...]:
    if type(value) is not list or any(type(item) is not float for item in value):
        _fail(f"D1 evaluation {field} must be a JSON float array")
    return tuple(cast(list[float], value))


def _matrix(value: object) -> tuple[tuple[float, ...], ...]:
    if type(value) is not list or any(type(row) is not list for row in value):
        _fail("D1 evaluation covariance must be a JSON float matrix")
    rows = cast(list[list[object]], value)
    if any(any(type(item) is not float for item in row) for row in rows):
        _fail("D1 evaluation covariance must contain only JSON floats")
    return tuple(tuple(cast(list[float], row)) for row in rows)


def _channel_to_mapping(value: D1ChannelMoments) -> dict[str, object]:
    return {
        "covariance_eigenvalues": list(value.covariance_eigenvalues),
        "effective_rank": value.effective_rank,
        "mean": list(value.mean),
        "numerical_rank": value.numerical_rank,
        "rms_l2_norm": value.rms_l2_norm,
        "variance": list(value.variance),
    }


def _channel_from_mapping(value: object, *, field: str) -> D1ChannelMoments:
    root = _mapping(value)
    if frozenset(root) != _CHANNEL_KEYS:
        _fail(f"D1 evaluation {field} channel keys must be exact")
    return D1ChannelMoments(
        mean=_vector(root["mean"], field=f"{field}.mean"),
        variance=_vector(root["variance"], field=f"{field}.variance"),
        covariance_eigenvalues=_vector(
            root["covariance_eigenvalues"], field=f"{field}.covariance_eigenvalues"
        ),
        effective_rank=_floating(root["effective_rank"], field=f"{field}.effective_rank"),
        numerical_rank=_integer(root["numerical_rank"], field=f"{field}.numerical_rank"),
        rms_l2_norm=_floating(root["rms_l2_norm"], field=f"{field}.rms_l2_norm"),
    )


def d1_evaluation_to_mapping(result: D1EvalResult) -> dict[str, object]:
    """Return the exact JSON-compatible process record, including dense covariance."""
    if type(result) is not D1EvalResult:
        raise TypeError("D1 evaluation codec requires D1EvalResult")
    mapping: dict[str, object] = {
        field: getattr(result, field) for field in (*_FLOAT_FIELDS, *_INT_FIELDS)
    }
    mapping.update(
        {
            "channels": {
                "predicted": _channel_to_mapping(result.predicted_moments),
                "state": _channel_to_mapping(result.state_moments),
                "target": _channel_to_mapping(result.target_moments),
            },
            "residual_mean_vector": list(result.residual_mean_vector),
            "schema_version": (
                _GAP_SCHEMA
                if type(result.mean_pairwise_cosine_distance) is D1CosineGap
                else _SCHEMA
            ),
            "mean_pairwise_cosine_distance": result.telemetry_manifest[
                "mean_pairwise_cosine_distance"
            ],
            "state_sample_covariance": [list(row) for row in result.state_sample_covariance],
        }
    )
    return mapping


def d1_evaluation_from_mapping(value: object) -> D1EvalResult:
    """Reconstruct the nominal evaluation only after exact wire admission."""
    root = _mapping(value)
    if frozenset(root) != _KEYS:
        _fail("D1 evaluation keys must be exact")
    expected_schema = (
        _SCHEMA if type(root["mean_pairwise_cosine_distance"]) is float else _GAP_SCHEMA
    )
    if root["schema_version"] != expected_schema:
        _fail(f"D1 evaluation schema_version must be {expected_schema}")
    channels = _mapping(root["channels"])
    if frozenset(channels) != frozenset({"state", "target", "predicted"}):
        _fail("D1 evaluation channel identities must be exact")
    try:
        result = D1EvalResult(
            language_loss=_floating(root["language_loss"], field="language_loss"),
            nats_per_byte=_floating(root["nats_per_byte"], field="nats_per_byte"),
            bits_per_byte=_floating(root["bits_per_byte"], field="bits_per_byte"),
            perplexity=_floating(root["perplexity"], field="perplexity"),
            future_nll=_floating(root["future_nll"], field="future_nll"),
            zero_displacement_nll=_floating(
                root["zero_displacement_nll"], field="zero_displacement_nll"
            ),
            mean_displacement_nll=_floating(
                root["mean_displacement_nll"], field="mean_displacement_nll"
            ),
            zero_baseline_improvement=_floating(
                root["zero_baseline_improvement"], field="zero_baseline_improvement"
            ),
            mean_baseline_improvement=_floating(
                root["mean_baseline_improvement"], field="mean_baseline_improvement"
            ),
            residual_mean=_floating(root["residual_mean"], field="residual_mean"),
            residual_rms=_floating(root["residual_rms"], field="residual_rms"),
            residual_variance=_floating(root["residual_variance"], field="residual_variance"),
            target_predicted_exact_match_count=_integer(
                root["target_predicted_exact_match_count"],
                field="target_predicted_exact_match_count",
            ),
            coverage_50=_floating(root["coverage_50"], field="coverage_50"),
            coverage_80=_floating(root["coverage_80"], field="coverage_80"),
            coverage_95=_floating(root["coverage_95"], field="coverage_95"),
            state_sample_covariance=_matrix(root["state_sample_covariance"]),
            state_moments=_channel_from_mapping(channels["state"], field="state"),
            target_moments=_channel_from_mapping(channels["target"], field="target"),
            predicted_moments=_channel_from_mapping(channels["predicted"], field="predicted"),
            mean_pairwise_cosine_distance=parse_d1_cosine(
                root["mean_pairwise_cosine_distance"],
            ),
            raw_variance_penalty=_floating(
                root["raw_variance_penalty"], field="raw_variance_penalty"
            ),
            weighted_variance_penalty=_floating(
                root["weighted_variance_penalty"], field="weighted_variance_penalty"
            ),
            raw_covariance_penalty=_floating(
                root["raw_covariance_penalty"], field="raw_covariance_penalty"
            ),
            weighted_covariance_penalty=_floating(
                root["weighted_covariance_penalty"], field="weighted_covariance_penalty"
            ),
            batch_count=_integer(root["batch_count"], field="batch_count"),
            online_positions=_integer(root["online_positions"], field="online_positions"),
            residual_elements=_integer(root["residual_elements"], field="residual_elements"),
        )
        wire_residual_vector = _vector(root["residual_mean_vector"], field="residual_mean_vector")
        wire_residual_norm = _floating(root["residual_mean_l2_norm"], field="residual_mean_l2_norm")
        wire_exact_fraction = _floating(
            root["target_predicted_exact_match_fraction"],
            field="target_predicted_exact_match_fraction",
        )
        if wire_residual_vector != result.residual_mean_vector:
            _fail("D1 evaluation residual mean vector differs from channel means")
        if wire_residual_norm != result.residual_mean_l2_norm:
            _fail("D1 evaluation residual mean norm differs from channel means")
        if wire_exact_fraction != result.target_predicted_exact_match_fraction:
            _fail("D1 evaluation exact-match fraction differs from its count")
        return result
    except D1EvaluationCodecError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"D1 evaluation value is invalid: {error}", error)


def serialize_d1_evaluation(result: D1EvalResult) -> bytes:
    return json.dumps(
        d1_evaluation_to_mapping(result),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def parse_d1_evaluation(payload: bytes) -> D1EvalResult:
    if type(payload) is not bytes or not payload:
        _fail("D1 evaluation payload must be nonempty bytes")
    try:
        value = strict_loads(payload, fail=_fail)
    except D1EvaluationCodecError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("D1 evaluation payload must be UTF-8 JSON", error)
    result = d1_evaluation_from_mapping(value)
    if serialize_d1_evaluation(result) != payload:
        _fail("D1 evaluation payload must use canonical JSON")
    return result
