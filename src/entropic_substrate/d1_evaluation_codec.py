"""Canonical process-boundary codec for the complete D1 evaluation value."""

from __future__ import annotations

import json
from typing import NoReturn, cast

from entropic_substrate.d1_evaluation import D1EvalResult

__all__ = [
    "D1EvaluationCodecError",
    "d1_evaluation_from_mapping",
    "d1_evaluation_to_mapping",
    "parse_d1_evaluation",
    "serialize_d1_evaluation",
]

_SCHEMA = "d1-evaluation-wire-v1"
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
_INT_FIELDS = ("numerical_rank", "batch_count", "online_positions", "residual_elements")
_VECTOR_FIELDS = ("state_variance", "state_covariance_eigenvalues")
_KEYS = frozenset(
    {
        "schema_version",
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
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail("D1 evaluation must be a JSON object with string keys")
    return cast(dict[str, object], value)


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


def d1_evaluation_to_mapping(result: D1EvalResult) -> dict[str, object]:
    """Return the exact JSON-compatible process record, including dense covariance."""
    if type(result) is not D1EvalResult:
        raise TypeError("D1 evaluation codec requires D1EvalResult")
    mapping: dict[str, object] = {
        field: getattr(result, field) for field in (*_FLOAT_FIELDS, *_INT_FIELDS)
    }
    mapping.update(
        {
            "schema_version": _SCHEMA,
            "state_covariance_eigenvalues": list(result.state_covariance_eigenvalues),
            "state_sample_covariance": [list(row) for row in result.state_sample_covariance],
            "state_variance": list(result.state_variance),
        }
    )
    return mapping


def d1_evaluation_from_mapping(value: object) -> D1EvalResult:
    """Reconstruct the nominal evaluation only after exact wire admission."""
    root = _mapping(value)
    if frozenset(root) != _KEYS:
        _fail("D1 evaluation keys must be exact")
    if root["schema_version"] != _SCHEMA:
        _fail(f"D1 evaluation schema_version must be {_SCHEMA}")
    try:
        return D1EvalResult(
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
            coverage_50=_floating(root["coverage_50"], field="coverage_50"),
            coverage_80=_floating(root["coverage_80"], field="coverage_80"),
            coverage_95=_floating(root["coverage_95"], field="coverage_95"),
            state_sample_covariance=_matrix(root["state_sample_covariance"]),
            state_variance=_vector(root["state_variance"], field="state_variance"),
            state_covariance_eigenvalues=_vector(
                root["state_covariance_eigenvalues"],
                field="state_covariance_eigenvalues",
            ),
            effective_rank=_floating(root["effective_rank"], field="effective_rank"),
            numerical_rank=_integer(root["numerical_rank"], field="numerical_rank"),
            mean_pairwise_cosine_distance=_floating(
                root["mean_pairwise_cosine_distance"],
                field="mean_pairwise_cosine_distance",
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


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"D1 evaluation contains duplicate key {key!r}")
        result[key] = value
    return result


def parse_d1_evaluation(payload: bytes) -> D1EvalResult:
    if type(payload) is not bytes or not payload:
        _fail("D1 evaluation payload must be nonempty bytes")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda constant: _fail(f"non-finite JSON constant {constant}"),
        )
    except D1EvaluationCodecError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("D1 evaluation payload must be UTF-8 JSON", error)
    result = d1_evaluation_from_mapping(value)
    if serialize_d1_evaluation(result) != payload:
        _fail("D1 evaluation payload must use canonical JSON")
    return result
