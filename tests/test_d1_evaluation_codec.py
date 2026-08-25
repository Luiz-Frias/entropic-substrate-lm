"""The complete D1 evaluation survives the trainer process boundary."""

from __future__ import annotations

import json

import pytest

from entropic_substrate.d1_evaluation import D1EvalResult
from entropic_substrate.d1_evaluation_codec import (
    d1_evaluation_from_mapping,
    d1_evaluation_to_mapping,
    parse_d1_evaluation,
    serialize_d1_evaluation,
)

pytestmark = pytest.mark.unit


def _result() -> D1EvalResult:
    language = 0.8
    future = 0.91
    zero = 1.02
    mean = 0.97
    return D1EvalResult(
        language_loss=language,
        nats_per_byte=language,
        bits_per_byte=language / 0.6931471805599453,
        perplexity=2.225540928492468,
        future_nll=future,
        zero_displacement_nll=zero,
        mean_displacement_nll=mean,
        zero_baseline_improvement=zero - future,
        mean_baseline_improvement=mean - future,
        residual_mean=0.0,
        residual_rms=1.03,
        residual_variance=0.87,
        coverage_50=0.49,
        coverage_80=0.79,
        coverage_95=0.94,
        state_sample_covariance=((0.5, 0.1), (0.1, 1.5)),
        state_variance=(0.5, 1.5),
        state_covariance_eigenvalues=(0.4900980486407215, 1.5099019513592786),
        effective_rank=1.73,
        numerical_rank=2,
        mean_pairwise_cosine_distance=0.71,
        raw_variance_penalty=0.011,
        weighted_variance_penalty=0.0011,
        raw_covariance_penalty=0.022,
        weighted_covariance_penalty=0.0022,
        batch_count=3,
        online_positions=8,
        residual_elements=16,
    )


def test_full_evaluation_round_trips_canonical_bytes_without_dropping_covariance() -> None:
    expected = _result()

    payload = serialize_d1_evaluation(expected)
    observed = parse_d1_evaluation(payload)

    assert observed == expected
    assert observed.state_sample_covariance == expected.state_sample_covariance
    mapping = json.loads(payload)
    assert mapping["schema_version"] == "d1-evaluation-wire-v1"
    assert mapping["state_sample_covariance"] == [[0.5, 0.1], [0.1, 1.5]]
    assert d1_evaluation_from_mapping(d1_evaluation_to_mapping(expected)) == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"extra": True}), "keys"),
        (lambda value: value.update({"future_nll": 1}), "future_nll"),
        (lambda value: value.update({"numerical_rank": True}), "numerical_rank"),
        (lambda value: value.update({"state_variance": [0.5, 1]}), "state_variance"),
        (
            lambda value: value.update({"state_sample_covariance": [[0.5], [0.1, 1.5]]}),
            "covariance",
        ),
    ],
)
def test_mapping_refuses_schema_scalar_integer_vector_and_matrix_drift(
    mutation: object,
    message: str,
) -> None:
    value = d1_evaluation_to_mapping(_result())
    mutation(value)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        d1_evaluation_from_mapping(value)


def test_parser_refuses_duplicate_nonfinite_and_noncanonical_json() -> None:
    canonical = serialize_d1_evaluation(_result())
    duplicate = canonical.replace(
        b'{"batch_count":',
        b'{"batch_count":3,"batch_count":',
        1,
    )
    nonfinite = canonical.replace(b'"future_nll":0.91', b'"future_nll":NaN', 1)
    indented = json.dumps(json.loads(canonical), indent=2).encode()

    with pytest.raises(ValueError, match="duplicate"):
        parse_d1_evaluation(duplicate)
    with pytest.raises(ValueError, match="non-finite"):
        parse_d1_evaluation(nonfinite)
    with pytest.raises(ValueError, match="canonical"):
        parse_d1_evaluation(indented)
