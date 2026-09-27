"""Scientific evals must attest their actual common training-offset prefix."""

import pytest

from entropic_substrate.ladder.analyze.ladder_analysis import parse_scientific_data_order

pytestmark = pytest.mark.unit


def test_prefix_attestation_is_required_after_step_zero() -> None:
    assert parse_scientific_data_order(None, 0) is None
    assert parse_scientific_data_order(["a" * 64, 2], 2) == ("a" * 64, 2)
    assert parse_scientific_data_order(["b" * 64, 32], 40) == ("b" * 64, 32)


@pytest.mark.parametrize(
    ("value", "step"),
    [
        (None, 1),
        (["a" * 64, 1], 0),
        (["a" * 64, 1], 2),
        (["a" * 64, True], 1),
        (["bad", 1], 1),
        (["a" * 64, 33], 33),
    ],
)
def test_missing_partial_or_invalid_prefix_is_rejected(value: object, step: int) -> None:
    with pytest.raises(ValueError):
        parse_scientific_data_order(value, step)
