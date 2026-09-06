"""Registration preparation binds scientific arithmetic to captured bytes."""

import hashlib
import json

import pytest

from entropic_substrate.analysis.ladder_registration import (
    read_statistical_design,
)

pytestmark = pytest.mark.unit


def specification() -> dict[str, object]:
    return {
        "schema": "d-ladder-statistical-design-v1",
        "run_class": "pilot",
        "registered_design_digest": None,
        "family_digest": "4" * 64,
        "construction_digests": {"both": "3" * 64, "sg": "5" * 64},
        "seeds": [7, 8],
        "contrasts": [
            {
                "contrast_id": "both-vs-sg",
                "terms": {"both": 1.0, "sg": -1.0},
                "metric": "nats_per_byte",
                "units": "nats/byte",
                "eval_steps": [1, 2],
                "source_bytes": 8,
                "kind": "superiority",
                "expected_effect": 0.01,
                "lower_bound": 0.0,
                "upper_bound": None,
            }
        ],
    }


def sealed(value: dict[str, object]) -> tuple[bytes, str]:
    payload = json.dumps(value, allow_nan=False).encode()
    return payload, hashlib.sha256(payload).hexdigest()


def test_design_binds_exact_bytes_and_explicit_endpoint() -> None:
    payload, digest = sealed(specification())
    design = read_statistical_design(payload, expected_sha256=digest)
    assert design.source_sha256 == digest
    assert design.seeds == (7, 8)
    assert design.contrasts[0].eval_steps == (1, 2)
    assert design.contrasts[0].hypothesis.expected_effect == 0.01
    with pytest.raises(ValueError, match="digest"):
        read_statistical_design(payload + b" ", expected_sha256=digest)


@pytest.mark.parametrize("key", ["eval_steps", "units", "expected_effect", "lower_bound"])
def test_missing_scientific_choices_never_default(key: str) -> None:
    value = specification()
    del value["contrasts"][0][key]
    with pytest.raises(ValueError):
        read_statistical_design(sealed(value)[0], expected_sha256=sealed(value)[1])


@pytest.mark.parametrize(
    "key,value",
    [
        ("eval_steps", []),
        ("eval_steps", [2, 1]),
        ("eval_steps", [1, 1]),
        ("source_bytes", True),
        ("terms", {"both": 1, "sg": 1}),
        ("terms", {"both": 1, "unknown": -1}),
        ("expected_effect", True),
        ("kind", "equivalence"),
    ],
)
def test_invalid_endpoint_or_hypothesis_refused(key: str, value: object) -> None:
    data = specification()
    data["contrasts"][0][key] = value
    with pytest.raises(ValueError):
        read_statistical_design(sealed(data)[0], expected_sha256=sealed(data)[1])
