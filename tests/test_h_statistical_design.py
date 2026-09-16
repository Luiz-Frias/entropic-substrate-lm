"""H statistical declarations reuse arithmetic without relabelling D evidence."""

import hashlib
import json

import pytest
from tests.test_ladder_registration import sealed, specification

from entropic_substrate.analysis.ladder_registration import (
    read_h_statistical_design,
    read_statistical_design,
)

pytestmark = pytest.mark.unit


def h_specification() -> dict[str, object]:
    value = specification()
    value["schema"] = "h-ladder-statistical-design-v1"
    value["construction_digests"] = {"magnitude": "3" * 64, "sphere": "5" * 64}
    value["contrasts"][0].update(
        contrast_id="preservation",
        terms={"magnitude": -1.0, "sphere": 1.0},
        metric="registered_fixture_measure",
        units="fixture units",
        expected_effect=0.2,
    )
    return value


def test_h_design_binds_original_bytes_and_retains_explicit_scientific_choices() -> None:
    payload, digest = sealed(h_specification())
    design = read_h_statistical_design(payload, expected_sha256=digest)
    assert design.source_sha256 == hashlib.sha256(payload).hexdigest()
    assert design.run_class == "pilot" and design.registered_design_digest is None
    assert design.seeds == (7, 8)
    assert design.contrasts[0].hypothesis.units == "fixture units"
    assert design.contrasts[0].hypothesis.expected_effect == 0.2
    assert design.contrasts[0].terms == (("magnitude", -1.0), ("sphere", 1.0))
    with pytest.raises(ValueError, match="digest"):
        read_h_statistical_design(payload + b" ", expected_sha256=digest)


def test_d_and_h_readers_reject_each_others_namespace() -> None:
    for reader, value in (
        (read_statistical_design, h_specification()),
        (read_h_statistical_design, specification()),
    ):
        payload, digest = sealed(value)
        with pytest.raises(ValueError, match="schema"):
            reader(payload, expected_sha256=digest)


@pytest.mark.parametrize(
    "field",
    ["eval_steps", "units", "expected_effect", "lower_bound", "upper_bound", "source_bytes"],
)
def test_h_omitted_scientific_choices_never_default(field: str) -> None:
    value = h_specification()
    del value["contrasts"][0][field]
    payload, digest = sealed(value)
    with pytest.raises(ValueError):
        read_h_statistical_design(payload, expected_sha256=digest)


@pytest.mark.parametrize("run_class", ["local_tiny", "remote_tiny", "smoke"])
def test_h_tiny_declarations_cannot_become_scientific_inputs(run_class: str) -> None:
    payload, digest = sealed(h_specification() | {"run_class": run_class})
    with pytest.raises(ValueError):
        read_h_statistical_design(payload, expected_sha256=digest)


def test_h_confirmation_requires_registration_and_pilot_cannot_claim_it() -> None:
    for update in ({"run_class": "confirmatory"}, {"registered_design_digest": "7" * 64}):
        payload, digest = sealed(h_specification() | update)
        with pytest.raises(ValueError):
            read_h_statistical_design(payload, expected_sha256=digest)


def test_h_duplicate_json_policy_is_rejected_before_namespace_dispatch() -> None:
    payload = (
        b'{"schema":"h-ladder-statistical-design-v1","schema":"h-ladder-statistical-design-v1"}'
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_h_statistical_design(payload, expected_sha256=hashlib.sha256(payload).hexdigest())


def test_h_nonfinite_effect_is_rejected() -> None:
    value = h_specification()
    value["contrasts"][0]["expected_effect"] = float("nan")
    payload = json.dumps(value).encode()
    with pytest.raises(ValueError):
        read_h_statistical_design(payload, expected_sha256=hashlib.sha256(payload).hexdigest())
