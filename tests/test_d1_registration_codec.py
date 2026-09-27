"""Canonical inverse codec for the complete registered D1 decision surface."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from d1_verdict_v2_fixtures import decision_contract
from entropic_substrate.ladder.register.d1_registered_design import decision_contract_manifest
from entropic_substrate.ladder.register.d1_registration_codec import (
    D1RegistrationCodecError,
    decision_contract_from_mapping,
    decision_contract_to_mapping,
    parse_decision_contract,
    serialize_decision_contract,
)

pytestmark = pytest.mark.unit


def test_decision_contract_canonical_round_trip_preserves_every_emitted_field() -> None:
    original = decision_contract()
    mapping = decision_contract_to_mapping(original)
    payload = serialize_decision_contract(original)
    restored = parse_decision_contract(payload)

    assert mapping == decision_contract_manifest(original)
    assert restored == original
    assert decision_contract_to_mapping(restored) == mapping
    assert payload == json.dumps(
        mapping,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _top_unknown(value: dict[str, object]) -> None:
    value["invented"] = True


def _top_missing(value: dict[str, object]) -> None:
    value.pop("verdict_vocabulary")


def _nested_unknown(value: dict[str, object]) -> None:
    value["contrast_rules"]["sg-main"]["invented"] = True  # type: ignore[index]


def _nested_missing(value: dict[str, object]) -> None:
    value["contrast_rules"]["sg-main"]["endpoint"].pop("metric")  # type: ignore[index,union-attr]


def _term_drift(value: dict[str, object]) -> None:
    value["contrast_rules"]["sg-main"]["terms"][0]["coefficient"] = 0.0  # type: ignore[index]


def _clause_drift(value: dict[str, object]) -> None:
    value["clause_rules"]["language"].pop("registered_rule")  # type: ignore[index,union-attr]


def _exclusion_drift(value: dict[str, object]) -> None:
    value["exclusion_rules"][0]["scope"] = ""  # type: ignore[index]


def _decision_drift(value: dict[str, object]) -> None:
    value["decision_rules"]["SUPPORTED"].pop("eligible_strategies")  # type: ignore[index,union-attr]


def _family_drift(value: dict[str, object]) -> None:
    value["family_identity"]["arm_construction_digests"].pop("both")  # type: ignore[index,union-attr]


@pytest.mark.parametrize(
    "mutate",
    [
        _top_unknown,
        _top_missing,
        _nested_unknown,
        _nested_missing,
        _term_drift,
        _clause_drift,
        _exclusion_drift,
        _decision_drift,
        _family_drift,
    ],
)
def test_mapping_decoder_rejects_unknown_missing_or_semantically_invalid_fields(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    mapping = decision_contract_to_mapping(decision_contract())
    mutate(mapping)

    with pytest.raises(D1RegistrationCodecError):
        decision_contract_from_mapping(mapping)


@pytest.mark.parametrize(
    "payload",
    [
        '{"arm_ids":[],"arm_ids":[]}',
        '{"schema_version":NaN}',
        " {}",
        "{}\n",
        "[]",
        "\udcff",
    ],
)
def test_text_decoder_rejects_duplicate_nonfinite_noncanonical_or_nonobject_json(
    payload: str,
) -> None:
    with pytest.raises(D1RegistrationCodecError):
        parse_decision_contract(payload)


def test_mapping_decoder_rejects_numeric_or_collection_coercion() -> None:
    mapping = decision_contract_to_mapping(decision_contract())
    mapping["contrast_rules"]["sg-main"]["terms"][0]["coefficient"] = 1  # type: ignore[index]
    with pytest.raises(D1RegistrationCodecError, match="canonical"):
        decision_contract_from_mapping(mapping)

    mapping = decision_contract_to_mapping(decision_contract())
    mapping["arm_ids"] = tuple(mapping["arm_ids"])  # type: ignore[arg-type]
    with pytest.raises(D1RegistrationCodecError, match="array"):
        decision_contract_from_mapping(mapping)


def test_codec_requires_nominal_contract_and_mapping_root() -> None:
    with pytest.raises(TypeError, match="D1DecisionContract"):
        serialize_decision_contract(object())  # type: ignore[arg-type]
    with pytest.raises(D1RegistrationCodecError, match="object"):
        decision_contract_from_mapping([])
