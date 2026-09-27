"""Canonical process-boundary codec for registered D1 launch intents."""

from __future__ import annotations

import json

import pytest

import d1_launch_admission_fixtures as fixtures
from entropic_substrate.ladder.run.campaign.d1_operator_intent_codec import (
    D1OperatorIntentCodecError,
    parse_d1_confirmatory_operator_intent,
    serialize_d1_operator_intent,
)

from .test_d1_campaign_queue import _campaign

pytestmark = pytest.mark.unit


def _intent():  # type: ignore[no-untyped-def]
    return _campaign()[1][0]


def test_confirmatory_intent_round_trip_preserves_exact_manifest() -> None:
    intent = _intent()
    payload = serialize_d1_operator_intent(intent)

    restored = parse_d1_confirmatory_operator_intent(
        payload,
        catalog=fixtures.profiles(),
    )

    assert restored == intent
    assert serialize_d1_operator_intent(restored) == payload
    assert json.loads(payload) == intent.manifest


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value.pop("execution_segment_id"),
        lambda value: value.update({"plan_digest": "0" * 64}),
        lambda value: value.update(
            {"image_reference": "registry.example/wrong@sha256:" + "0" * 64}
        ),
        lambda value: value["resource"].update({"accelerator_form_factor": "pcie"}),
        lambda value: value.update({"segment_start_mode": "resume"}),
    ],
)
def test_parser_refuses_missing_unknown_or_internally_inconsistent_fields(
    mutation: object,
) -> None:
    value = json.loads(serialize_d1_operator_intent(_intent()))
    mutation(value)  # type: ignore[operator]

    with pytest.raises(D1OperatorIntentCodecError):
        parse_d1_confirmatory_operator_intent(
            json.dumps(value, separators=(",", ":"), sort_keys=True),
            catalog=fixtures.profiles(),
        )


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "[]",
        '{"schema_version":NaN}',
        '{"schema_version":"x","schema_version":"x"}',
    ],
)
def test_parser_refuses_blank_nonobject_nonfinite_and_duplicate_json(payload: str) -> None:
    with pytest.raises(D1OperatorIntentCodecError):
        parse_d1_confirmatory_operator_intent(payload, catalog=fixtures.profiles())


def test_parser_refuses_noncanonical_json_and_wrong_profile_catalog() -> None:
    payload = serialize_d1_operator_intent(_intent())

    with pytest.raises(D1OperatorIntentCodecError, match="canonical"):
        parse_d1_confirmatory_operator_intent(" " + payload, catalog=fixtures.profiles())
    with pytest.raises(D1OperatorIntentCodecError, match="profile"):
        parse_d1_confirmatory_operator_intent(payload, catalog=())
