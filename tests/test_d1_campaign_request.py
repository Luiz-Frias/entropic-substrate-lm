"""Strict command request for D1 campaign execution and capture recovery."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import d1_launch_admission_fixtures as fixtures
from entropic_substrate.ladder.register.d1_registration_codec import (
    parse_registered_design_declaration,
    serialize_registered_design,
)
from entropic_substrate.ladder.run.campaign.d1_campaign_request import (
    D1CampaignCommandRequest,
    D1CampaignLocations,
    D1CampaignRequestError,
    D1FreshCampaignCursor,
    D1PersistedCampaignCursor,
    parse_d1_campaign_request,
    serialize_d1_campaign_request,
)

from .test_d1_campaign_queue import _campaign

pytestmark = pytest.mark.unit


def _request(tmp_path: Path, *, persisted: bool = True) -> D1CampaignCommandRequest:
    design, intents, _starts = _campaign()
    declaration = parse_registered_design_declaration(serialize_registered_design(design))
    contract = fixtures.contract(fixtures.family(), seeds=(101,))
    locations = D1CampaignLocations(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        lifecycle_path=tmp_path / "campaign" / "lifecycle.jsonl",
        checkpoint_path=tmp_path / "campaign" / "checkpoint.json",
        finalized_cell_custody_root=tmp_path / "custody" / "cells",
        capture_preparation_custody_root=tmp_path / "custody" / "capture",
    )
    cursor = D1PersistedCampaignCursor("9" * 64) if persisted else D1FreshCampaignCursor()
    return D1CampaignCommandRequest(declaration, contract, intents, cursor, locations)


def test_campaign_request_round_trip_preserves_registration_roster_and_cursor(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    payload = serialize_d1_campaign_request(request)

    restored = parse_d1_campaign_request(payload, catalog=fixtures.profiles())

    assert restored == request
    assert serialize_d1_campaign_request(restored) == payload
    assert restored.design.design_digest == request.intents[0].registered_design_digest
    assert restored.cursor == D1PersistedCampaignCursor("9" * 64)
    assert tuple((item.plan.profile.arm, item.plan.seed) for item in restored.intents) == tuple(
        (item.plan.profile.arm, item.plan.seed) for item in request.intents
    )


def test_fresh_cursor_is_nominal_and_forbids_a_checkpoint_digest(tmp_path: Path) -> None:
    request = _request(tmp_path, persisted=False)
    payload = serialize_d1_campaign_request(request)

    restored = parse_d1_campaign_request(payload, catalog=fixtures.profiles())

    assert type(restored.cursor) is D1FreshCampaignCursor
    assert json.loads(payload)["cursor"] == {"kind": "fresh"}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value["locations"].pop("lifecycle_path"),
        lambda value: value["cursor"].update({"sha256": "bad"}),
        lambda value: value["intents"].append(value["intents"][0]),
        lambda value: value["intents"][0].update({"registered_design_digest": "0" * 64}),
        lambda value: value["decision_contract"].update({"verdict_vocabulary": "wrong"}),
    ],
)
def test_parser_refuses_unknown_missing_or_cross_authority_drift(
    tmp_path: Path,
    mutation: object,
) -> None:
    value = json.loads(serialize_d1_campaign_request(_request(tmp_path)))
    mutation(value)  # type: ignore[operator]

    with pytest.raises(D1CampaignRequestError):
        parse_d1_campaign_request(
            json.dumps(value, separators=(",", ":"), sort_keys=True),
            catalog=fixtures.profiles(),
        )


def test_locations_refuse_overlapping_or_nul_paths(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(ValueError, match="distinct"):
        replace(
            request.locations,
            checkpoint_path=request.locations.lifecycle_path,
        )
    with pytest.raises(ValueError, match="NUL"):
        replace(request.locations, repo_root=Path("bad\x00path"))


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "[]",
        '{"schema_version":NaN}',
        '{"schema_version":"x","schema_version":"x"}',
    ],
)
def test_parser_refuses_blank_nonobject_nonfinite_and_duplicate_json(
    payload: str,
) -> None:
    with pytest.raises(D1CampaignRequestError):
        parse_d1_campaign_request(payload, catalog=fixtures.profiles())


def test_parser_refuses_noncanonical_json_and_missing_catalog(tmp_path: Path) -> None:
    payload = serialize_d1_campaign_request(_request(tmp_path))

    with pytest.raises(D1CampaignRequestError, match="canonical"):
        parse_d1_campaign_request(" " + payload, catalog=fixtures.profiles())
    with pytest.raises(D1CampaignRequestError, match="profile"):
        parse_d1_campaign_request(payload, catalog=())
