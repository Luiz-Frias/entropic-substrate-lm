"""The worker request carries its run authority: v1 for smoke (unchanged bytes), v2 for science.

Rung B (operator ruling 2026-10-27, "option B"). A smoke request renders exactly the
pre-rung-B ``h-probe-worker-request-v1`` bytes, so every captured ``request_sha256``
still re-derives; a scientific request renders ``h-probe-worker-request-v2`` with its
sealed ``RunAuthority`` and names the authority's digest as both of its authorities.
"""

import json
from dataclasses import replace

import pytest
from tests.test_h_probe_launch import pilot_request, remote_request
from tests.test_run_authority import authority as sealed_authority
from tests.test_sentence_worker_request import request

from entropic_substrate.ladder.run.worker_request import (
    REQUEST_SCHEMA_V1,
    REQUEST_SCHEMA_V2,
    parse_probe_request,
)

pytestmark = pytest.mark.unit


def _reencode(row: dict[str, object]) -> bytes:
    return json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


@pytest.mark.parametrize("build", [request, remote_request])
def test_a_smoke_request_renders_v1_with_no_authority_key(build: object) -> None:
    value = build()  # type: ignore[operator]
    assert value.authority is None
    assert value.manifest["schema"] == REQUEST_SCHEMA_V1
    assert "authority" not in value.manifest
    assert value.manifest["scientific_authority"] == value.manifest["spending_authority"] == "none"
    assert parse_probe_request(value.payload, max_bytes=1_000_000) == value


def test_a_pilot_request_renders_v2_naming_its_authority_and_roundtrips() -> None:
    value = pilot_request()
    assert value.authority is not None
    assert value.manifest["schema"] == REQUEST_SCHEMA_V2
    assert value.manifest["authority"] == value.authority.manifest
    assert value.manifest["scientific_authority"] == value.authority.digest
    assert value.manifest["spending_authority"] == value.authority.digest
    assert parse_probe_request(value.payload, max_bytes=1_000_000) == value
    assert value.prefix == f"h-probes/v1/{value.nonce}"


def test_the_authority_binds_the_request_digest() -> None:
    base = pilot_request()
    other = pilot_request(spend_cap=replace(sealed_authority().spend_cap, usd=41.0))
    assert base.digest != other.digest


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("v1_pilot", "scientific run requires a sealed run authority"),
        ("v2_smoke", "smoke run classes carry no run authority"),
        ("v2_missing_authority", "fields differ"),
        ("v1_with_authority", "fields differ"),
        ("mandate_none", "unknown worker request scope or authority"),
        ("mandate_other", "unknown worker request scope or authority"),
        ("authority_tamper", "run authority digest does not match its content"),
        ("authority_other_stage", "run authority names another stage"),
        ("schema_v3", "unknown worker request scope or authority"),
    ],
)
def test_the_request_reader_refuses_a_mandate_out_of_family(mutation: str, message: str) -> None:
    pilot = pilot_request()
    assert pilot.authority is not None
    if mutation == "v1_pilot":
        row = pilot.manifest
        del row["authority"]
        row["schema"] = REQUEST_SCHEMA_V1
        row["scientific_authority"] = row["spending_authority"] = "none"
    elif mutation == "v2_smoke":
        row = remote_request().manifest
        sealed = sealed_authority()
        row["schema"] = REQUEST_SCHEMA_V2
        row["authority"] = sealed.manifest
        row["scientific_authority"] = row["spending_authority"] = sealed.digest
    elif mutation == "v2_missing_authority":
        row = pilot.manifest
        del row["authority"]
    elif mutation == "v1_with_authority":
        row = pilot.manifest
        row["schema"] = REQUEST_SCHEMA_V1
    elif mutation == "mandate_none":
        row = pilot.manifest
        row["spending_authority"] = "none"
    elif mutation == "mandate_other":
        row = pilot.manifest
        row["scientific_authority"] = sealed_authority(stage="H2").digest
    elif mutation == "authority_tamper":
        row = pilot.manifest
        authority = dict(pilot.authority.manifest)
        authority["gpu_class"] = "NVIDIA H100 80GB HBM3"
        row["authority"] = authority
    elif mutation == "authority_other_stage":
        row = pilot.manifest
        sealed = sealed_authority(stage="H2")
        row["authority"] = sealed.manifest
        row["scientific_authority"] = row["spending_authority"] = sealed.digest
    else:
        row = remote_request().manifest
        row["schema"] = "h-probe-worker-request-v3"
    with pytest.raises(ValueError, match=message):
        parse_probe_request(_reencode(row), max_bytes=1_000_000)


def test_a_pilot_request_without_authority_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="scientific run requires a sealed run authority"):
        replace(pilot_request(), authority=None)


def test_a_smoke_request_with_an_authority_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="smoke run classes carry no run authority"):
        replace(remote_request(), authority=sealed_authority())
