"""Strict D1 capture-finalize request binds campaign and preparation custody."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import d1_launch_admission_fixtures as fixtures
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.ckptsync.d1_capture_finalize_request import (
    D1CaptureFinalizeRequest,
    D1CaptureFinalizeRequestError,
    parse_d1_capture_finalize_request,
    serialize_d1_capture_finalize_request,
)

from .test_d1_campaign_request import _request

pytestmark = pytest.mark.unit


def _finalize_request(tmp_path: Path) -> D1CaptureFinalizeRequest:
    return D1CaptureFinalizeRequest(
        campaign=_request(tmp_path),
        preparation_artifact=ArtifactIdentity(
            str(tmp_path / "custody" / "capture" / f"{'7' * 64}.json"),
            ContentHash("sha256", "7" * 64),
        ),
    )


def test_finalize_request_round_trip_preserves_campaign_and_preparation_identity(
    tmp_path: Path,
) -> None:
    request = _finalize_request(tmp_path)
    payload = serialize_d1_capture_finalize_request(request)

    restored = parse_d1_capture_finalize_request(payload, catalog=fixtures.profiles())

    assert restored == request
    assert serialize_d1_capture_finalize_request(restored) == payload
    assert restored.preparation_artifact.content_hash.digest == "7" * 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value.pop("preparation_artifact"),
        lambda value: value["preparation_artifact"].update({"sha256": "bad"}),
        lambda value: value["campaign"]["cursor"].update({"sha256": "bad"}),
    ],
)
def test_parser_refuses_missing_unknown_or_substituted_authority(
    tmp_path: Path,
    mutation: object,
) -> None:
    value = json.loads(serialize_d1_capture_finalize_request(_finalize_request(tmp_path)))
    mutation(value)  # type: ignore[operator]

    with pytest.raises(D1CaptureFinalizeRequestError):
        parse_d1_capture_finalize_request(
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
    with pytest.raises(D1CaptureFinalizeRequestError):
        parse_d1_capture_finalize_request(payload, catalog=fixtures.profiles())


def test_parser_refuses_noncanonical_json(tmp_path: Path) -> None:
    payload = serialize_d1_capture_finalize_request(_finalize_request(tmp_path))

    with pytest.raises(D1CaptureFinalizeRequestError, match="canonical"):
        parse_d1_capture_finalize_request(" " + payload, catalog=fixtures.profiles())
