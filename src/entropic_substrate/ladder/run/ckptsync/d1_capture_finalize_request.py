"""Strict D1 capture-finalize request over one COMMIT_REQUIRED artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final, NoReturn, cast

from entropic_substrate.kernel.identity.d1_primitives import canonical_json
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.register.d1_config import D1LaunchProfile
from entropic_substrate.ladder.run.campaign.d1_campaign_request import (
    D1CampaignCommandRequest,
    D1CampaignRequestError,
    parse_d1_campaign_request,
)

__all__ = [
    "D1CaptureFinalizeRequest",
    "D1CaptureFinalizeRequestError",
    "parse_d1_capture_finalize_request",
    "serialize_d1_capture_finalize_request",
]

_SCHEMA: Final = "d1-capture-finalize-request-v1"
_ROOT_KEYS: Final = frozenset({"campaign", "preparation_artifact", "schema_version"})
_ARTIFACT_KEYS: Final = frozenset({"path", "sha256"})


class D1CaptureFinalizeRequestError(ValueError):
    """Finalize authority is malformed, substituted, or noncanonical."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1CaptureFinalizeRequestError(message)
    raise D1CaptureFinalizeRequestError(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"D1 capture-finalize {label} must be an object with string keys")
    return cast(dict[str, object], value)


def _exact(value: dict[str, object], keys: frozenset[str], *, label: str) -> None:
    if frozenset(value) != keys:
        _fail(f"D1 capture-finalize {label} keys must be exact")


@dataclass(frozen=True, slots=True)
class D1CaptureFinalizeRequest:
    campaign: D1CampaignCommandRequest
    preparation_artifact: ArtifactIdentity

    def __post_init__(self) -> None:
        if type(self.campaign) is not D1CampaignCommandRequest:
            raise TypeError("capture-finalize request requires campaign request")
        if type(self.preparation_artifact) is not ArtifactIdentity:
            raise TypeError("capture-finalize request requires preparation artifact")
        if self.preparation_artifact.content_hash.algorithm != "sha256":
            raise ValueError("capture-finalize preparation requires SHA-256 identity")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "campaign": self.campaign.manifest,
            "preparation_artifact": {
                "path": self.preparation_artifact.path,
                "sha256": self.preparation_artifact.content_hash.digest,
            },
            "schema_version": _SCHEMA,
        }


def capture_finalize_request_from_mapping(
    value: object,
    *,
    catalog: tuple[D1LaunchProfile, ...],
) -> D1CaptureFinalizeRequest:
    root = _mapping(value, label="root")
    _exact(root, _ROOT_KEYS, label="root")
    if root["schema_version"] != _SCHEMA:
        _fail("D1 capture-finalize schema_version is unsupported")
    artifact = _mapping(root["preparation_artifact"], label="preparation artifact")
    _exact(artifact, _ARTIFACT_KEYS, label="preparation artifact")
    try:
        campaign = parse_d1_campaign_request(
            canonical_json(root["campaign"]),
            catalog=catalog,
        )
        result = D1CaptureFinalizeRequest(
            campaign,
            ArtifactIdentity(
                cast(str, artifact["path"]),
                ContentHash("sha256", cast(str, artifact["sha256"])),
            ),
        )
    except D1CaptureFinalizeRequestError:
        raise
    except (D1CampaignRequestError, TypeError, ValueError) as error:
        _fail(f"D1 capture-finalize authority is invalid: {error}", error)
    if result.manifest != root:
        _fail("D1 capture-finalize manifest differs from recomputed authority")
    return result


def serialize_d1_capture_finalize_request(request: D1CaptureFinalizeRequest) -> str:
    if type(request) is not D1CaptureFinalizeRequest:
        raise TypeError("capture-finalize serializer requires its nominal request")
    return canonical_json(request.manifest)


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate D1 capture-finalize key {key!r}")
        result[key] = value
    return result


def parse_d1_capture_finalize_request(
    payload: str,
    *,
    catalog: tuple[D1LaunchProfile, ...],
) -> D1CaptureFinalizeRequest:
    if type(payload) is not str or not payload:
        _fail("D1 capture-finalize payload must be nonempty text")
    try:
        value: object = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda constant: _fail(
                f"non-finite D1 capture-finalize constant {constant}"
            ),
        )
    except D1CaptureFinalizeRequestError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        _fail("D1 capture-finalize payload is invalid JSON", error)
    request = capture_finalize_request_from_mapping(value, catalog=catalog)
    if canonical_json(value) != payload:
        _fail("D1 capture-finalize JSON is noncanonical")
    return request
