"""Canonical transport envelopes for the external D1 analysis boundary.

The envelopes deliberately contain no container-engine spelling or host path.
They can cross a CLI, OCI, or other process boundary without changing the
scientific request identity.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, NoReturn, cast

__all__ = [
    "D1ExternalAnalysisRequestEnvelope",
    "D1ExternalAnalysisResponseEnvelope",
    "decode_d1_external_analysis_request",
    "decode_d1_external_analysis_response",
    "encode_d1_external_analysis_request",
    "encode_d1_external_analysis_response",
]

_REQUEST_SCHEMA: Final = "d1-external-analysis-request-v1"
_RESPONSE_SCHEMA: Final = "d1-external-analysis-response-v1"
_REQUEST_FIELDS: Final = frozenset(
    {"analysis_request_digest", "isolation_request_digest", "schema_version"}
)
_RESPONSE_FIELDS: Final = frozenset(
    {
        "analysis_request_digest",
        "analysis_sha256",
        "isolation_request_digest",
        "schema_version",
    }
)


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"D1 external analysis envelope has duplicate key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    raise ValueError(f"D1 external analysis envelope has non-finite constant {value!r}")


def _canonical(manifest: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(manifest),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decode(payload: bytes, *, fields: frozenset[str], schema: str) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError("D1 external analysis envelope must be bytes")
    try:
        value: object = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"D1 external analysis envelope is invalid UTF-8 JSON: {error}") from error
    if type(value) is not dict:
        raise ValueError("D1 external analysis envelope must be a JSON object")
    manifest = cast(dict[str, object], value)
    if set(manifest) != set(fields):
        raise ValueError("D1 external analysis envelope fields must be exact")
    if manifest["schema_version"] != schema:
        raise ValueError(f"D1 external analysis envelope schema must be {schema!r}")
    if _canonical(manifest) != payload:
        raise ValueError("D1 external analysis envelope must use canonical JSON bytes")
    return manifest


@dataclass(frozen=True, slots=True)
class D1ExternalAnalysisRequestEnvelope:
    analysis_request_digest: str
    isolation_request_digest: str

    def __post_init__(self) -> None:
        _sha256(self.analysis_request_digest, label="analysis_request_digest")
        _sha256(self.isolation_request_digest, label="isolation_request_digest")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "analysis_request_digest": self.analysis_request_digest,
            "isolation_request_digest": self.isolation_request_digest,
            "schema_version": _REQUEST_SCHEMA,
        }


@dataclass(frozen=True, slots=True)
class D1ExternalAnalysisResponseEnvelope:
    analysis_request_digest: str
    isolation_request_digest: str
    analysis_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.analysis_request_digest, label="analysis_request_digest")
        _sha256(self.isolation_request_digest, label="isolation_request_digest")
        _sha256(self.analysis_sha256, label="analysis_sha256")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "analysis_request_digest": self.analysis_request_digest,
            "analysis_sha256": self.analysis_sha256,
            "isolation_request_digest": self.isolation_request_digest,
            "schema_version": _RESPONSE_SCHEMA,
        }


def encode_d1_external_analysis_request(value: D1ExternalAnalysisRequestEnvelope) -> bytes:
    if type(value) is not D1ExternalAnalysisRequestEnvelope:
        raise TypeError("D1 external analysis request codec requires its nominal value")
    return _canonical(value.manifest)


def decode_d1_external_analysis_request(payload: bytes) -> D1ExternalAnalysisRequestEnvelope:
    manifest = _decode(payload, fields=_REQUEST_FIELDS, schema=_REQUEST_SCHEMA)
    return D1ExternalAnalysisRequestEnvelope(
        analysis_request_digest=_sha256(
            manifest["analysis_request_digest"], label="analysis_request_digest"
        ),
        isolation_request_digest=_sha256(
            manifest["isolation_request_digest"], label="isolation_request_digest"
        ),
    )


def encode_d1_external_analysis_response(value: D1ExternalAnalysisResponseEnvelope) -> bytes:
    if type(value) is not D1ExternalAnalysisResponseEnvelope:
        raise TypeError("D1 external analysis response codec requires its nominal value")
    return _canonical(value.manifest)


def decode_d1_external_analysis_response(payload: bytes) -> D1ExternalAnalysisResponseEnvelope:
    manifest = _decode(payload, fields=_RESPONSE_FIELDS, schema=_RESPONSE_SCHEMA)
    return D1ExternalAnalysisResponseEnvelope(
        analysis_request_digest=_sha256(
            manifest["analysis_request_digest"], label="analysis_request_digest"
        ),
        isolation_request_digest=_sha256(
            manifest["isolation_request_digest"], label="isolation_request_digest"
        ),
        analysis_sha256=_sha256(manifest["analysis_sha256"], label="analysis_sha256"),
    )
