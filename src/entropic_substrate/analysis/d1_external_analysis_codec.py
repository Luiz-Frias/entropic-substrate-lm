"""Canonical transport envelopes for the external D1 analysis boundary.

The envelopes deliberately contain no container-engine spelling or host path.
They can cross a CLI, OCI, or other process boundary without changing the
scientific request identity.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, NoReturn, cast

from entropic_substrate.analysis.d1_external_analysis_control import (
    D1ExternalAnalysisInputControl,
    decode_d1_external_analysis_input_control,
)
from entropic_substrate.experiments.d1_primitives import canonical_digest
from entropic_substrate.experiments.d1_registered_design import decision_contract_digest
from entropic_substrate.experiments.d1_registration_codec import (
    D1PreregistrationByteEvidence,
)

__all__ = [
    "D1ExternalAnalysisRequestEnvelope",
    "D1ExternalAnalysisRequestEnvelopeV2",
    "D1ExternalAnalysisResponseEnvelope",
    "D1ExternalAnalysisWorkerResponseV2",
    "build_d1_external_analysis_request_v2",
    "decode_d1_external_analysis_request",
    "decode_d1_external_analysis_request_v2",
    "decode_d1_external_analysis_response",
    "decode_d1_external_analysis_worker_response_v2",
    "encode_d1_external_analysis_request",
    "encode_d1_external_analysis_request_v2",
    "encode_d1_external_analysis_response",
    "encode_d1_external_analysis_worker_response_v2",
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
_REQUEST_SCHEMA_V2: Final = "d1-external-analysis-request-v2"
_WORKER_RESPONSE_SCHEMA_V2: Final = "d1-external-analysis-response-v2"
_REQUEST_FIELDS_V2: Final = frozenset(
    {
        "analysis_request_digest",
        "input_manifest_base64",
        "input_manifest_sha256",
        "isolation_request_digest",
        "schema_version",
    }
)
_WORKER_RESPONSE_FIELDS_V2: Final = frozenset(
    {
        "analysis_request_digest",
        "analysis_sha256",
        "input_manifest_sha256",
        "isolation_request_digest",
        "schema_version",
    }
)
_REQUEST_V2_TOKEN = object()


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
    """Legacy v1 digest-only request retained for the current Docker adapter."""

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
    """Legacy v1 worker response retained for the current Docker adapter."""

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


def _analysis_request_digest(
    input_control: D1ExternalAnalysisInputControl,
    isolation_request_digest: str,
) -> str:
    return canonical_digest(
        {
            "alias_commitment": input_control.alias_application.aliases.sealed_commitment,
            "capture_digest": input_control.aggregate.manifest_digest,
            "decision_contract_digest": decision_contract_digest(input_control.decision_contract),
            "design_digest": input_control.design.design_digest,
            "isolation_request_digest": isolation_request_digest,
            "schema_version": "d1-analysis-composition-request-v1",
        }
    )


@dataclass(frozen=True, slots=True)
class D1ExternalAnalysisRequestEnvelopeV2:
    input_control: D1ExternalAnalysisInputControl = field(repr=False)
    analysis_request_digest: str
    isolation_request_digest: str
    _factory_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _REQUEST_V2_TOKEN:
            raise TypeError(
                "D1ExternalAnalysisRequestEnvelopeV2 is factory-only; "
                "use build_d1_external_analysis_request_v2"
            )
        if type(self.input_control) is not D1ExternalAnalysisInputControl:
            raise TypeError("v2 request requires verified external analysis input")
        _sha256(self.analysis_request_digest, label="analysis_request_digest")
        isolation = _sha256(self.isolation_request_digest, label="isolation_request_digest")
        expected = _analysis_request_digest(self.input_control, isolation)
        if self.analysis_request_digest != expected:
            raise ValueError("v2 analysis_request_digest differs from input and isolation")

    @property
    def input_manifest_base64(self) -> str:
        return base64.b64encode(self.input_control.canonical_bytes).decode("ascii")

    @property
    def input_manifest_sha256(self) -> str:
        return self.input_control.input_manifest_sha256

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "analysis_request_digest": self.analysis_request_digest,
            "input_manifest_base64": self.input_manifest_base64,
            "input_manifest_sha256": self.input_manifest_sha256,
            "isolation_request_digest": self.isolation_request_digest,
            "schema_version": _REQUEST_SCHEMA_V2,
        }


@dataclass(frozen=True, slots=True)
class D1ExternalAnalysisWorkerResponseV2:
    """Worker hash response; deliberately not an execution or filing receipt."""

    analysis_request_digest: str
    isolation_request_digest: str
    input_manifest_sha256: str
    analysis_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.analysis_request_digest, label="analysis_request_digest")
        _sha256(self.isolation_request_digest, label="isolation_request_digest")
        _sha256(self.input_manifest_sha256, label="input_manifest_sha256")
        _sha256(self.analysis_sha256, label="analysis_sha256")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "analysis_request_digest": self.analysis_request_digest,
            "analysis_sha256": self.analysis_sha256,
            "input_manifest_sha256": self.input_manifest_sha256,
            "isolation_request_digest": self.isolation_request_digest,
            "schema_version": _WORKER_RESPONSE_SCHEMA_V2,
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


def _decode_v2(
    payload: bytes,
    *,
    fields: frozenset[str],
    schema: str,
    legacy_schema: str,
) -> dict[str, object]:
    try:
        return _decode(payload, fields=fields, schema=schema)
    except ValueError as error:
        if legacy_schema.encode("ascii") in payload:
            raise ValueError(
                f"v2 decoder explicitly rejects legacy schema {legacy_schema}"
            ) from error
        raise


def _base64_input(value: object) -> bytes:
    if type(value) is not str or not value:
        raise ValueError("input_manifest_base64 must be nonempty canonical base64")
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("input_manifest_base64 must be canonical base64") from error
    if not payload or base64.b64encode(payload).decode("ascii") != value:
        raise ValueError("input_manifest_base64 must round-trip as canonical base64")
    return payload


def build_d1_external_analysis_request_v2(
    *,
    input_control: D1ExternalAnalysisInputControl,
    isolation_request_digest: str,
) -> D1ExternalAnalysisRequestEnvelopeV2:
    if type(input_control) is not D1ExternalAnalysisInputControl:
        raise TypeError("v2 request factory requires D1ExternalAnalysisInputControl")
    isolation = _sha256(isolation_request_digest, label="isolation_request_digest")
    return D1ExternalAnalysisRequestEnvelopeV2(
        input_control,
        _analysis_request_digest(input_control, isolation),
        isolation,
        _REQUEST_V2_TOKEN,
    )


def encode_d1_external_analysis_request_v2(
    value: D1ExternalAnalysisRequestEnvelopeV2,
) -> bytes:
    if type(value) is not D1ExternalAnalysisRequestEnvelopeV2:
        raise TypeError("v2 external analysis request codec requires its nominal value")
    return _canonical(value.manifest)


def decode_d1_external_analysis_request_v2(
    payload: bytes,
    *,
    preregistration_evidence: D1PreregistrationByteEvidence,
) -> D1ExternalAnalysisRequestEnvelopeV2:
    manifest = _decode_v2(
        payload,
        fields=_REQUEST_FIELDS_V2,
        schema=_REQUEST_SCHEMA_V2,
        legacy_schema=_REQUEST_SCHEMA,
    )
    input_bytes = _base64_input(manifest["input_manifest_base64"])
    input_sha256 = _sha256(manifest["input_manifest_sha256"], label="input_manifest_sha256")
    if hashlib.sha256(input_bytes).hexdigest() != input_sha256:
        raise ValueError("v2 input_manifest_sha256 differs from embedded input bytes")
    input_control = decode_d1_external_analysis_input_control(
        input_bytes,
        preregistration_evidence=preregistration_evidence,
    )
    request = build_d1_external_analysis_request_v2(
        input_control=input_control,
        isolation_request_digest=_sha256(
            manifest["isolation_request_digest"], label="isolation_request_digest"
        ),
    )
    if manifest["analysis_request_digest"] != request.analysis_request_digest:
        raise ValueError("v2 analysis_request_digest differs from decoded input")
    if request.manifest != manifest:
        raise ValueError("v2 request fields do not rebind decoded input bytes")
    return request


def encode_d1_external_analysis_worker_response_v2(
    value: D1ExternalAnalysisWorkerResponseV2,
) -> bytes:
    if type(value) is not D1ExternalAnalysisWorkerResponseV2:
        raise TypeError("v2 external analysis worker response codec requires its nominal value")
    return _canonical(value.manifest)


def decode_d1_external_analysis_worker_response_v2(
    payload: bytes,
) -> D1ExternalAnalysisWorkerResponseV2:
    manifest = _decode_v2(
        payload,
        fields=_WORKER_RESPONSE_FIELDS_V2,
        schema=_WORKER_RESPONSE_SCHEMA_V2,
        legacy_schema=_RESPONSE_SCHEMA,
    )
    return D1ExternalAnalysisWorkerResponseV2(
        analysis_request_digest=_sha256(
            manifest["analysis_request_digest"], label="analysis_request_digest"
        ),
        isolation_request_digest=_sha256(
            manifest["isolation_request_digest"], label="isolation_request_digest"
        ),
        input_manifest_sha256=_sha256(
            manifest["input_manifest_sha256"], label="input_manifest_sha256"
        ),
        analysis_sha256=_sha256(manifest["analysis_sha256"], label="analysis_sha256"),
    )
