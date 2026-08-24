"""Strict transport-neutral envelopes for an isolated D1 analysis worker."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from entropic_substrate.analysis.d1_external_analysis_codec import (
    D1ExternalAnalysisRequestEnvelope,
    D1ExternalAnalysisResponseEnvelope,
    decode_d1_external_analysis_request,
    decode_d1_external_analysis_response,
    encode_d1_external_analysis_request,
    encode_d1_external_analysis_response,
)

pytestmark = pytest.mark.unit


def _request() -> D1ExternalAnalysisRequestEnvelope:
    return D1ExternalAnalysisRequestEnvelope(
        analysis_request_digest="1" * 64,
        isolation_request_digest="2" * 64,
    )


def _response() -> D1ExternalAnalysisResponseEnvelope:
    return D1ExternalAnalysisResponseEnvelope(
        analysis_request_digest="1" * 64,
        isolation_request_digest="2" * 64,
        analysis_sha256="3" * 64,
    )


def test_request_and_response_round_trip_as_canonical_cli_neutral_bytes() -> None:
    request = _request()
    response = _response()

    request_bytes = encode_d1_external_analysis_request(request)
    response_bytes = encode_d1_external_analysis_response(response)

    assert decode_d1_external_analysis_request(request_bytes) == request
    assert decode_d1_external_analysis_response(response_bytes) == response
    assert request_bytes == (
        b'{"analysis_request_digest":"'
        + b"1" * 64
        + b'","isolation_request_digest":"'
        + b"2" * 64
        + b'","schema_version":"d1-external-analysis-request-v1"}'
    )
    assert response_bytes.endswith(b'"schema_version":"d1-external-analysis-response-v1"}')
    assert b"docker" not in request_bytes + response_bytes
    assert b"/" not in request_bytes + response_bytes


@pytest.mark.parametrize(
    "payload",
    [
        b"{}",
        b'{"analysis_request_digest":"' + b"1" * 64 + b'","analysis_request_digest":"x"}',
        b'{"analysis_request_digest":"' + b"1" * 64 + b'","unexpected":true}',
        b'{"analysis_request_digest":NaN}',
        b"[]",
        b"\xff",
    ],
)
def test_request_decoder_rejects_missing_duplicate_unknown_nonfinite_or_nonobject_bytes(
    payload: bytes,
) -> None:
    with pytest.raises(ValueError):
        decode_d1_external_analysis_request(payload)


def test_decoders_reject_valid_json_that_is_not_the_canonical_byte_encoding() -> None:
    request = encode_d1_external_analysis_request(_request())
    response = encode_d1_external_analysis_response(_response())

    with pytest.raises(ValueError, match="canonical"):
        decode_d1_external_analysis_request(request + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        decode_d1_external_analysis_response(response.replace(b'":', b'": '))


def test_response_requires_exact_request_isolation_and_analysis_digests() -> None:
    with pytest.raises(ValueError, match="analysis_sha256"):
        D1ExternalAnalysisResponseEnvelope(
            analysis_request_digest="1" * 64,
            isolation_request_digest="2" * 64,
            analysis_sha256="NOT-A-DIGEST",
        )

    encoded = encode_d1_external_analysis_response(_response())
    drifted = json.loads(encoded)
    drifted["isolation_request_digest"] = "4" * 64
    decoded = decode_d1_external_analysis_response(
        json.dumps(drifted, separators=(",", ":"), sort_keys=True).encode()
    )
    assert decoded != _response()
    assert decoded == replace(_response(), isolation_request_digest="4" * 64)
