"""Canonical v2 request and worker-response transport for external D1 analysis."""

from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.analysis.d1_execution_receipt import D1IsolationRequest
from entropic_substrate.analysis.d1_external_analysis_codec import (
    D1ExternalAnalysisRequestEnvelope,
    D1ExternalAnalysisRequestEnvelopeV2,
    D1ExternalAnalysisResponseEnvelope,
    D1ExternalAnalysisWorkerResponseV2,
    build_d1_external_analysis_request_v2,
    decode_d1_external_analysis_request_v2,
    decode_d1_external_analysis_worker_response_v2,
    encode_d1_external_analysis_request,
    encode_d1_external_analysis_request_v2,
    encode_d1_external_analysis_response,
    encode_d1_external_analysis_worker_response_v2,
)
from entropic_substrate.analysis.d1_external_analysis_control import (
    decode_d1_external_analysis_input_control,
)
from entropic_substrate.experiments.d1_primitives import canonical_digest
from entropic_substrate.experiments.d1_registered_design import decision_contract_digest

from .test_d1_external_analysis_input import _canonical, _case

pytestmark = pytest.mark.unit


def _request_v2(tmp_path: Path):
    manifest, evidence, _, _, _ = _case(tmp_path)
    control = decode_d1_external_analysis_input_control(
        _canonical(manifest), preregistration_evidence=evidence
    )
    isolation_digest = D1IsolationRequest().request_digest
    request = build_d1_external_analysis_request_v2(
        input_control=control,
        isolation_request_digest=isolation_digest,
    )
    return request, evidence


def _response_v2(request: D1ExternalAnalysisRequestEnvelopeV2):
    return D1ExternalAnalysisWorkerResponseV2(
        analysis_request_digest=request.analysis_request_digest,
        isolation_request_digest=request.isolation_request_digest,
        input_manifest_sha256=request.input_manifest_sha256,
        analysis_sha256="f" * 64,
    )


def test_v2_request_embeds_exact_input_bytes_and_derives_scientific_digest(
    tmp_path: Path,
) -> None:
    request, evidence = _request_v2(tmp_path)
    payload = encode_d1_external_analysis_request_v2(request)
    restored = decode_d1_external_analysis_request_v2(
        payload,
        preregistration_evidence=evidence,
    )
    control = request.input_control
    expected_request_digest = canonical_digest(
        {
            "alias_commitment": control.alias_application.aliases.sealed_commitment,
            "capture_digest": control.aggregate.manifest_digest,
            "decision_contract_digest": decision_contract_digest(control.decision_contract),
            "design_digest": control.design.design_digest,
            "isolation_request_digest": request.isolation_request_digest,
            "schema_version": "d1-analysis-composition-request-v1",
        }
    )

    assert restored == request
    assert request.analysis_request_digest == expected_request_digest
    assert request.input_manifest_sha256 == control.input_manifest_sha256
    assert base64.b64decode(request.input_manifest_base64, validate=True) == (
        control.canonical_bytes
    )
    assert payload.endswith(b'"schema_version":"d1-external-analysis-request-v2"}')


def test_v2_request_is_factory_only_and_copy_rebinds_input_and_isolation(tmp_path: Path) -> None:
    request, _ = _request_v2(tmp_path)
    with pytest.raises(TypeError, match="factory-only"):
        D1ExternalAnalysisRequestEnvelopeV2(
            input_control=request.input_control,
            analysis_request_digest=request.analysis_request_digest,
            isolation_request_digest=request.isolation_request_digest,
        )
    with pytest.raises(ValueError, match="analysis_request_digest"):
        replace(request, isolation_request_digest="1" * 64)


def test_v2_request_decoder_rejects_hash_base64_digest_and_canonical_drift(
    tmp_path: Path,
) -> None:
    request, evidence = _request_v2(tmp_path)
    mapping = request.manifest
    for field, replacement in (
        ("input_manifest_base64", "not-base64"),
        ("input_manifest_sha256", "0" * 64),
        ("analysis_request_digest", "1" * 64),
        ("isolation_request_digest", "A" * 64),
    ):
        changed = dict(mapping)
        changed[field] = replacement
        with pytest.raises((TypeError, ValueError)):
            decode_d1_external_analysis_request_v2(
                _canonical(changed), preregistration_evidence=evidence
            )
    with pytest.raises(ValueError, match="canonical"):
        decode_d1_external_analysis_request_v2(
            encode_d1_external_analysis_request_v2(request) + b"\n",
            preregistration_evidence=evidence,
        )


def test_v2_decoder_explicitly_rejects_legacy_v1_request(tmp_path: Path) -> None:
    _, evidence = _request_v2(tmp_path)
    legacy = encode_d1_external_analysis_request(
        D1ExternalAnalysisRequestEnvelope("1" * 64, "2" * 64)
    )
    with pytest.raises(ValueError, match="v2"):
        decode_d1_external_analysis_request_v2(legacy, preregistration_evidence=evidence)


def test_worker_response_v2_round_trip_binds_input_and_analysis_hashes(tmp_path: Path) -> None:
    request, _ = _request_v2(tmp_path)
    response = _response_v2(request)
    payload = encode_d1_external_analysis_worker_response_v2(response)

    assert decode_d1_external_analysis_worker_response_v2(payload) == response
    assert response.input_manifest_sha256 == request.input_manifest_sha256
    assert payload.endswith(b'"schema_version":"d1-external-analysis-response-v2"}')
    assert "receipt" not in type(response).__name__.lower()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("analysis_request_digest", "A" * 64),
        ("isolation_request_digest", "g" * 64),
        ("input_manifest_sha256", "1" * 63),
        ("analysis_sha256", True),
    ],
)
def test_worker_response_v2_rejects_noncanonical_hashes(
    tmp_path: Path, field: str, replacement: object
) -> None:
    request, _ = _request_v2(tmp_path)
    mapping = _response_v2(request).manifest
    mapping[field] = replacement
    with pytest.raises(ValueError):
        decode_d1_external_analysis_worker_response_v2(_canonical(mapping))


def test_v2_decoder_explicitly_rejects_legacy_v1_response() -> None:
    legacy = encode_d1_external_analysis_response(
        D1ExternalAnalysisResponseEnvelope("1" * 64, "2" * 64, "3" * 64)
    )
    with pytest.raises(ValueError, match="v2"):
        decode_d1_external_analysis_worker_response_v2(legacy)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":"d1-external-analysis-request-v2","schema_version":"x"}',
        b'{"analysis_request_digest":NaN}',
        b" {}",
        b"[]",
        b"{}",
        b'{"schema_version":"d1-external-analysis-request-v2","unexpected":true}',
        b"\xff",
    ],
)
def test_v2_transport_rejects_duplicate_nonfinite_noncanonical_or_nonobject_bytes(
    tmp_path: Path, payload: bytes
) -> None:
    _, evidence = _request_v2(tmp_path)
    with pytest.raises(ValueError):
        decode_d1_external_analysis_request_v2(payload, preregistration_evidence=evidence)


@pytest.mark.parametrize(
    "payload",
    [
        b"{}",
        b'{"schema_version":"d1-external-analysis-response-v2","schema_version":"x"}',
        b'{"schema_version":"d1-external-analysis-response-v2","unexpected":true}',
        b" {}",
        b"\xff",
    ],
)
def test_v2_worker_response_rejects_missing_duplicate_unknown_or_noncanonical_bytes(
    payload: bytes,
) -> None:
    with pytest.raises(ValueError):
        decode_d1_external_analysis_worker_response_v2(payload)
