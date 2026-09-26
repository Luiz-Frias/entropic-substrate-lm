"""Verified nested control manifest for one external D1 analysis request."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from d1_launch_admission_fixtures import PREREG_COMMIT, PREREG_PATH, contract, family
from entropic_substrate.analysis.d1_blind_aliases import (
    D1InMemoryAliasCustodyFake,
    prepare_d1_blind_aliases,
)
from entropic_substrate.analysis.d1_external_analysis_control import (
    D1ExternalAnalysisControlError,
    D1ExternalAnalysisInputControl,
    decode_d1_alias_application_control,
    decode_d1_external_analysis_input_control,
)
from entropic_substrate.evidence.d1_capture_aggregate_codec import (
    serialize_d1_capture_aggregate,
)
from entropic_substrate.evidence.d1_capture_ports import D1DvcPointerIdentity
from entropic_substrate.evidence.d1_capture_service import D1HeterogeneousCaptureReceipt
from entropic_substrate.experiments.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)
from entropic_substrate.experiments.d1_registration_codec import (
    D1PreregistrationByteEvidence,
    serialize_decision_contract,
    serialize_registered_design,
)

from .test_d1_analysis_capture import _analysis_case
from .test_d1_blind_alias_custody import _DeterministicCrypto
from .test_d1_registered_design_hydration import _RAW_PREREGISTRATION

pytestmark = pytest.mark.unit


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _design_and_contract():
    decision = contract(family(), seeds=(7,))
    preregistration = D1PreregistrationArtifact(
        path=PREREG_PATH,
        blob_sha256=hashlib.sha256(_RAW_PREREGISTRATION).hexdigest(),
        commit=PREREG_COMMIT,
    )
    admitted = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family=family(),
            decision_contract=decision,
            registered_seeds=(7,),
            preregistration=preregistration,
        )
    )
    assert type(admitted) is D1RegisteredDesign
    return admitted, decision


def _case(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _, _, _, aggregate, _ = _analysis_case(tmp_path)
    design, decision = _design_and_contract()
    dvc = aggregate.dvc_handoff.artifacts[0].dvc
    receipt = D1HeterogeneousCaptureReceipt(
        aggregate=aggregate,
        pointer=D1DvcPointerIdentity(
            pointer_path=dvc.pointer_path,
            pointer_sha256=str(dvc.pointer_sha256),
            object_id=str(dvc.object_id),
        ),
        pointer_commit="abcdef1",
    )
    custody = D1InMemoryAliasCustodyFake()
    prepare_d1_blind_aliases(randomness=_DeterministicCrypto(), custody=custody)
    evidence = D1PreregistrationByteEvidence.from_bytes(_RAW_PREREGISTRATION)
    preregistration_control = {
        "commit": design.preregistration.commit,
        "path": design.preregistration.path,
        "schema_version": "d1-preregistration-control-v1",
        "sha256": evidence.sha256,
        "size_bytes": evidence.size_bytes,
    }
    nested = {
        "alias_commitment_base64": _b64(custody.public_bytes),
        "alias_reveal_base64": _b64(custody.private_bytes),
        "capture_aggregate_base64": _b64(serialize_d1_capture_aggregate(aggregate).encode("utf-8")),
        "capture_receipt_base64": _b64(_canonical(receipt.manifest)),
        "decision_contract_base64": _b64(serialize_decision_contract(decision).encode("utf-8")),
        "expected_capture_digest": aggregate.manifest_digest,
        "expected_design_digest": design.design_digest,
        "expected_family_digest": design.family_digest,
        "preregistration_control_base64": _b64(_canonical(preregistration_control)),
        "registered_design_base64": _b64(serialize_registered_design(design).encode("utf-8")),
        "schema_version": "d1-external-analysis-input-v1",
    }
    return nested, evidence, design, decision, aggregate


def test_nested_input_decodes_every_strict_control_and_cross_binding(tmp_path: Path) -> None:
    manifest, evidence, design, decision, aggregate = _case(tmp_path)
    payload = _canonical(manifest)

    control = decode_d1_external_analysis_input_control(
        payload,
        preregistration_evidence=evidence,
    )

    assert control.canonical_bytes == payload
    assert control.input_manifest_sha256 == hashlib.sha256(payload).hexdigest()
    assert control.expected_family_digest == design.family_digest
    assert control.design.design_digest == design.design_digest
    assert control.decision_contract == decision
    assert control.aggregate.manifest_digest == aggregate.manifest_digest
    assert control.capture_receipt.aggregate_digest == aggregate.manifest_digest
    assert control.alias_application.aliases.sealed_commitment
    assert control.preregistration.sha256 == evidence.sha256
    assert "nonce" not in repr(control)
    assert manifest["alias_reveal_base64"] not in repr(control)


def test_verified_nested_input_is_factory_only(tmp_path: Path) -> None:
    manifest, evidence, _, _, _ = _case(tmp_path)
    verified = decode_d1_external_analysis_input_control(
        _canonical(manifest), preregistration_evidence=evidence
    )
    values = {
        field: getattr(verified, field)
        for field in (
            "canonical_bytes",
            "input_manifest_sha256",
            "expected_family_digest",
            "decision_contract",
            "design",
            "aggregate",
            "capture_receipt",
            "alias_application",
            "preregistration",
        )
    }
    with pytest.raises(TypeError, match="factory-only"):
        D1ExternalAnalysisInputControl(**values)


@pytest.mark.parametrize(
    "field",
    [
        "decision_contract_base64",
        "registered_design_base64",
        "capture_aggregate_base64",
        "capture_receipt_base64",
        "alias_commitment_base64",
        "alias_reveal_base64",
        "preregistration_control_base64",
    ],
)
def test_each_nested_payload_must_be_canonical_and_valid(tmp_path: Path, field: str) -> None:
    manifest, evidence, _, _, _ = _case(tmp_path)
    nested = base64.b64decode(str(manifest[field]), validate=True)
    manifest[field] = _b64(nested + b"\n")

    with pytest.raises(D1ExternalAnalysisControlError):
        decode_d1_external_analysis_input_control(
            _canonical(manifest), preregistration_evidence=evidence
        )


@pytest.mark.parametrize(
    "field",
    ["expected_family_digest", "expected_design_digest", "expected_capture_digest"],
)
def test_expected_family_design_and_capture_digest_drift_is_rejected(
    tmp_path: Path, field: str
) -> None:
    manifest, evidence, _, _, _ = _case(tmp_path)
    manifest[field] = "0" * 64

    with pytest.raises(D1ExternalAnalysisControlError):
        decode_d1_external_analysis_input_control(
            _canonical(manifest), preregistration_evidence=evidence
        )


def test_cross_component_seed_construction_and_preregistration_drift_is_rejected(
    tmp_path: Path,
) -> None:
    manifest, evidence, _, _, _ = _case(tmp_path)
    aggregate = json.loads(base64.b64decode(manifest["capture_aggregate_base64"]))
    aggregate["roster"]["expected_confirmatory"][0]["construction_digest"] = "0" * 64
    manifest["capture_aggregate_base64"] = _b64(_canonical(aggregate))
    manifest["expected_capture_digest"] = hashlib.sha256(_canonical(aggregate)).hexdigest()
    with pytest.raises(D1ExternalAnalysisControlError):
        decode_d1_external_analysis_input_control(
            _canonical(manifest), preregistration_evidence=evidence
        )

    manifest, _, _, _, _ = _case(tmp_path / "prereg")
    with pytest.raises(D1ExternalAnalysisControlError, match="preregistration"):
        decode_d1_external_analysis_input_control(
            _canonical(manifest),
            preregistration_evidence=D1PreregistrationByteEvidence.from_bytes(b"different"),
        )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":"d1-external-analysis-input-v1","schema_version":"x"}',
        b'{"expected_family_digest":NaN}',
        b" {}",
        b"{}\n",
        b"[]",
    ],
)
def test_input_manifest_rejects_duplicate_nonfinite_noncanonical_or_nonobject_bytes(
    payload: bytes,
) -> None:
    with pytest.raises(D1ExternalAnalysisControlError):
        decode_d1_external_analysis_input_control(
            payload,
            preregistration_evidence=D1PreregistrationByteEvidence.from_bytes(_RAW_PREREGISTRATION),
        )


@pytest.mark.parametrize("value", ["not-base64", "YWJjZA", 7])
def test_nested_encoding_must_be_exact_canonical_base64(tmp_path: Path, value: object) -> None:
    manifest, evidence, _, _, _ = _case(tmp_path)
    manifest["decision_contract_base64"] = value
    with pytest.raises(D1ExternalAnalysisControlError, match="base64"):
        decode_d1_external_analysis_input_control(
            _canonical(manifest), preregistration_evidence=evidence
        )


def test_token_preserving_copy_cannot_substitute_alias_controls(
    tmp_path: Path,
) -> None:
    manifest, evidence, _, _, _ = _case(tmp_path)
    verified = decode_d1_external_analysis_input_control(
        _canonical(manifest), preregistration_evidence=evidence
    )

    reveal = json.loads(base64.b64decode(manifest["alias_reveal_base64"]))
    reveal["nonce"] = "f" * 64
    reveal_bytes = _canonical(reveal)
    commitment_bytes = _canonical(
        {
            "reveal_sha256": hashlib.sha256(reveal_bytes).hexdigest(),
            "schema_version": "d1-alias-commitment-v1",
        }
    )
    other_aliases = decode_d1_alias_application_control(commitment_bytes, reveal_bytes)
    with pytest.raises(ValueError, match="canonical"):
        replace(verified, alias_application=other_aliases)
