"""D1 analysis receipts distinguish test doubles from enforced isolation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from entropic_substrate.analysis.d1_capture import D1AdmittedAnalysisCapture
from entropic_substrate.analysis.d1_execution_receipt import (
    D1AnalysisFilingBindings,
    D1AnalysisImplementationIdentity,
    D1AnalysisRunnerRequest,
    D1ExecutorKind,
    D1ExternalIsolationAttestation,
    D1InProcessIsolationRunnerFake,
    D1IsolationExecutionEvidence,
    D1IsolationRequest,
    issue_d1_analysis_receipt,
)
from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1ArtifactRole,
    D1DvcBinding,
)

from .test_d1_aliased_analysis import _aliases, _packet

pytestmark = pytest.mark.unit


def _artifact() -> D1ArtifactIdentity:
    return D1ArtifactIdentity(
        normalized_path="results/captured/d1-a.jsonl",
        role=D1ArtifactRole.JOURNAL,
        size_bytes=123,
        md5="a" * 32,
        sha256="b" * 64,
        dvc=D1DvcBinding.pending("results/captured.dvc").bind(
            pointer_sha256="c" * 64,
            object_id="d" * 32 + ".dir",
        ),
    )


def _admitted():
    packet = _packet()
    bindings = packet.manifest["bindings"]
    assert isinstance(bindings, dict)
    return packet, D1AdmittedAnalysisCapture(
        design_digest=str(bindings["design_digest"]),
        decision_contract_digest=str(bindings["decision_contract_digest"]),
        capture_digest=str(bindings["capture_digest"]),
        cells=(),
        exclusions=(),
        runs=(),
        input_artifacts=(_artifact(),),
    )


def _implementation() -> D1AnalysisImplementationIdentity:
    return D1AnalysisImplementationIdentity(
        code_commit="1" * 40,
        repository_digest="2" * 64,
        analysis_code_digest="3" * 64,
        image_digest="sha256:" + "4" * 64,
        complete_reader_digest="5" * 64,
        defect_reader_digest="6" * 64,
    )


def _execution_evidence() -> D1IsolationExecutionEvidence:
    return D1IsolationExecutionEvidence(
        command_sha256="1" * 64,
        exit_code=0,
        request_sha256="2" * 64,
        analysis_sha256="3" * 64,
        response_sha256="4" * 64,
        stdout_sha256="5" * 64,
        stderr_sha256="6" * 64,
        image_digest="sha256:" + "7" * 64,
        platform="linux/amd64",
    )


def test_in_process_fake_satisfies_port_but_cannot_issue_filing_eligible_receipt() -> None:
    packet, admitted = _admitted()
    request = D1AnalysisRunnerRequest("7" * 64, D1IsolationRequest())
    runner = D1InProcessIsolationRunnerFake(lambda _request: packet)

    result = runner.run(request)
    receipt = issue_d1_analysis_receipt(
        analysis=result.analysis,
        admitted=admitted,
        aliases=_aliases(),
        implementation=_implementation(),
        isolation=result.isolation,
    )

    assert result.isolation.executor_kind is D1ExecutorKind.TEST_FAKE
    assert receipt.manifest["filing_eligible"] is False
    assert receipt.manifest["filing_bindings"] is None
    assert receipt.manifest["isolation"]["mechanism"] == "in-process-contract-fake"  # type: ignore[index]
    assert receipt.manifest["inputs"] == [  # exact MD5 and SHA survive
        {
            "dvc_object_id": "d" * 32 + ".dir",
            "dvc_pointer_path": "results/captured.dvc",
            "dvc_pointer_sha256": "c" * 64,
            "md5": "a" * 32,
            "path": "results/captured/d1-a.jsonl",
            "role": "authoritative_journal",
            "sha256": "b" * 64,
            "size_bytes": 123,
        }
    ]


def test_receipt_is_deterministic_without_promoting_the_in_process_fake() -> None:
    packet, admitted = _admitted()
    isolation = D1ExternalIsolationAttestation.test_fake(D1IsolationRequest())
    first = issue_d1_analysis_receipt(
        analysis=packet,
        admitted=admitted,
        aliases=_aliases(),
        implementation=_implementation(),
        isolation=isolation,
    )
    second = issue_d1_analysis_receipt(
        analysis=packet,
        admitted=admitted,
        aliases=_aliases(),
        implementation=_implementation(),
        isolation=isolation,
    )

    assert first.canonical_json == second.canonical_json
    assert first.receipt_digest == second.receipt_digest
    assert first.manifest["filing_eligible"] is False
    changed = issue_d1_analysis_receipt(
        analysis=packet,
        admitted=admitted,
        aliases=_aliases(),
        implementation=replace(_implementation(), analysis_code_digest="8" * 64),
        isolation=isolation,
    )
    assert changed.receipt_digest != first.receipt_digest


def test_enforced_attestation_cannot_be_constructed_outside_verified_adapter() -> None:
    request = D1IsolationRequest()

    with pytest.raises(TypeError, match="factory-only"):
        D1ExternalIsolationAttestation(
            request.request_digest,
            D1ExecutorKind.EXTERNAL_ENFORCED,
            True,
            True,
            True,
            "fabricated",
            _execution_evidence(),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"network_disabled": False},
        {"filesystem_read_only": False},
        {"live_tracking_blocked": False},
    ],
)
def test_isolation_request_refuses_any_weaker_environment(overrides: dict[str, bool]) -> None:
    with pytest.raises(ValueError, match="requires"):
        D1IsolationRequest(**overrides)


def test_receipt_rejects_alias_or_analysis_binding_drift() -> None:
    packet, admitted = _admitted()
    isolation = D1ExternalIsolationAttestation.test_fake(D1IsolationRequest())
    with pytest.raises(ValueError, match="alias commitment"):
        issue_d1_analysis_receipt(
            analysis=packet,
            admitted=admitted,
            aliases=replace(_aliases(), sealed_commitment="9" * 64),
            implementation=_implementation(),
            isolation=isolation,
        )


def test_filing_bindings_require_exact_sha256_values() -> None:
    with pytest.raises(ValueError, match="reader_sha256"):
        D1AnalysisFilingBindings(
            capture_manifest_sha256="1" * 64,
            preregistration_sha256="2" * 64,
            reader_sha256="not-a-digest",
            isolation_attestation_sha256="4" * 64,
            output_sha256="5" * 64,
        )
