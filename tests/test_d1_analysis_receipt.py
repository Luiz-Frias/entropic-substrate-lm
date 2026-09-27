"""D1 analysis receipts distinguish test doubles from enforced isolation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from entropic_substrate.kernel.identity.d1_primitives import canonical_json
from entropic_substrate.ladder.analyze.d1_capture import D1AdmittedAnalysisCapture
from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1ArtifactRole,
    D1DvcBinding,
)
from entropic_substrate.ladder.review.d1_execution_receipt import (
    D1AnalysisExecutionReceipt,
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


# FOR-1523 (2026-09-20): the claimed image must equal the image the attested
# isolation records. The v2 adapter tests launch with `_IMAGE = ...@sha256:` +
# 64 x "a" (tests/test_d1_external_analysis_adapter_v2.py), and every
# verified-receipt test in the four consumer modules shares THIS fixture, so
# it carries that same digest. MEASURED before the fix: this fixture said
# sha256:4...4 and 42 tests across these four modules passed only because the issuer
# never compared the two.
_LAUNCH_IMAGE_DIGEST = "sha256:" + "a" * 64


def _implementation() -> D1AnalysisImplementationIdentity:
    return D1AnalysisImplementationIdentity(
        code_commit="1" * 40,
        repository_digest="2" * 64,
        image_digest=_LAUNCH_IMAGE_DIGEST,
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
        image_digest=_LAUNCH_IMAGE_DIGEST,  # FOR-1523: same image as the claim
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
        implementation=replace(_implementation(), repository_digest="8" * 64),
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
    with pytest.raises(ValueError, match="output_sha256"):
        D1AnalysisFilingBindings(
            capture_manifest_sha256="1" * 64,
            preregistration_sha256="2" * 64,
            isolation_attestation_sha256="4" * 64,
            output_sha256="not-a-digest",
        )


def test_implementation_identity_exact_key_set_guards_retired_analysis_code_digest() -> None:
    """Guard-teeth: analysis_code_digest retirement is not re-introducible.

    Reintroducing the retired "analysis_code_digest" key into the implementation
    identity's manifest must cause this test to fail, pinning the exact-key-set
    assertion on both the dataclass and the on-the-wire manifest property.
    """
    impl = _implementation()

    # Retired field check: none of the three retired fields exist as dataclass
    # attributes. MEASURED 2026-09-20 (scratch-worktree mutation proof, local
    # CodeRabbit review): a hasattr check scoped to analysis_code_digest alone
    # stays GREEN if complete_reader_digest or defect_reader_digest is
    # reintroduced as a dataclass field without also being added to the
    # manifest property below, because the exact-key-set assertion only sees
    # what manifest chooses to emit. Parameterizing across all three closes
    # that gap.
    for field_name in ("analysis_code_digest", "complete_reader_digest", "defect_reader_digest"):
        assert not hasattr(impl, field_name)

    # Expected keys in the implementation identity manifest (wire format).
    # Retired: "analysis_code_digest", "complete_reader_digest", "defect_reader_digest"
    # (D1 reader-source closure, measured 2026-09-20).
    expected_keys = frozenset({"code_commit", "image_digest", "repository_digest"})
    assert frozenset(impl.manifest) == expected_keys


def test_wire_decoder_rejects_a_synthetic_digest_reintroduced_into_implementation() -> None:
    """Guard-teeth: the on-wire decoder must reject a resurrected retired digest.

    CodeRabbit (remote review, PR #287, inline finding at
    .claude/workflows/briefs/p2-d1-closure.md:130-132): the emit-side exact-key
    guard above pins ``D1AnalysisImplementationIdentity.manifest``, but
    ``_decode`` in d1_analysis_receipt_codec.py only checked that
    ``implementation`` was a dict, not that its keys were exact -- so a
    reintroduced "analysis_code_digest" (or any other synthetic 64-hex field)
    on the wire passed parsing untouched. MEASURED 2026-09-20: constructing a
    D1AnalysisExecutionReceipt from a manifest with implementation keys
    {"code_commit", "image_digest", "repository_digest", "analysis_code_digest"}
    parsed successfully before the fix (design-gate constraint 3: no synthetic
    64-hex digest may satisfy a provenance guard).
    """
    packet, admitted = _admitted()
    isolation = D1ExternalIsolationAttestation.test_fake(D1IsolationRequest())
    receipt = issue_d1_analysis_receipt(
        analysis=packet,
        admitted=admitted,
        aliases=_aliases(),
        implementation=_implementation(),
        isolation=isolation,
    )
    mutated_manifest = dict(receipt.manifest)
    mutated_manifest["implementation"] = {
        **mutated_manifest["implementation"],  # type: ignore[dict-item]
        "analysis_code_digest": "9" * 64,
    }

    with pytest.raises(ValueError, match="implementation"):
        D1AnalysisExecutionReceipt(canonical_json(mutated_manifest))
