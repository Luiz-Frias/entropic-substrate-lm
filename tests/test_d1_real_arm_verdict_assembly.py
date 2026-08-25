"""Verified blind analysis becomes an attested real-arm D1 verdict."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.analysis.d1_blind_aliases import (
    D1InMemoryAliasCustodyFake,
    prepare_d1_blind_aliases,
)
from entropic_substrate.analysis.d1_execution_receipt import issue_d1_verified_analysis_receipt
from entropic_substrate.analysis.d1_real_arm_verdict import (
    D1VerdictDecision,
    assemble_d1_real_arm_verdict,
)
from entropic_substrate.analysis.d1_review_execution import (
    D1CloudReviewerFake,
    D1CloudReviewRequest,
    D1InMemoryReviewReceiptStoreFake,
    execute_d1_blind_review,
    release_d1_alias_reveal,
)
from entropic_substrate.analysis.d1_review_packet import (
    D1BlindReviewPacket,
    build_d1_blind_review_packet,
    build_d1_factorial_reviewer_prompt,
)
from entropic_substrate.evidence.d1_review_adjudication import (
    D1_NO_MAJOR_OUTCOME,
    adjudicate_d1_review,
    parse_d1_major_findings,
)
from entropic_substrate.evidence.d1_validator import ValidatedD1Verdict

from .test_d1_analysis_receipt import _implementation
from .test_d1_blind_alias_custody import _DeterministicCrypto
from .test_d1_external_analysis_adapter_v2 import _adapter, _adapter_case

pytestmark = pytest.mark.unit


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    generic_packet: bool = False,
):
    case, launch, runner, request = _adapter_case(tmp_path, monkeypatch)
    result = _adapter(launch, runner).run(request)
    verified = result.analysis
    custody = D1InMemoryAliasCustodyFake()
    aliases = prepare_d1_blind_aliases(
        randomness=_DeterministicCrypto(),
        custody=custody,
    )
    receipt = issue_d1_verified_analysis_receipt(
        analysis=verified,
        request=result.request,
        worker_response=result.worker_response,
        implementation=_implementation(),
        isolation=result.isolation,
    )
    evidence = tuple(
        case.runs[item.logical_run_id] for item in case.aggregate.roster.expected_confirmatory
    )
    packet = build_d1_blind_review_packet(
        decision_contract=request.envelope.input_control.decision_contract,
        aggregate=case.aggregate,
        analysis=verified.analysis,
        analysis_receipt=receipt,
        aliases=aliases,
        run_evidence=evidence,
    )
    if generic_packet:
        packet = D1BlindReviewPacket.from_manifest(packet.manifest)  # type: ignore[assignment]
    prompt = build_d1_factorial_reviewer_prompt(
        packet,
        request.envelope.input_control.decision_contract,
    )
    execution = execute_d1_blind_review(
        request=D1CloudReviewRequest(packet, prompt, "review-provider", "review-model-v1", 1),
        reviewer=D1CloudReviewerFake(
            response=b"review-response",
            report=f"{D1_NO_MAJOR_OUTCOME}\n".encode(),
        ),
        receipt_store=D1InMemoryReviewReceiptStoreFake(),
    )
    reveal = release_d1_alias_reveal(
        execution=execution,
        aliases=aliases,
        custody=custody,
        public_commitment=custody.public_bytes,
    )
    report = parse_d1_major_findings(
        execution.report,
        reviewer_prompt=prompt.text.encode(),
    )
    adjudication = adjudicate_d1_review(
        report=report,
        dispositions=(),
        operator_record_sha256="c" * 64,
        execution_receipt_sha256=execution.receipt.receipt_digest,
        review_request_sha256=execution.request.request_digest,
        review_packet_sha256=execution.request.packet.packet_digest,
    )
    return verified, execution, reveal, adjudication


def test_assembler_preserves_complete_frontier_from_authoritative_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, execution, reveal, adjudication = _inputs(tmp_path, monkeypatch)
    assembled = assemble_d1_real_arm_verdict(
        verified=verified,
        execution=execution,
        reveal=reveal,
        adjudication=adjudication,
        decision=D1VerdictDecision("SUPPORTED"),
    )

    assert type(assembled.verdict) is ValidatedD1Verdict
    assert assembled.verdict.feasible_arms == ("sg", "ac", "both")
    assert assembled.verdict.successor.eligible_arms == ("sg", "ac", "both")
    assert tuple(assembled.verdict.clauses_by_arm) == ("sg", "ac", "both")
    assert (assembled.verified, assembled.execution, assembled.reveal) == (
        verified,
        execution,
        reveal,
    )
    assembled.require_factory_attested()


def test_generic_packet_cannot_acquire_verdict_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, execution, reveal, adjudication = _inputs(
        tmp_path,
        monkeypatch,
        generic_packet=True,
    )
    with pytest.raises(TypeError, match="validated review packet"):
        assemble_d1_real_arm_verdict(
            verified=verified,
            execution=execution,
            reveal=reveal,
            adjudication=adjudication,
            decision=D1VerdictDecision("SUPPORTED"),
        )
