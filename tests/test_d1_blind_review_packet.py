"""D1 review packets preserve evidence while structurally hiding mechanisms."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import pytest

from entropic_substrate.analysis.d1_blind_aliases import (
    D1InMemoryAliasCustodyFake,
    D1PreparedBlindAliases,
    prepare_d1_blind_aliases,
)
from entropic_substrate.analysis.d1_execution_receipt import (
    D1AnalysisExecutionReceipt,
    issue_d1_verified_analysis_receipt,
)
from entropic_substrate.analysis.d1_machine_analysis import D1AliasedMachineAnalysis
from entropic_substrate.analysis.d1_review_packet import (
    D1BlindnessLabel,
    D1BlindReviewPacket,
    D1PacketBlindnessError,
    build_d1_blind_review_packet,
    build_d1_factorial_reviewer_prompt,
    validate_d1_packet_blindness,
)
from entropic_substrate.evidence.d1_capture_aggregate import D1HeterogeneousCaptureAggregate
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.evidence.d1_review_adjudication import (
    D1_NO_MAJOR_OUTCOME,
    D1_NO_MAJOR_PROMPT_DIRECTIVE,
)
from entropic_substrate.evidence.d1_telemetry_values import D1RunEvidence

from .test_d1_analysis_receipt import _implementation
from .test_d1_blind_alias_custody import _DeterministicCrypto
from .test_d1_external_analysis_adapter_v2 import _adapter, _adapter_case

pytestmark = pytest.mark.unit


class _BlindCase(NamedTuple):
    decision: D1DecisionContract
    aggregate: D1HeterogeneousCaptureAggregate
    aliases: D1PreparedBlindAliases
    analysis: D1AliasedMachineAnalysis
    receipt: D1AnalysisExecutionReceipt
    evidence: tuple[D1RunEvidence, ...]


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _BlindCase:
    case, launch, runner, request = _adapter_case(tmp_path, monkeypatch)
    result = _adapter(launch, runner).run(request)
    aliases = prepare_d1_blind_aliases(
        randomness=_DeterministicCrypto(), custody=D1InMemoryAliasCustodyFake()
    )
    receipt = issue_d1_verified_analysis_receipt(
        analysis=result.analysis,
        request=result.request,
        worker_response=result.worker_response,
        implementation=_implementation(),
        isolation=result.isolation,
    )
    evidence = tuple(
        case.runs[item.logical_run_id] for item in case.aggregate.roster.expected_confirmatory
    )
    return _BlindCase(
        case.request.input_control.decision_contract,
        case.aggregate,
        aliases,
        result.analysis.analysis,
        receipt,
        evidence,
    )


def test_packet_has_complete_aliased_roster_readers_and_anonymous_run_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision, aggregate, aliases, analysis, receipt, evidence = _case(tmp_path, monkeypatch)

    packet = build_d1_blind_review_packet(
        decision_contract=decision,
        aggregate=aggregate,
        analysis=analysis,
        analysis_receipt=receipt,
        aliases=aliases,
        run_evidence=evidence,
    )
    manifest = packet.manifest

    assert manifest["blindness_label"] == D1BlindnessLabel.ROLE_MASKED.value
    assert len(manifest["capture_roster"]["confirmatory"]) == 4  # type: ignore[index]
    assert len(manifest["capture_roster"]["attempts"]) == len(aggregate.attempts)  # type: ignore[index]
    assert {row["arm_alias"] for row in manifest["run_attestations"]} == {  # type: ignore[index]
        "A",
        "B",
        "C",
        "D",
    }
    assert all(row["registered_construction_match"] for row in manifest["run_attestations"])  # type: ignore[index]
    readers = manifest["validated_readers"]
    assert readers["complete"]["sha256"] == "5" * 64  # type: ignore[index]
    assert readers["terminal_defect"]["sha256"] == "6" * 64  # type: ignore[index]
    assert manifest["nonblindable_fields"]
    validate_d1_packet_blindness(packet, decision, aggregate, aliases)


def test_prompt_quotes_registered_rules_riders_and_aliased_coefficients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision, aggregate, aliases, analysis, receipt, evidence = _case(tmp_path, monkeypatch)
    packet = build_d1_blind_review_packet(
        decision_contract=decision,
        aggregate=aggregate,
        analysis=analysis,
        analysis_receipt=receipt,
        aliases=aliases,
        run_evidence=evidence,
    )

    prompt = build_d1_factorial_reviewer_prompt(packet, decision)

    assert "recompute only from the sealed packet" in prompt.text
    assert "do not authorize a successor gate" in prompt.text.lower()
    assert prompt.text.splitlines().count(D1_NO_MAJOR_PROMPT_DIRECTIVE) == 1
    assert D1_NO_MAJOR_OUTCOME in prompt.text
    assert decision.verdict_vocabulary in prompt.text
    for clause in decision.clause_rules.values():
        assert clause.registered_rule in prompt.text
        assert clause.decision_vocabulary in prompt.text
    for index, rule in enumerate(decision.contrast_rules.values(), start=1):
        assert f"C{index:02d}" in prompt.text
        for term in rule.terms:
            alias = aliases.aliases.for_arm(type(next(iter(aliases.aliases.aliases)))(term.arm_id))
            assert f'"arm_alias":"{alias}","coefficient":{term.coefficient}' in prompt.text


def test_structural_scan_rejects_hidden_identity_and_construction_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision, aggregate, aliases, analysis, receipt, evidence = _case(tmp_path, monkeypatch)
    packet = build_d1_blind_review_packet(
        decision_contract=decision,
        aggregate=aggregate,
        analysis=analysis,
        analysis_receipt=receipt,
        aliases=aliases,
        run_evidence=evidence,
    )
    leaked = packet.manifest
    leaked["innocent_label"] = "sg"
    with pytest.raises(D1PacketBlindnessError, match="canonical arm identity"):
        validate_d1_packet_blindness(
            D1BlindReviewPacket.from_manifest(leaked), decision, aggregate, aliases
        )

    leaked = packet.manifest
    leaked["run_attestations"][0]["construction_digest"] = "0" * 64  # type: ignore[index]
    with pytest.raises(D1PacketBlindnessError, match="construction-bearing field"):
        validate_d1_packet_blindness(
            D1BlindReviewPacket.from_manifest(leaked), decision, aggregate, aliases
        )


def test_packet_refuses_incomplete_all_run_reader_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision, aggregate, aliases, analysis, receipt, evidence = _case(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="every expected cell"):
        build_d1_blind_review_packet(
            decision_contract=decision,
            aggregate=aggregate,
            analysis=analysis,
            analysis_receipt=receipt,
            aliases=aliases,
            run_evidence=evidence[:-1],
        )
    with pytest.raises(ValueError, match="filing-eligible"):
        build_d1_blind_review_packet(
            decision_contract=decision,
            aggregate=aggregate,
            analysis=analysis,
            analysis_receipt=replace(
                receipt, canonical_json=receipt.canonical_json.replace("true", "false")
            ),
            aliases=aliases,
            run_evidence=evidence,
        )
