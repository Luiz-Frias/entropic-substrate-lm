"""D1 review packets preserve evidence while structurally hiding mechanisms."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import entropic_substrate.telemetry as telemetry_api
from entropic_substrate.analysis.d1_analysis_api import (
    D1AnalysisCompositionRequest,
    compose_d1_analysis,
)
from entropic_substrate.analysis.d1_blind_aliases import (
    D1InMemoryAliasCustodyFake,
    prepare_d1_blind_aliases,
)
from entropic_substrate.analysis.d1_execution_receipt import (
    D1ExternalIsolationAttestation,
    D1IsolationRequest,
    issue_d1_analysis_receipt,
)
from entropic_substrate.analysis.d1_review_packet import (
    D1BlindnessLabel,
    D1BlindReviewPacket,
    D1PacketBlindnessError,
    build_d1_blind_review_packet,
    build_d1_factorial_reviewer_prompt,
    validate_d1_packet_blindness,
)

from .test_d1_analysis_api import _decision, _design
from .test_d1_analysis_capture import _analysis_case
from .test_d1_analysis_receipt import _execution_evidence, _implementation

pytestmark = pytest.mark.unit


class _IdentityCrypto:
    def __init__(self) -> None:
        self._draws = iter((3, 2, 1))

    def randbelow(self, upper: int) -> int:
        return next(self._draws)

    def token_bytes(self, length: int) -> bytes:
        return b"z" * length


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, _, base, aggregate, runs = _analysis_case(tmp_path)
    decision = _decision(base)
    design = _design(decision)
    aliases = prepare_d1_blind_aliases(
        randomness=_IdentityCrypto(), custody=D1InMemoryAliasCustodyFake()
    )

    def read(_path: Path, **expected: object):
        return runs[str(expected["expected_logical_run_id"])]

    monkeypatch.setattr(telemetry_api, "read_d1_run_evidence", read)
    request = D1AnalysisCompositionRequest(
        root=root,
        design=design,
        decision_contract=decision,
        aggregate=aggregate,
        aliases=aliases.aliases,
        isolation=D1IsolationRequest(),
    )
    composition = compose_d1_analysis(request)
    isolation = D1ExternalIsolationAttestation.enforced(
        request=request.isolation,
        mechanism="read-only-network-none",
        execution_evidence=_execution_evidence(),
    )
    receipt = issue_d1_analysis_receipt(
        analysis=composition.analysis,
        admitted=composition.admitted,
        aliases=aliases.aliases,
        implementation=_implementation(),
        isolation=isolation,
    )
    evidence = tuple(runs[item.logical_run_id] for item in aggregate.roster.expected_confirmatory)
    return decision, aggregate, aliases, composition.analysis, receipt, evidence


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
