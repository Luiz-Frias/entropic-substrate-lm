"""D1 review packets preserve evidence while structurally hiding mechanisms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple, cast

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
    d1_review_roster_alias_order,
    validate_d1_packet_blindness,
)
from entropic_substrate.evidence.d1_capture_aggregate import (
    D1CapturePartitionProof,
    D1HeterogeneousCaptureAggregate,
)
from entropic_substrate.evidence.d1_capture_bundle import D1AttemptEvidence
from entropic_substrate.evidence.d1_capture_roster import (
    D1AttemptLane,
    D1DeclaredAttempt,
    D1LaunchRoster,
)
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.evidence.d1_review_adjudication import (
    D1_NO_MAJOR_OUTCOME,
    D1_NO_MAJOR_PROMPT_DIRECTIVE,
)
from entropic_substrate.evidence.d1_telemetry_values import D1RunEvidence
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass

from .test_d1_analysis_receipt import _implementation
from .test_d1_blind_alias_custody import _DeterministicCrypto
from .test_d1_external_analysis_adapter_v2 import _adapter, _adapter_case

pytestmark = pytest.mark.unit


def _roster_rows(manifest: dict[str, object], lane: str) -> list[dict[str, object]]:
    roster = cast(dict[str, object], manifest["capture_roster"])
    return cast(list[dict[str, object]], roster[lane])


def _randomization(manifest: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], manifest["roster_randomization"])


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def test_registered_seed_alias_order_is_reproducible_and_seed_sensitive() -> None:
    assert d1_review_roster_alias_order(7) == d1_review_roster_alias_order(7)
    assert d1_review_roster_alias_order(7) != d1_review_roster_alias_order(8)
    assert set(d1_review_roster_alias_order(7)) == {"A", "B", "C", "D"}
    with pytest.raises(ValueError, match="registered seed"):
        d1_review_roster_alias_order(-1)
    with pytest.raises(ValueError, match="registered seed"):
        d1_review_roster_alias_order(cast(int, True))


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


def _case_with_attempts(case: _BlindCase) -> _BlindCase:
    decision, aggregate, aliases, analysis, receipt, evidence = case
    attempts = tuple(
        D1DeclaredAttempt(
            logical_run_id=f"blind-attempt-{index}",
            arm=arm,
            seed=7,
            pairing_id="attempt-pair",
            run_class=RunClass.CONFIRMATORY,
            lane=D1AttemptLane.ABANDONED,
            disposition="provider evidence unavailable",
        )
        for index, arm in enumerate((D1Arm.NONE, D1Arm.SG), start=1)
    )
    attempt_evidence = tuple(
        D1AttemptEvidence(item, (), "provider evidence unavailable") for item in attempts
    )
    roster = D1LaunchRoster(
        preregistration_path=aggregate.roster.preregistration_path,
        preregistration_commit=aggregate.roster.preregistration_commit,
        family_digest=aggregate.roster.family_digest,
        expected_confirmatory=aggregate.roster.expected_confirmatory,
        attempts=attempts,
    )
    with_attempts = D1HeterogeneousCaptureAggregate.from_validated_evidence(
        roster=roster,
        partition_proof=aggregate.partition_proof,
        complete_runs=aggregate.complete_runs,
        terminal_defects=aggregate.terminal_defects,
        attempts=attempt_evidence,
        pair_exclusions=aggregate.pair_exclusions,
        dvc_handoff=aggregate.dvc_handoff,
    )
    bound_receipt = replace(
        receipt,
        canonical_json=receipt.canonical_json.replace(
            aggregate.manifest_digest,
            with_attempts.manifest_digest,
        ),
    )
    return _BlindCase(decision, with_attempts, aliases, analysis, bound_receipt, evidence)


def _case_with_seed_blocks(case: _BlindCase) -> _BlindCase:
    decision, aggregate, aliases, analysis, receipt, evidence = case
    originals = aggregate.roster.expected_confirmatory
    seed_eight = tuple(
        replace(
            item,
            logical_run_id=f"{item.logical_run_id}-seed-8",
            seed=8,
            pairing_id="pair-8",
        )
        for item in originals
    )
    shuffled_expected = (*reversed(originals), *reversed(seed_eight))
    roster = D1LaunchRoster(
        preregistration_path=aggregate.roster.preregistration_path,
        preregistration_commit=aggregate.roster.preregistration_commit,
        family_digest=aggregate.roster.family_digest,
        expected_confirmatory=shuffled_expected,
        attempts=(),
    )
    captures_by_arm = {item.expected.arm: item for item in aggregate.complete_runs}
    seed_eight_captures = tuple(
        replace(
            captures_by_arm[item.arm],
            expected=item,
            capture_input=replace(
                captures_by_arm[item.arm].capture_input,
                logical_run_id=item.logical_run_id,
            ),
        )
        for item in seed_eight
    )
    all_ids = frozenset(item.logical_run_id for item in shuffled_expected)
    with_seed_blocks = D1HeterogeneousCaptureAggregate.from_validated_evidence(
        roster=roster,
        partition_proof=D1CapturePartitionProof(all_ids, all_ids, frozenset()),
        complete_runs=(*aggregate.complete_runs, *seed_eight_captures),
        terminal_defects=(),
        attempts=(),
        pair_exclusions=(),
        dvc_handoff=aggregate.dvc_handoff,
    )
    evidence_by_construction = {item.identity.construction_digest: item for item in evidence}
    seed_eight_evidence = tuple(
        replace(
            evidence_by_construction[item.construction_digest],
            identity=replace(
                evidence_by_construction[item.construction_digest].identity,
                logical_run_id=item.logical_run_id,
            ),
        )
        for item in seed_eight
    )
    registered = cast(dict[str, object], _thaw(decision.registered_constants))
    registered["seeds"] = [8, 7]
    seeded_decision = replace(decision, registered_constants=registered)
    bound_receipt = replace(
        receipt,
        canonical_json=receipt.canonical_json.replace(
            aggregate.manifest_digest,
            with_seed_blocks.manifest_digest,
        ),
    )
    return _BlindCase(
        seeded_decision,
        with_seed_blocks,
        aliases,
        analysis,
        bound_receipt,
        (*reversed(evidence), *reversed(seed_eight_evidence)),
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
    randomized = _randomization(manifest)
    assert randomized["algorithm"] == "sha256-registered-seed-alias-sort-v1"
    assert len(cast(str, randomized["confirmatory_order_sha256"])) == 64
    confirmatory = _roster_rows(manifest, "confirmatory")
    for seed in {cast(int, row["seed"]) for row in confirmatory}:
        observed = tuple(cast(str, row["arm_alias"]) for row in confirmatory if row["seed"] == seed)
        assert observed == d1_review_roster_alias_order(seed)
    repeated = build_d1_blind_review_packet(
        decision_contract=decision,
        aggregate=aggregate,
        analysis=analysis,
        analysis_receipt=receipt,
        aliases=aliases,
        run_evidence=evidence,
    )
    assert repeated.manifest["capture_roster"] == manifest["capture_roster"]
    assert repeated.manifest["roster_randomization"] == randomized
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


def test_attempt_roster_order_and_record_are_independently_load_bearing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision, aggregate, aliases, analysis, receipt, evidence = _case_with_attempts(
        _case(tmp_path, monkeypatch)
    )
    packet = build_d1_blind_review_packet(
        decision_contract=decision,
        aggregate=aggregate,
        analysis=analysis,
        analysis_receipt=receipt,
        aliases=aliases,
        run_evidence=evidence,
    )
    manifest = packet.manifest
    randomization = _randomization(manifest)
    attempt_rows = _roster_rows(manifest, "attempts")
    assert len(cast(str, randomization["attempt_order_sha256"])) == 64
    assert len(attempt_rows) == 2
    expected_aliases = tuple(
        alias
        for alias in d1_review_roster_alias_order(7)
        if alias in {aliases.aliases.for_arm(D1Arm.NONE), aliases.aliases.for_arm(D1Arm.SG)}
    )
    assert tuple(cast(str, row["arm_alias"]) for row in attempt_rows) == expected_aliases
    assert tuple(cast(str, row["attempt_alias"]) for row in attempt_rows) == ("T001", "T002")

    reordered = packet.manifest
    _roster_rows(reordered, "attempts").reverse()
    with pytest.raises(D1PacketBlindnessError, match="randomized roster order or record differs"):
        validate_d1_packet_blindness(
            D1BlindReviewPacket.from_manifest(reordered), decision, aggregate, aliases
        )

    record_drift = packet.manifest
    _randomization(record_drift)["attempt_order_sha256"] = "0" * 64
    with pytest.raises(D1PacketBlindnessError, match="randomized roster order or record differs"):
        validate_d1_packet_blindness(
            D1BlindReviewPacket.from_manifest(record_drift), decision, aggregate, aliases
        )


def test_confirmatory_seed_blocks_follow_registered_order_before_alias_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision, aggregate, aliases, analysis, receipt, evidence = _case_with_seed_blocks(
        _case(tmp_path, monkeypatch)
    )
    packet = build_d1_blind_review_packet(
        decision_contract=decision,
        aggregate=aggregate,
        analysis=analysis,
        analysis_receipt=receipt,
        aliases=aliases,
        run_evidence=evidence,
    )
    rows = _roster_rows(packet.manifest, "confirmatory")
    seed_blocks = tuple(dict.fromkeys(cast(int, row["seed"]) for row in rows))
    assert seed_blocks == (8, 7)
    for seed in seed_blocks:
        aliases_for_seed = tuple(cast(str, row["arm_alias"]) for row in rows if row["seed"] == seed)
        assert aliases_for_seed == d1_review_roster_alias_order(seed)


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

    reordered = packet.manifest
    _roster_rows(reordered, "confirmatory").reverse()
    with pytest.raises(D1PacketBlindnessError, match="randomized roster order"):
        validate_d1_packet_blindness(
            D1BlindReviewPacket.from_manifest(reordered), decision, aggregate, aliases
        )

    record_drift = packet.manifest
    _randomization(record_drift)["confirmatory_order_sha256"] = "0" * 64
    with pytest.raises(D1PacketBlindnessError, match="randomized roster order or record differs"):
        validate_d1_packet_blindness(
            D1BlindReviewPacket.from_manifest(record_drift),
            decision,
            aggregate,
            aliases,
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
