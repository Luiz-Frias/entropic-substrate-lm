"""D1 review packets preserve evidence while structurally hiding mechanisms."""

from __future__ import annotations

import copy
import pickle
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple, cast

import pytest

import entropic_substrate.ladder.review.d1_review_packet as review_packet_module
from entropic_substrate.kernel.evidence_class.resource_provenance import execution_segment_id
from entropic_substrate.kernel.identity.d1_primitives import (
    D1Arm,
    RunClass,
    canonical_digest,
    canonical_json,
    thaw_for_digest,
)
from entropic_substrate.kernel.verdict.d1_contract import D1DecisionContract
from entropic_substrate.ladder.analyze.d1_machine_analysis import D1AliasedMachineAnalysis
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate import (
    D1CapturePartitionProof,
    D1HeterogeneousCaptureAggregate,
)
from entropic_substrate.ladder.capture.custody.d1_capture_bundle import D1AttemptEvidence
from entropic_substrate.ladder.capture.custody.d1_capture_roster import (
    D1AttemptLane,
    D1DeclaredAttempt,
    D1ExpectedRun,
    D1LaunchRoster,
)
from entropic_substrate.ladder.capture.custody.d1_capture_validation import (
    D1_RUN_IDENTITY_ATTESTATION_PATHS,
    D1RunIdentityAttestation,
)
from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1RunEvidence, D1RunStatus
from entropic_substrate.ladder.register.d1_registered_design import decision_contract_digest
from entropic_substrate.ladder.review.d1_blind_aliases import (
    D1InMemoryAliasCustodyFake,
    D1PreparedBlindAliases,
    prepare_d1_blind_aliases,
)
from entropic_substrate.ladder.review.d1_execution_receipt import (
    D1AnalysisExecutionReceipt,
    issue_d1_verified_analysis_receipt,
)
from entropic_substrate.ladder.review.d1_review_adjudication import (
    D1_NO_MAJOR_OUTCOME,
    D1_NO_MAJOR_PROMPT_DIRECTIVE,
)
from entropic_substrate.ladder.review.d1_review_packet import (
    D1BlindnessLabel,
    D1BlindReviewPacket,
    D1PacketBlindnessError,
    D1ValidatedBlindReviewPacket,
    build_d1_blind_review_packet,
    build_d1_factorial_reviewer_prompt,
    d1_review_roster_alias_order,
    validate_d1_packet_blindness,
)

from .test_d1_analysis_receipt import _implementation
from .test_d1_blind_alias_custody import _DeterministicCrypto
from .test_d1_external_analysis_adapter_v2 import _adapter, _adapter_case

pytestmark = pytest.mark.unit


def _roster_rows(manifest: dict[str, object], lane: str) -> list[dict[str, object]]:
    roster = cast(dict[str, object], manifest["capture_roster"])
    return cast(list[dict[str, object]], roster[lane])


def _randomization(manifest: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], manifest["roster_randomization"])


def _rebind_analysis(
    analysis: D1AliasedMachineAnalysis,
    receipt: D1AnalysisExecutionReceipt,
    *,
    capture_digest: str,
    contract_digest: str,
) -> tuple[D1AliasedMachineAnalysis, D1AnalysisExecutionReceipt]:
    analysis_manifest = analysis.manifest
    analysis_bindings = cast(dict[str, object], analysis_manifest["bindings"])
    analysis_bindings["capture_digest"] = capture_digest
    analysis_bindings["decision_contract_digest"] = contract_digest
    rebound_analysis = D1AliasedMachineAnalysis(canonical_json(analysis_manifest))
    receipt_manifest = receipt.manifest
    receipt_bindings = cast(dict[str, object], receipt_manifest["bindings"])
    receipt_bindings["capture_digest"] = capture_digest
    receipt_bindings["decision_contract_digest"] = contract_digest
    receipt_manifest["analysis_digest"] = rebound_analysis.analysis_digest
    filing_bindings = cast(dict[str, object], receipt_manifest["filing_bindings"])
    filing_bindings["output_sha256"] = rebound_analysis.analysis_digest
    rebound_receipt = D1AnalysisExecutionReceipt(canonical_json(receipt_manifest))
    return rebound_analysis, rebound_receipt


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
    bound_analysis, bound_receipt = _rebind_analysis(
        analysis,
        receipt,
        capture_digest=with_attempts.manifest_digest,
        contract_digest=decision_contract_digest(decision),
    )
    return _BlindCase(
        decision,
        with_attempts,
        aliases,
        bound_analysis,
        bound_receipt,
        evidence,
    )


def _case_with_seed_blocks(case: _BlindCase) -> _BlindCase:
    decision, aggregate, aliases, analysis, receipt, evidence = case
    originals = aggregate.roster.expected_confirmatory
    evidence_by_construction = {item.identity.construction_digest: item for item in evidence}
    captures_by_arm = {item.expected.arm: item for item in aggregate.complete_runs}
    seed_eight: list[D1ExpectedRun] = []
    seed_eight_captures = []
    seed_eight_evidence = []
    for original in originals:
        source_run = evidence_by_construction[original.construction_digest]
        logical_run_id = f"{original.logical_run_id}-seed-8"
        plan = cast(dict[str, object], thaw_for_digest(source_run.plan_manifest))
        plan.update(
            {
                "logical_run_id": logical_run_id,
                "pairing_id": "pair-8",
                "seed": 8,
            }
        )
        plan_digest = canonical_digest(plan)
        expected = replace(
            original,
            logical_run_id=logical_run_id,
            seed=8,
            pairing_id="pair-8",
            plan_digest=plan_digest,
        )
        seed_eight.append(expected)
        source_capture = captures_by_arm[expected.arm]
        seed_eight_captures.append(
            replace(
                source_capture,
                expected=expected,
                capture_input=replace(
                    source_capture.capture_input,
                    logical_run_id=logical_run_id,
                ),
            )
        )
        segments = tuple(
            replace(
                segment,
                logical_run_id=logical_run_id,
                execution_segment_id=execution_segment_id(
                    logical_run_id,
                    segment.execution_segment_index,
                ),
            )
            for segment in source_run.segments
        )
        seed_eight_evidence.append(
            replace(
                source_run,
                identity=replace(source_run.identity, logical_run_id=logical_run_id),
                plan_digest=plan_digest,
                plan_manifest=plan,
                segments=segments,
            )
        )
    seed_eight_tuple = tuple(seed_eight)
    shuffled_expected = (*reversed(originals), *reversed(seed_eight_tuple))
    roster = D1LaunchRoster(
        preregistration_path=aggregate.roster.preregistration_path,
        preregistration_commit=aggregate.roster.preregistration_commit,
        family_digest=aggregate.roster.family_digest,
        expected_confirmatory=shuffled_expected,
        attempts=(),
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
    registered = cast(dict[str, object], thaw_for_digest(decision.registered_constants))
    registered["seeds"] = [8, 7]
    seeded_decision = replace(decision, registered_constants=registered)
    analysis_manifest = analysis.manifest
    analysis_manifest["included_seeds"] = [8, 7]
    analysis = D1AliasedMachineAnalysis(canonical_json(analysis_manifest))
    seeded_analysis, bound_receipt = _rebind_analysis(
        analysis,
        receipt,
        capture_digest=with_seed_blocks.manifest_digest,
        contract_digest=decision_contract_digest(seeded_decision),
    )
    return _BlindCase(
        seeded_decision,
        with_seed_blocks,
        aliases,
        seeded_analysis,
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

    assert type(packet) is D1ValidatedBlindReviewPacket
    packet.require_factory_attested()
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
    for row in cast(list[dict[str, object]], manifest["run_attestations"]):
        identity = cast(dict[str, bool], row["registered_identity_match"])
        assert set(identity) == {f"I{index:02d}" for index in range(1, 19)}
        assert all(identity.values())
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


def test_validated_packet_is_factory_attested_and_noncopyable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    with pytest.raises(TypeError, match="factory-attested"):
        D1ValidatedBlindReviewPacket.from_manifest(packet.manifest)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(packet)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(packet)
    with pytest.raises(TypeError, match="cannot be copied"):
        pickle.dumps(packet)
    with pytest.raises(TypeError, match="factory-attested"):
        replace(packet, canonical_json="{}")
    object.__setattr__(packet, "canonical_json", "{}")
    with pytest.raises(TypeError, match="evidence changed"):
        packet.require_factory_attested()


def test_packet_binds_the_supplied_decision_contract_to_analysis_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision, aggregate, aliases, analysis, receipt, evidence = _case(tmp_path, monkeypatch)
    drifted_contract = replace(
        decision,
        registered_constants=cast(
            dict[str, object], thaw_for_digest(decision.registered_constants)
        ),
        verdict_vocabulary="d1-drifted-vocabulary/v1",
    )
    with pytest.raises(ValueError, match="decision contract"):
        build_d1_blind_review_packet(
            decision_contract=drifted_contract,
            aggregate=aggregate,
            analysis=analysis,
            analysis_receipt=receipt,
            aliases=aliases,
            run_evidence=evidence,
        )
    bindings = cast(dict[str, str], receipt.manifest["bindings"])
    drifted_receipt = replace(
        receipt,
        canonical_json=receipt.canonical_json.replace(
            bindings["decision_contract_digest"],
            "0" * 64,
        ),
    )
    with pytest.raises(ValueError, match="bindings"):
        build_d1_blind_review_packet(
            decision_contract=decision,
            aggregate=aggregate,
            analysis=analysis,
            analysis_receipt=drifted_receipt,
            aliases=aliases,
            run_evidence=evidence,
        )


def test_packet_rejects_incomplete_validator_issued_identity_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision, aggregate, aliases, analysis, receipt, evidence = _case(tmp_path, monkeypatch)
    required = frozenset(
        path for paths in D1_RUN_IDENTITY_ATTESTATION_PATHS.values() for path in paths
    )
    incomplete = D1RunIdentityAttestation(required - {next(iter(required))})
    monkeypatch.setattr(
        review_packet_module,
        "validate_d1_run_identity",
        lambda *_args: incomplete,
    )

    with pytest.raises(ValueError, match="attestation is incomplete"):
        build_d1_blind_review_packet(
            decision_contract=decision,
            aggregate=aggregate,
            analysis=analysis,
            analysis_receipt=receipt,
            aliases=aliases,
            run_evidence=evidence,
        )
    with pytest.raises(ValueError, match="unregistered check"):
        D1RunIdentityAttestation(frozenset({"unknown.identity.path"}))
    with pytest.raises(TypeError, match="frozenset"):
        D1RunIdentityAttestation(cast(frozenset[str], []))


def _mutate_run_identity(run: D1RunEvidence, mutation: str) -> D1RunEvidence:
    if mutation == "status":
        return replace(run, status=D1RunStatus.PARTIAL)
    if mutation == "terminal":
        return replace(run, terminal=None)
    if mutation.startswith("journal_"):
        field = mutation.removeprefix("journal_")
        value: object = {
            "logical_run_id": "substituted-run",
            "execution_digest": "a" * 64,
            "aggregation_digest": "b" * 64,
            "construction_digest": "c" * 64,
            "family_digest": "d" * 64,
            "run_class": RunClass.PILOT,
        }[field]
        return replace(run, identity=replace(run.identity, **{field: value}))
    if mutation == "plan_digest":
        return replace(run, plan_digest="e" * 64)
    if mutation == "plan_unvalidated_key":
        plan = cast(dict[str, object], thaw_for_digest(run.plan_manifest))
        plan["unvalidated_identity"] = "drift"
        return replace(
            run,
            plan_digest=canonical_digest(plan),
            plan_manifest=plan,
        )
    if mutation.startswith("segment_"):
        field = mutation.removeprefix("segment_")
        segment = run.segments[0]
        if field == "image_digest":
            changed = replace(segment, image_digest="sha256:" + "f" * 64)
        else:
            changed = replace(
                segment,
                logical_run_id="substituted-run",
                execution_segment_id=execution_segment_id("substituted-run", 0),
            )
        return replace(run, segments=(changed, *run.segments[1:]))
    plan = cast(dict[str, object], thaw_for_digest(run.plan_manifest))
    path = mutation.removeprefix("plan_").split("__")
    target = plan
    for key in path[:-1]:
        target = cast(dict[str, object], target[key])
    field = path[-1]
    changed_arm = D1Arm.SG.value if target.get(field) != D1Arm.SG.value else D1Arm.NONE.value
    target[field] = {
        "arm": changed_arm,
        "seed": 999,
        "pairing_id": "substituted-pair",
        "logical_run_id": "substituted-run",
        "construction_digest": "c" * 64,
        "family_digest": "d" * 64,
        "run_class": RunClass.PILOT.value,
        "commit": "1234567",
        "repository_digest": "e" * 64,
        "image_digest": "sha256:" + "f" * 64,
        "dataset_digest": "1" * 64,
        "split_digest": "2" * 64,
        "sampler_construction_digest": "3" * 64,
        "path": "docs/substituted.md",
        "dvc_capture_required": False,
    }[field]
    return replace(
        run,
        plan_digest=canonical_digest(plan),
        plan_manifest=plan,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "journal_logical_run_id",
        "journal_execution_digest",
        "journal_aggregation_digest",
        "journal_construction_digest",
        "journal_family_digest",
        "journal_run_class",
        "status",
        "terminal",
        "plan_digest",
        "plan_unvalidated_key",
        "plan_arm",
        "plan_seed",
        "plan_pairing_id",
        "plan_logical_run_id",
        "plan_construction_digest",
        "plan_family_digest",
        "plan_run_class",
        "plan_code__commit",
        "plan_code__repository_digest",
        "plan_container__image_digest",
        "plan_data__dataset_digest",
        "plan_data__split_digest",
        "plan_data__sampler_construction_digest",
        "plan_preregistration__commit",
        "plan_preregistration__family_digest",
        "plan_preregistration__path",
        "plan_evidence__dvc_capture_required",
        "segment_image_digest",
        "segment_logical_run_id",
    ],
)
def test_packet_rejects_each_substituted_run_identity_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    decision, aggregate, aliases, analysis, receipt, evidence = _case(tmp_path, monkeypatch)
    changed = (_mutate_run_identity(evidence[0], mutation), *evidence[1:])
    with pytest.raises(ValueError, match="run identity"):
        build_d1_blind_review_packet(
            decision_contract=decision,
            aggregate=aggregate,
            analysis=analysis,
            analysis_receipt=receipt,
            aliases=aliases,
            run_evidence=changed,
        )


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

    identity_drift = packet.manifest
    runs = cast(list[dict[str, object]], identity_drift["run_attestations"])
    identity = cast(dict[str, bool], runs[0]["registered_identity_match"])
    identity["I01"] = False
    with pytest.raises(D1PacketBlindnessError, match="complete registered run identity"):
        validate_d1_packet_blindness(
            D1BlindReviewPacket.from_manifest(identity_drift),
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
    nonfiling_manifest = receipt.manifest
    nonfiling_manifest["filing_eligible"] = False
    nonfiling_manifest["filing_bindings"] = None
    with pytest.raises(ValueError, match="filing-eligible"):
        build_d1_blind_review_packet(
            decision_contract=decision,
            aggregate=aggregate,
            analysis=analysis,
            analysis_receipt=D1AnalysisExecutionReceipt(canonical_json(nonfiling_manifest)),
            aliases=aliases,
            run_evidence=evidence,
        )


def test_packet_manifest_exact_key_set_guards_retired_validated_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard-teeth: validated_readers retirement is not re-introducible.

    Reintroducing the retired "validated_readers" key into the packet manifest
    must cause this test to fail, pinning the exact-key-set assertion.
    """
    decision, aggregate, aliases, analysis, receipt, evidence = _case(tmp_path, monkeypatch)

    packet = build_d1_blind_review_packet(
        decision_contract=decision,
        aggregate=aggregate,
        analysis=analysis,
        analysis_receipt=receipt,
        aliases=aliases,
        run_evidence=evidence,
    )

    # Expected keys in the blind review packet manifest (H0 exit).
    # Retired: "validated_readers" (D1 closure, measured 2026-09-20).
    expected_keys = frozenset(
        {
            "alias_commitment",
            "analysis",
            "analysis_receipt_digest",
            "bindings",
            "blindness_label",
            "capture_roster",
            "nonblindable_fields",
            "parity_attestations",
            "run_attestations",
            "roster_randomization",
            "schema_version",
            "source_artifacts",
        }
    )
    assert frozenset(packet.manifest) == expected_keys
