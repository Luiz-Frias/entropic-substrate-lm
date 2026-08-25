"""Role-masked D1 four-arm review packet, prompt, and structural leak scan."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from entropic_substrate.analysis.d1_blind_aliases import D1PreparedBlindAliases
from entropic_substrate.analysis.d1_execution_receipt import D1AnalysisExecutionReceipt
from entropic_substrate.analysis.d1_machine_analysis import D1AliasedMachineAnalysis
from entropic_substrate.evidence.d1_capture_aggregate import D1HeterogeneousCaptureAggregate
from entropic_substrate.evidence.d1_capture_artifacts import D1ArtifactIdentity
from entropic_substrate.evidence.d1_capture_roster import (
    D1DeclaredAttempt,
    D1ExpectedRun,
)
from entropic_substrate.evidence.d1_capture_validation import (
    D1CaptureBundleError,
    validate_d1_run_identity,
)
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.evidence.d1_review_adjudication import (
    D1_NO_MAJOR_PROMPT_DIRECTIVE,
    validate_d1_review_outcome_prompt,
)
from entropic_substrate.evidence.d1_telemetry_values import D1RunEvidence
from entropic_substrate.experiments.d1_primitives import (
    D1Arm,
    canonical_digest,
    canonical_json,
)
from entropic_substrate.experiments.d1_registered_design import decision_contract_digest

__all__ = [
    "D1BlindReviewPacket",
    "D1BlindnessLabel",
    "D1FactorialReviewerPrompt",
    "D1PacketBlindnessError",
    "build_d1_blind_review_packet",
    "build_d1_factorial_reviewer_prompt",
    "d1_review_roster_alias_order",
    "validate_d1_packet_blindness",
]

_SCHEMA = "d1-blind-review-packet-v1"
_PROHIBITED_KEYS = frozenset(
    {
        "arm",
        "arm_id",
        "construction_digest",
        "logical_run_id",
        "target_update",
        "collapse_strategy",
        "ema",
    }
)
_ARM_WORDS = frozenset(arm.value.casefold() for arm in D1Arm)
_NONBLINDABLE = (
    {
        "field": "run_attestations[].segments[].accelerator/provider/runtime",
        "reason": "required execution-parity evidence may link to operator records",
    },
    {
        "field": "run_attestations[].segments[].timestamps",
        "reason": "required run-start chronology may link to provider records",
    },
    {
        "field": "run_attestations[].segments[].quoted_rate",
        "reason": "required resource provenance may identify an execution lane",
    },
    {
        "field": "capture_roster[].capture_lane",
        "reason": "failure/completion role is reviewable integrity evidence",
    },
)
_BLIND_ALIASES = ("A", "B", "C", "D")
_ROSTER_ALGORITHM = "sha256-registered-seed-alias-sort-v1"
_RUN_IDENTITY_ATTESTATIONS = (
    ("I01", "aggregation_digest"),
    ("I02", "arm"),
    ("I03", "code_commit"),
    ("I04", "construction_digest"),
    ("I05", "dataset_digest"),
    ("I06", "execution_digest"),
    ("I07", "family_digest"),
    ("I08", "image_digest"),
    ("I09", "logical_run_id"),
    ("I10", "pairing_id"),
    ("I11", "plan_digest"),
    ("I12", "preregistration"),
    ("I13", "repository_digest"),
    ("I14", "run_class"),
    ("I15", "sampler_construction_digest"),
    ("I16", "seed"),
    ("I17", "segment_logical_run_id"),
    ("I18", "split_digest"),
)


class D1BlindnessLabel(StrEnum):
    ROLE_MASKED = "ROLE_MASKED_NOT_FULLY_BLIND"


class D1PacketBlindnessError(ValueError):
    """The packet leaks or structurally drifts from its four-arm registration."""


def _canonical(value: object) -> str:
    return canonical_json(value)


def d1_review_roster_alias_order(seed: int) -> tuple[str, ...]:
    """Derive one deterministic blind-alias order from a registered seed."""
    if type(seed) is not int or seed < 0:
        raise ValueError("D1 review roster requires a nonnegative registered seed")
    return tuple(
        sorted(
            _BLIND_ALIASES,
            key=lambda alias: hashlib.sha256(
                f"d1-review-roster-v1:{seed}:{alias}".encode()
            ).digest(),
        )
    )


@dataclass(frozen=True, slots=True)
class D1BlindReviewPacket:
    canonical_json: str

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> D1BlindReviewPacket:
        return cls(_canonical(manifest))

    @property
    def manifest(self) -> dict[str, object]:
        value = json.loads(self.canonical_json)
        if type(value) is not dict:
            raise AssertionError("D1 review packet lost its object root")
        return cast(dict[str, object], value)

    @property
    def packet_digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class D1FactorialReviewerPrompt:
    text: str

    def __post_init__(self) -> None:
        validate_d1_review_outcome_prompt(self.text)

    @property
    def prompt_digest(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


def _artifact(item: D1ArtifactIdentity, alias: str) -> dict[str, object]:
    return {
        "artifact_alias": alias,
        "dvc_object_id": item.dvc.object_id,
        "dvc_pointer_sha256": item.dvc.pointer_sha256,
        "md5": item.md5,
        "role": item.role.value,
        "sha256": item.sha256,
        "size_bytes": item.size_bytes,
    }


def _segments(run: D1RunEvidence) -> list[dict[str, object]]:
    return [
        {
            "accelerator_count": start.accelerator_count,
            "accelerator_model": start.accelerator_model,
            "compiler_enabled": start.compiler_enabled,
            "compiler_version": start.compiler_version,
            "cuda_version": start.cuda_version,
            "driver_version": start.driver_version,
            "image_digest": start.image_digest,
            "market": start.market,
            "pod_created_at": start.pod_created_at.isoformat(),
            "provider": start.provider,
            "provider_sku": start.provider_sku,
            "purchase_option": start.purchase_option.value,
            "pytorch_version": start.pytorch_version,
            "quoted_rate": start.quoted_rate.manifest,
            "region": start.region,
            "resource_activation": start.resource_activation.manifest,
            "resume_checkpoint_present": start.resume_checkpoint_sha256 is not None,
            "segment_index": start.execution_segment_index,
            "segment_start_mode": start.segment_start_mode.value,
            "trainer_started_at": start.trainer_started_at.isoformat(),
        }
        for start in run.segments
    ]


def _source_artifacts(aggregate: D1HeterogeneousCaptureAggregate) -> list[dict[str, object]]:
    return [
        _artifact(item, f"I{index:03d}")
        for index, item in enumerate(aggregate.dvc_handoff.artifacts, start=1)
    ]


def _registered_seeds(contract: D1DecisionContract) -> tuple[int, ...]:
    seeds = contract.registered_constants.get("seeds")
    if (
        type(seeds) is not tuple
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("D1 review roster requires the exact registered seed tuple")
    return cast(tuple[int, ...], seeds)


def _expected_order(
    expected: tuple[D1ExpectedRun, ...],
    aliases: D1PreparedBlindAliases,
    registered_seeds: tuple[int, ...],
) -> tuple[D1ExpectedRun, ...]:
    if {item.seed for item in expected} != set(registered_seeds):
        raise ValueError("D1 review roster seed set differs from the registered seeds")
    seed_rank = {seed: index for index, seed in enumerate(registered_seeds)}
    alias_rank = {
        seed: {alias: index for index, alias in enumerate(d1_review_roster_alias_order(seed))}
        for seed in registered_seeds
    }
    return tuple(
        sorted(
            expected,
            key=lambda item: (
                seed_rank[item.seed],
                alias_rank[item.seed][aliases.aliases.for_arm(item.arm)],
            ),
        )
    )


def _attempt_order(
    attempts: tuple[D1DeclaredAttempt, ...], aliases: D1PreparedBlindAliases
) -> tuple[D1DeclaredAttempt, ...]:
    return tuple(
        sorted(
            attempts,
            key=lambda item: (
                item.seed,
                d1_review_roster_alias_order(item.seed).index(aliases.aliases.for_arm(item.arm)),
                item.lane.value,
                item.run_class.value,
                item.disposition,
            ),
        )
    )


def _roster(
    aggregate: D1HeterogeneousCaptureAggregate,
    aliases: D1PreparedBlindAliases,
    registered_seeds: tuple[int, ...],
) -> tuple[dict[str, object], dict[str, str], dict[str, str], tuple[D1ExpectedRun, ...]]:
    expected = _expected_order(
        aggregate.roster.expected_confirmatory,
        aliases,
        registered_seeds,
    )
    run_aliases = {item.logical_run_id: f"R{index:03d}" for index, item in enumerate(expected, 1)}
    pair_aliases = {
        value: f"P{index:03d}" for index, value in enumerate(aggregate.roster.pairing_ids, 1)
    }
    complete = {item.expected.logical_run_id for item in aggregate.complete_runs}
    confirmatory = [
        {
            "arm_alias": aliases.aliases.for_arm(item.arm),
            "capture_lane": "complete" if item.logical_run_id in complete else "terminal_defect",
            "pair_alias": pair_aliases[item.pairing_id],
            "run_alias": run_aliases[item.logical_run_id],
            "seed": item.seed,
        }
        for item in expected
    ]
    attempts_by_id = {item.attempt.logical_run_id: item for item in aggregate.attempts}
    if set(attempts_by_id) != {item.logical_run_id for item in aggregate.roster.attempts}:
        raise ValueError("D1 packet requires the complete attempt evidence roster")
    ordered_attempts = _attempt_order(aggregate.roster.attempts, aliases)
    attempts = [
        {
            "arm_alias": aliases.aliases.for_arm(item.arm),
            "attempt_alias": f"T{index:03d}",
            "evidence_gap_present": attempts_by_id[item.logical_run_id].evidence_gap is not None,
            "lane": item.lane.value,
            "run_class": item.run_class.value,
            "seed": item.seed,
        }
        for index, item in enumerate(ordered_attempts, 1)
    ]
    return (
        {"attempts": attempts, "confirmatory": confirmatory},
        run_aliases,
        pair_aliases,
        expected,
    )


def _roster_randomization(roster: Mapping[str, object]) -> dict[str, object]:
    return {
        "algorithm": _ROSTER_ALGORITHM,
        "attempt_order_sha256": canonical_digest({"order": roster["attempts"]}),
        "confirmatory_order_sha256": canonical_digest({"order": roster["confirmatory"]}),
    }


def _run_attestations(
    aggregate: D1HeterogeneousCaptureAggregate,
    aliases: D1PreparedBlindAliases,
    run_evidence: tuple[D1RunEvidence, ...],
    run_aliases: Mapping[str, str],
    pair_aliases: Mapping[str, str],
    expected: tuple[D1ExpectedRun, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    evidence = {item.identity.logical_run_id: item for item in run_evidence}
    if len(evidence) != len(run_evidence) or set(evidence) != {
        item.logical_run_id for item in expected
    }:
        raise ValueError("D1 packet run identity roster differs from every expected cell")
    rows: list[dict[str, object]] = []
    for item in expected:
        run = evidence[item.logical_run_id]
        try:
            validate_d1_run_identity(run, item, aggregate.roster)
        except D1CaptureBundleError as error:
            raise ValueError("D1 packet run identity differs from the launch roster") from error
        rows.append(
            {
                "arm_alias": aliases.aliases.for_arm(item.arm),
                "pair_alias": pair_aliases[item.pairing_id],
                "registered_construction_match": True,
                "registered_identity_match": {
                    attestation_id: True for attestation_id, _ in _RUN_IDENTITY_ATTESTATIONS
                },
                "run_alias": run_aliases[item.logical_run_id],
                "seed": item.seed,
                "segments": _segments(run),
                "status": run.status.value,
            }
        )
    parity: list[dict[str, object]] = []
    fields = (
        "code_commit",
        "repository_digest",
        "image_digest",
        "dataset_digest",
        "split_digest",
        "sampler_construction_digest",
    )
    for pair_id in aggregate.roster.pairing_ids:
        block = tuple(item for item in expected if item.pairing_id == pair_id)
        parity.append(
            {
                "pair_alias": pair_aliases[pair_id],
                "registered_identity_parity": {
                    field: len({getattr(item, field) for item in block}) == 1 for field in fields
                },
                "segment_count_equal": len(
                    {len(evidence[item.logical_run_id].segments) for item in block}
                )
                == 1,
            }
        )
    return rows, parity


def _receipt_bindings(
    receipt: D1AnalysisExecutionReceipt,
    analysis: D1AliasedMachineAnalysis,
    aggregate: D1HeterogeneousCaptureAggregate,
    aliases: D1PreparedBlindAliases,
    decision_contract: D1DecisionContract,
) -> tuple[dict[str, object], dict[str, object]]:
    data = receipt.manifest
    if data.get("filing_eligible") is not True:
        raise ValueError("D1 review packet requires a filing-eligible analysis receipt")
    if data.get("analysis_digest") != analysis.analysis_digest:
        raise ValueError("D1 review packet analysis receipt differs from analysis")
    if data.get("alias_commitment") != aliases.aliases.sealed_commitment:
        raise ValueError("D1 review packet alias commitment differs")
    bindings = data.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("D1 review packet analysis bindings are absent")
    if bindings.get("capture_digest") != aggregate.manifest_digest:
        raise ValueError("D1 review packet capture binding differs")
    analysis_bindings = analysis.manifest.get("bindings")
    if not isinstance(analysis_bindings, dict) or analysis_bindings != bindings:
        raise ValueError("D1 review packet analysis and receipt bindings differ")
    if bindings.get("decision_contract_digest") != decision_contract_digest(decision_contract):
        raise ValueError("D1 review packet decision contract binding differs")
    implementation = cast(dict[str, object], data.get("implementation"))
    readers = cast(dict[str, object], implementation.get("readers"))
    if set(readers) != {"complete", "terminal_defect"}:
        raise ValueError("D1 review packet requires both validated reader identities")
    return bindings, readers


def build_d1_blind_review_packet(
    *,
    decision_contract: D1DecisionContract,
    aggregate: D1HeterogeneousCaptureAggregate,
    analysis: D1AliasedMachineAnalysis,
    analysis_receipt: D1AnalysisExecutionReceipt,
    aliases: D1PreparedBlindAliases,
    run_evidence: tuple[D1RunEvidence, ...],
) -> D1BlindReviewPacket:
    """Assemble exact evidence without serializing canonical mechanism identities."""
    bindings, readers = _receipt_bindings(
        analysis_receipt,
        analysis,
        aggregate,
        aliases,
        decision_contract,
    )
    registered_seeds = _registered_seeds(decision_contract)
    roster, run_aliases, pair_aliases, expected = _roster(
        aggregate,
        aliases,
        registered_seeds,
    )
    runs, parity = _run_attestations(
        aggregate,
        aliases,
        run_evidence,
        run_aliases,
        pair_aliases,
        expected,
    )
    manifest: dict[str, object] = {
        "alias_commitment": aliases.aliases.sealed_commitment,
        "analysis": analysis.manifest,
        "analysis_receipt_digest": analysis_receipt.receipt_digest,
        "bindings": bindings,
        "blindness_label": D1BlindnessLabel.ROLE_MASKED.value,
        "capture_roster": roster,
        "nonblindable_fields": list(_NONBLINDABLE),
        "parity_attestations": parity,
        "run_attestations": runs,
        "roster_randomization": _roster_randomization(roster),
        "schema_version": _SCHEMA,
        "source_artifacts": _source_artifacts(aggregate),
        "validated_readers": readers,
    }
    packet = D1BlindReviewPacket.from_manifest(manifest)
    validate_d1_packet_blindness(packet, decision_contract, aggregate, aliases)
    return packet


def _walk(value: object) -> tuple[set[str], set[str]]:
    keys: set[str] = set()
    strings: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            child_keys, child_strings = _walk(child)
            keys |= child_keys
            strings |= child_strings
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            child_keys, child_strings = _walk(child)
            keys |= child_keys
            strings |= child_strings
    elif isinstance(value, str):
        strings.add(value)
    return keys, strings


def validate_d1_packet_blindness(
    packet: D1BlindReviewPacket,
    contract: D1DecisionContract,
    aggregate: D1HeterogeneousCaptureAggregate,
    aliases: D1PreparedBlindAliases,
) -> None:
    """Use schema, coefficient, key, token, and digest checks redundantly."""
    manifest = packet.manifest
    bindings = cast(Mapping[str, object], manifest.get("bindings", {}))
    if (
        manifest.get("schema_version") != _SCHEMA
        or bindings.get("capture_digest") != aggregate.manifest_digest
        or bindings.get("decision_contract_digest") != decision_contract_digest(contract)
    ):
        raise D1PacketBlindnessError("packet schema, capture, or decision contract differs")
    keys, strings = _walk(manifest)
    if leaked := sorted(keys & _PROHIBITED_KEYS):
        raise D1PacketBlindnessError(f"construction-bearing field leaked: {leaked[0]}")
    arm_tokens = {
        token.casefold() for value in strings for token in re.findall(r"[A-Za-z0-9_]+", value)
    }
    if arm_tokens & _ARM_WORDS:
        raise D1PacketBlindnessError("canonical arm identity leaked into packet")
    if strings & set(contract.family_identity.arm_construction_digests.values()):
        raise D1PacketBlindnessError("canonical construction digest leaked into packet")
    analysis = cast(dict[str, object], manifest["analysis"])
    contrasts = cast(list[dict[str, object]], analysis["contrasts"])
    if len(contrasts) != len(contract.contrast_ids):
        raise D1PacketBlindnessError("registered contrast count differs")
    for index, contrast_id in enumerate(contract.contrast_ids, start=1):
        expected = [
            {
                "arm_alias": aliases.aliases.for_arm(D1Arm(term.arm_id)),
                "coefficient": term.coefficient,
            }
            for term in contract.contrast_rules[contrast_id].terms
        ]
        observed = contrasts[index - 1]
        if observed.get("contrast_alias") != f"C{index:02d}" or observed.get("terms") != expected:
            raise D1PacketBlindnessError(
                "registered contrast coefficients were not exactly aliased"
            )
    roster = cast(dict[str, list[object]], manifest["capture_roster"])
    expected_roster, _, _, _ = _roster(
        aggregate,
        aliases,
        _registered_seeds(contract),
    )
    if roster != expected_roster or manifest.get("roster_randomization") != (
        _roster_randomization(expected_roster)
    ):
        raise D1PacketBlindnessError("randomized roster order or record differs")
    runs = cast(list[dict[str, object]], manifest.get("run_attestations", []))
    expected_identity = {attestation_id: True for attestation_id, _ in _RUN_IDENTITY_ATTESTATIONS}
    if len(runs) != len(aggregate.roster.expected_confirmatory) or any(
        row.get("registered_identity_match") != expected_identity for row in runs
    ):
        raise D1PacketBlindnessError("complete registered run identity attestation differs")


def build_d1_factorial_reviewer_prompt(
    packet: D1BlindReviewPacket, contract: D1DecisionContract
) -> D1FactorialReviewerPrompt:
    """Quote the registered D1 decision surface without revealing mechanism labels."""
    analysis = cast(dict[str, object], packet.manifest["analysis"])
    contrasts = cast(list[dict[str, object]], analysis["contrasts"])
    contrast_lines = [
        f"- C{index:02d}: {_canonical({'terms': row['terms'], 'confidence_level': rule.confidence_level, 'multiplicity': rule.multiplicity, 'margins': dict(rule.margins), 'decision_vocabulary': rule.decision_vocabulary})}"
        for index, (row, rule) in enumerate(
            zip(contrasts, contract.contrast_rules.values(), strict=True), 1
        )
    ]
    clause_lines = [
        f"- {index}. {clause_id}: {_canonical({'registered_rule': rule.registered_rule, 'decision_vocabulary': rule.decision_vocabulary, 'threshold': rule.threshold, 'margin': rule.margin})}"
        for index, (clause_id, rule) in enumerate(contract.clause_rules.items(), 1)
    ]
    text = "\n".join(
        [
            "D1 four-arm factorial blind review",
            f"Packet SHA-256: {packet.packet_digest}",
            f"Blindness: {packet.manifest['blindness_label']}",
            "Independently recompute only from the sealed packet. Do not use repository, tracking-server, operator, or prior-answer state.",
            "Apply every registered clause as a mandatory co-equal rider; use the quoted precedence only for presentation.",
            f"Registered verdict vocabulary: {contract.verdict_vocabulary}",
            "Registered aliased contrasts (coefficients are exact):",
            *contrast_lines,
            "Registered clauses/riders in exact precedence:",
            *clause_lines,
            D1_NO_MAJOR_PROMPT_DIRECTIVE,
            "Report integrity findings as stable MAJOR-N identifiers and list every open MAJOR explicitly.",
            "Do not authorize a successor gate; review ends with the report and finding IDs.",
        ]
    )
    return D1FactorialReviewerPrompt(text + "\n")
