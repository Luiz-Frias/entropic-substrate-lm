"""Assemble a filed real-arm D1 verdict from one attested blind lineage."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NoReturn, SupportsIndex, cast

from entropic_substrate.kernel.identity.d1_primitives import (
    D1Arm,
    canonical_digest,
    thaw_for_digest,
)
from entropic_substrate.kernel.identity.factory_attestation import FactoryAttestation
from entropic_substrate.kernel.lifecycle import SuccessorAuthorization
from entropic_substrate.kernel.verdict.d1_contract import D1DecisionContract
from entropic_substrate.kernel.verdict.d1_statistical_gaps import (
    D1EstimateGap,
    D1GapCell,
    parse_estimate_gap,
)
from entropic_substrate.kernel.verdict.d1_types import (
    D1_STABILIZATION_ARM_IDS,
    VERDICTS,
    ClauseStatus,
    ContrastDecision,
    ContrastKind,
)
from entropic_substrate.ladder.file.d1_filing import canonical_machine_verdict_bytes
from entropic_substrate.ladder.file.d1_filing_graph import (
    D1RecomputedIntegrity,
    recompute_capture_integrity,
)
from entropic_substrate.ladder.file.d1_filing_receipts import parse_alias_reveal
from entropic_substrate.ladder.file.d1_validator import ValidatedD1Verdict
from entropic_substrate.ladder.review.d1_machine_analysis_verification import (
    D1VerifiedMachineAnalysis,
    verify_d1_machine_analysis,
)
from entropic_substrate.ladder.review.d1_review_adjudication import (
    D1ReviewAdjudicationSurface,
    parse_d1_major_findings,
    parse_d1_review_execution_receipt,
)
from entropic_substrate.ladder.review.d1_review_execution import (
    D1ExecutedBlindReview,
    D1ReleasedAliasReveal,
)
from entropic_substrate.ladder.review.d1_review_packet import D1ValidatedBlindReviewPacket


@dataclass(frozen=True, slots=True)
class D1VerdictDecision:
    verdict: str

    def __post_init__(self) -> None:
        if type(self.verdict) is not str or self.verdict not in VERDICTS:
            raise ValueError("D1 verdict decision is unregistered")


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class D1AssembledRealArmVerdict:
    """Validated payload retaining every attested input needed by filing."""

    verified: D1VerifiedMachineAnalysis
    execution: D1ExecutedBlindReview
    reveal: D1ReleasedAliasReveal
    adjudication: D1ReviewAdjudicationSurface
    verdict: ValidatedD1Verdict
    canonical_bytes: bytes = field(repr=False)

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("D1AssembledRealArmVerdict is factory-attested")

    def require_factory_attested(self) -> None:
        if not _ASSEMBLY_ATTESTATION.registered(self):
            raise TypeError("D1 assembled real-arm verdict is not factory-attested")
        if not _ASSEMBLY_ATTESTATION.contains(self):
            raise TypeError("D1 assembled real-arm verdict evidence changed after minting")
        _validate_assembled(self)

    def __copy__(self) -> NoReturn:
        raise TypeError("D1AssembledRealArmVerdict cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        raise TypeError("D1AssembledRealArmVerdict cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("D1AssembledRealArmVerdict cannot be copied")

    @property
    def manifest(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self.canonical_bytes))


def _assembly_fingerprint(value: D1AssembledRealArmVerdict) -> str:
    return canonical_digest(
        {
            "adjudication": value.adjudication.manifest,
            "aggregate_digest": value.verified.aggregate.manifest_digest,
            "analysis_digest": value.verified.analysis_digest,
            "canonical_sha256": hashlib.sha256(value.canonical_bytes).hexdigest(),
            "execution_receipt": value.execution.receipt.receipt_digest,
            "reveal_sha256": hashlib.sha256(value.reveal.payload).hexdigest(),
        }
    )


_ASSEMBLY_ATTESTATION = FactoryAttestation[
    D1AssembledRealArmVerdict,
    str,
](_assembly_fingerprint)


def _canonical(value: object) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"D1 verdict {label} must be an object")
    return cast(dict[str, object], value)


def _rows(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"D1 verdict {label} must be an array")
    return cast(list[object], value)


def _evidence_hash(*, analysis_digest: str, role: str, value: object) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "digest": canonical_digest(
            {"analysis_digest": analysis_digest, "role": role, "value": value}
        ),
    }


def _contrast_decision(kind: str, row: Mapping[str, object], margin: float) -> str:
    lower, upper = row["interval_lower"], row["interval_upper"]
    if lower is None or upper is None:
        return (
            ContrastDecision.INTERACTION_INCONCLUSIVE.value
            if kind == ContrastKind.FACTORIAL_INTERACTION
            else ContrastDecision.INCONCLUSIVE.value
        )
    low, high = float(cast(int | float, lower)), float(cast(int | float, upper))
    outside = low > margin or high < -margin
    if kind == ContrastKind.FACTORIAL_INTERACTION:
        return (
            ContrastDecision.INTERACTION_DETECTED.value
            if outside
            else ContrastDecision.INTERACTION_INCONCLUSIVE.value
        )
    if low > margin:
        return ContrastDecision.FAVORS_LEFT.value
    if high < -margin:
        return ContrastDecision.FAVORS_RIGHT.value
    return (
        ContrastDecision.PRACTICALLY_NULL.value
        if low >= -margin and high <= margin
        else ContrastDecision.INCONCLUSIVE.value
    )


def _contrast_rows(
    verified: D1VerifiedMachineAnalysis, contract: D1DecisionContract
) -> list[dict[str, object]]:
    source = _rows(verified.manifest["contrasts"], "contrasts")
    if len(source) != len(contract.contrast_ids):
        raise ValueError("D1 verdict contrast count differs from registration")
    result: list[dict[str, object]] = []
    for index, (contrast_id, raw) in enumerate(zip(contract.contrast_ids, source, strict=True)):
        row, rule = _mapping(raw, f"contrast[{index}]"), contract.contrast_rules[contrast_id]
        margin_value = rule.margins["practical"]
        if not math.isfinite(margin_value) or margin_value < 0.0:
            raise ValueError("D1 contrast practical margin must be finite and nonnegative")
        filed: dict[str, object] = {
            "confidence_level": row["confidence_level"],
            "contrast_id": contrast_id,
            "contrast_kind": rule.contrast_kind,
            "decision": _contrast_decision(rule.contrast_kind, row, margin_value),
            "decision_vocabulary": rule.decision_vocabulary,
            "df": row["df"],
            "endpoint": {
                "estimator": rule.endpoint.estimator,
                "metric": rule.endpoint.metric,
                "smoothing_window": rule.endpoint.smoothing_window,
            },
            "estimate": row["estimate"],
            "evidence_class": row["evidence_class"],
            "interval_lower": row["interval_lower"],
            "interval_upper": row["interval_upper"],
            "margins": dict(rule.margins),
            "multiplicity": row["multiplicity"],
            "n": row["n"],
            "sample_sd": row["sample_sd"],
            "terms": [
                {"arm_id": item.arm_id, "coefficient": item.coefficient} for item in rule.terms
            ],
        }
        if "unavailable" in row:
            gap = parse_estimate_gap(row["unavailable"], aliases=True)
            mapping = {
                verified.expected.alias_application.aliases.for_arm(arm): arm.value for arm in D1Arm
            }
            filed["unavailable"] = D1EstimateGap(
                tuple(D1GapCell(mapping[cell.arm], cell.seed, cell.endpoint) for cell in gap.cells)
            ).manifest()
        filed["evidence_hash"] = _evidence_hash(
            analysis_digest=verified.analysis_digest,
            role=f"contrast:{contrast_id}",
            value=filed,
        )
        result.append(filed)
    return result


def _clause_rows(verified: D1VerifiedMachineAnalysis) -> dict[str, dict[str, object]]:
    contract = verified.expected.decision_contract
    source = _mapping(verified.manifest["clauses_by_alias"], "clauses_by_alias")
    result: dict[str, dict[str, object]] = {}
    for arm_id in D1_STABILIZATION_ARM_IDS:
        alias = verified.expected.alias_application.aliases.for_arm(D1Arm(arm_id))
        by_clause = _mapping(source[alias], f"clauses for {alias}")
        clauses: dict[str, object] = {}
        for clause_id in contract.clause_precedence:
            registered, observed = (
                contract.clause_rules[clause_id],
                _mapping(by_clause[clause_id], clause_id),
            )
            filed: dict[str, object] = {
                "decision_vocabulary": registered.decision_vocabulary,
                "margin": registered.margin,
                "registered_rule": registered.registered_rule,
                "status": observed["status"],
                "threshold": registered.threshold,
            }
            filed["evidence_hash"] = _evidence_hash(
                analysis_digest=verified.analysis_digest,
                role=f"clause:{arm_id}:{clause_id}",
                value=observed,
            )
            clauses[clause_id] = filed
        result[arm_id] = clauses
    return result


def _review_lineage(
    verified: D1VerifiedMachineAnalysis,
    execution: D1ExecutedBlindReview,
    reveal: D1ReleasedAliasReveal,
    adjudication: D1ReviewAdjudicationSurface,
) -> None:
    if (
        type(execution) is not D1ExecutedBlindReview
        or type(reveal) is not D1ReleasedAliasReveal
        or type(adjudication) is not D1ReviewAdjudicationSurface
    ):
        raise TypeError("D1 review lineage requires exact factory-attested evidence types")
    execution.require_factory_attested()
    reveal.require_factory_attested()
    adjudication.require_factory_attested()
    receipt = parse_d1_review_execution_receipt(execution.receipt.canonical_json.encode())
    packet = execution.request.packet
    if type(packet) is not D1ValidatedBlindReviewPacket:
        raise TypeError("D1 verdict assembly requires a validated review packet")
    packet.require_factory_attested()
    if (
        packet.decision_contract is not verified.expected.decision_contract
        or packet.aggregate is not verified.aggregate
        or packet.aliases.aliases != verified.expected.alias_application.aliases
    ):
        raise ValueError("D1 validated review packet authority differs from verified analysis")
    if (
        receipt.packet_sha256 != packet.packet_digest
        or receipt.report_sha256 != hashlib.sha256(execution.report).hexdigest()
        or receipt.request_digest != execution.request.request_digest
    ):
        raise ValueError("D1 review execution receipt differs from execution evidence")
    if (
        reveal.review_receipt_sha256 != execution.receipt.receipt_digest
        or reveal.review_packet_sha256 != packet.packet_digest
    ):
        raise ValueError("D1 review lineage differs between execution and reveal")
    if _canonical(packet.manifest.get("analysis")) != _canonical(verified.manifest):
        raise ValueError("D1 reviewed packet analysis differs from verified analysis")
    commitment = verified.expected.alias_application.commitment_sha256
    if (
        packet.manifest.get("alias_commitment") != commitment
        or reveal.alias_commitment_sha256 != commitment
    ):
        raise ValueError("D1 review lineage alias commitment differs")
    report = parse_d1_major_findings(
        execution.report,
        reviewer_prompt=execution.request.prompt.text.encode(),
    )
    if (
        adjudication.execution_receipt_sha256 != execution.receipt.receipt_digest
        or adjudication.review_request_sha256 != execution.request.request_digest
        or adjudication.review_packet_sha256 != packet.packet_digest
    ):
        raise ValueError("D1 adjudication execution identity differs from reviewed execution")
    if (
        adjudication.report_sha256 != report.report_sha256
        or adjudication.major_finding_ids != report.major_finding_ids
    ):
        raise ValueError("D1 adjudication differs from the executed review report")


def _feasible_arms(
    verified: D1VerifiedMachineAnalysis, reveal: D1ReleasedAliasReveal
) -> tuple[str, ...]:
    mapping = parse_alias_reveal(reveal.payload).mapping
    aliases = cast(list[str], verified.manifest["feasible_aliases"])
    real = {mapping[alias] for alias in aliases}
    expected = set(D1_STABILIZATION_ARM_IDS)
    if len(real) != len(aliases) or not real <= expected:
        raise ValueError("D1 feasible aliases do not de-alias to stabilization arms")
    return tuple(arm_id for arm_id in D1_STABILIZATION_ARM_IDS if arm_id in real)


def _integrity_manifest(integrity: D1RecomputedIntegrity) -> dict[str, object]:
    return {
        "completion_verified": integrity.completion_verified,
        "data_order_verified": integrity.data_order_verified,
        "evidence_hash": {"algorithm": "sha256", "digest": integrity.evidence_digest},
        "pairing_verified": integrity.pairing_verified,
        "roster_complete": integrity.roster_complete,
        "status": ClauseStatus.PASS.value if integrity.passed else ClauseStatus.FAIL.value,
    }


def _validate_assembled(value: D1AssembledRealArmVerdict) -> None:
    _review_lineage(value.verified, value.execution, value.reveal, value.adjudication)
    reverified = verify_d1_machine_analysis(
        binding=value.verified.binding,
        aggregate=value.verified.aggregate,
    )
    if reverified.analysis_digest != value.verified.analysis_digest:
        raise ValueError("D1 assembled verdict analysis changed after verification")
    if canonical_machine_verdict_bytes(value.manifest) != value.canonical_bytes:
        raise ValueError("D1 assembled verdict bytes are not canonical")
    parsed = ValidatedD1Verdict.from_mapping(
        value.manifest,
        decision_contract=value.verified.expected.decision_contract,
    )
    if parsed != value.verdict:
        raise ValueError("D1 assembled verdict value differs from canonical bytes")


def _mint_assembled(
    *,
    verified: D1VerifiedMachineAnalysis,
    execution: D1ExecutedBlindReview,
    reveal: D1ReleasedAliasReveal,
    adjudication: D1ReviewAdjudicationSurface,
    verdict: ValidatedD1Verdict,
    canonical_bytes: bytes,
) -> D1AssembledRealArmVerdict:
    value = object.__new__(D1AssembledRealArmVerdict)
    for name, item in locals().items():
        if name not in {"value"}:
            object.__setattr__(value, name, item)
    _validate_assembled(value)
    _ASSEMBLY_ATTESTATION.register(value)
    return value


def assemble_d1_real_arm_verdict(
    *,
    verified: D1VerifiedMachineAnalysis,
    execution: D1ExecutedBlindReview,
    reveal: D1ReleasedAliasReveal,
    adjudication: D1ReviewAdjudicationSurface,
    decision: D1VerdictDecision,
) -> D1AssembledRealArmVerdict:
    if (
        type(verified) is not D1VerifiedMachineAnalysis
        or type(execution) is not D1ExecutedBlindReview
        or type(reveal) is not D1ReleasedAliasReveal
        or type(adjudication) is not D1ReviewAdjudicationSurface
        or type(decision) is not D1VerdictDecision
    ):
        raise TypeError("D1 verdict assembly requires exact attested evidence types")
    _review_lineage(verified, execution, reveal, adjudication)
    alias_control = verified.expected.alias_application
    alias_control.require_factory_attested()
    opened = parse_alias_reveal(reveal.payload)
    if dict(opened.mapping) != dict(alias_control.reveal.mapping):
        raise ValueError("D1 released alias mapping differs from verified analysis")
    contract = verified.expected.decision_contract
    rule = contract.rule_for(decision.verdict)
    feasible = _feasible_arms(verified, reveal)
    integrity = recompute_capture_integrity(verified.aggregate, contract)
    if integrity.applied_exclusions is None:
        raise ValueError("D1 verdict exclusions lack registered authority")
    if not verified.aggregate.dvc_handoff.analysis_ready:
        raise ValueError("D1 verdict inputs are not analysis-ready")
    constants = thaw_for_digest(contract.registered_constants)
    if type(constants) is not dict:
        raise TypeError("D1 registered constants did not thaw to an object")
    manifest: dict[str, object] = {
        "arms": [
            {
                "arm_id": arm_id,
                "construction_digest": contract.family_identity.arm_construction_digests[arm_id],
            }
            for arm_id in contract.arm_ids
        ],
        "clause_precedence": list(contract.clause_precedence),
        "clauses_by_arm": _clause_rows(verified),
        "contrasts": _contrast_rows(verified, contract),
        "exclusions": [
            {
                "evidence_hash": {"algorithm": "sha256", "digest": digest},
                "reason": reason,
                "subject": claim.pairing_id,
            }
            for claim, reason, digest in integrity.applied_exclusions
        ],
        "feasible_arms": list(feasible),
        "inputs": [
            {"md5": item.md5, "path": item.normalized_path, "sha256": item.sha256}
            for item in verified.aggregate.dvc_handoff.artifacts
        ],
        "integrity": _integrity_manifest(integrity),
        "preregistration": {
            "commit": verified.expected.design.preregistration.commit,
            "family_digest": verified.expected.design.family_identity.family_digest,
            "path": verified.expected.design.preregistration.path,
        },
        "registered_constants": constants,
        "review": {
            "adjudication": adjudication.manifest,
            "open_major_findings": list(adjudication.open_major_finding_ids),
        },
        "rung": "d1",
        "schema_version": "d1-verdict/v3"
        if any(
            "unavailable" in _mapping(row, "contrast")
            for row in _rows(verified.manifest["contrasts"], "contrasts")
        )
        else "d1-verdict/v2",
        "successor": {
            "authorization": rule.successor_authorization.value,
            "eligible_arms": (
                list(feasible)
                if rule.successor_authorization is SuccessorAuthorization.AUTHORIZED
                else []
            ),
            "eligible_strategies": list(rule.eligible_strategies),
        },
        "verdict": decision.verdict,
        "verdict_vocabulary": contract.verdict_vocabulary,
    }
    canonical = canonical_machine_verdict_bytes(manifest)
    verdict = ValidatedD1Verdict.from_mapping(manifest, decision_contract=contract)
    return _mint_assembled(
        verified=verified,
        execution=execution,
        reveal=reveal,
        adjudication=adjudication,
        verdict=verdict,
        canonical_bytes=canonical,
    )
