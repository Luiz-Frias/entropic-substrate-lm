"""Shared v2 D1 decision fixtures with no detached-filing dependency."""

from __future__ import annotations

from entropic_substrate.evidence.d1_contract import D1DecisionContract, D1DecisionRule
from entropic_substrate.evidence.d1_types import D1_ARM_IDS, D1FamilyIdentity
from entropic_substrate.evidence.lifecycle import SuccessorAuthorization


def digest(index: int, *, width: int = 64) -> str:
    return f"{index:0{width}x}"


def evidence_hash(index: int) -> dict[str, str]:
    return {"algorithm": "sha256", "digest": digest(index)}


def family_identity() -> D1FamilyIdentity:
    return D1FamilyIdentity(
        family_digest="f" * 64,
        arm_construction_digests={arm: digest(100 + index) for index, arm in enumerate(D1_ARM_IDS)},
    )


def decision_contract() -> D1DecisionContract:
    return D1DecisionContract(
        family_identity=family_identity(),
        registered_constants={"seeds": [1, 2, 3, 4], "primary": {"alpha": 0.025}},
        arm_ids=D1_ARM_IDS,
        contrast_ids=("sg-main", "ac-main", "interaction"),
        clause_ids=("noncollapse", "future_grounding", "language"),
        decision_rules={
            "SUPPORTED": D1DecisionRule(
                successor_authorization=SuccessorAuthorization.AUTHORIZED,
                required_clause_statuses={
                    "noncollapse": "PASS",
                    "future_grounding": "PASS",
                    "language": "PASS",
                },
                require_integrity_pass=True,
                require_review_clearance=True,
                require_eligible_strategies=True,
            ),
            "VALID_NEGATIVE": D1DecisionRule(
                successor_authorization=SuccessorAuthorization.VALID_NEGATIVE,
                required_clause_statuses={
                    "noncollapse": "FAIL",
                    "future_grounding": "INCONCLUSIVE",
                    "language": "PASS",
                },
                require_integrity_pass=True,
                require_review_clearance=True,
                require_eligible_strategies=False,
            ),
        },
    )


def machine_verdict(*, verdict: str = "SUPPORTED") -> dict[str, object]:
    contract = decision_contract()
    statuses = (
        {"noncollapse": "PASS", "future_grounding": "PASS", "language": "PASS"}
        if verdict == "SUPPORTED"
        else {"noncollapse": "FAIL", "future_grounding": "INCONCLUSIVE", "language": "PASS"}
    )
    return {
        "schema_version": "d1-verdict/v2",
        "rung": "d1",
        "preregistration": {
            "path": "docs/d1-preregistration.md",
            "commit": "a" * 40,
            "family_digest": contract.family_identity.family_digest,
        },
        "registered_constants": {"seeds": [1, 2, 3, 4], "primary": {"alpha": 0.025}},
        "arms": [
            {
                "arm_id": arm,
                "construction_digest": contract.family_identity.arm_construction_digests[arm],
            }
            for arm in D1_ARM_IDS
        ],
        "contrasts": [
            {
                "contrast_id": "sg-main",
                "contrast_kind": "PAIRWISE",
                "terms": [
                    {"arm_id": "sg", "coefficient": 1.0},
                    {"arm_id": "none", "coefficient": -1.0},
                ],
                "estimate": 0.0123456789012345,
                "sample_sd": 0.02,
                "interval_lower": -0.02,
                "interval_upper": 0.04,
                "confidence_level": 0.975,
                "n": 4,
                "df": 3,
                "evidence_class": "DERIVED_FROM_MEASURED",
                "decision_vocabulary": "d1-contrast/v1",
                "decision": "INCONCLUSIVE",
                "evidence_hash": evidence_hash(200),
            },
            {
                "contrast_id": "ac-main",
                "contrast_kind": "PAIRWISE",
                "terms": [
                    {"arm_id": "ac", "coefficient": 1.0},
                    {"arm_id": "none", "coefficient": -1.0},
                ],
                "estimate": 0.01,
                "sample_sd": 0.02,
                "interval_lower": -0.03,
                "interval_upper": 0.05,
                "confidence_level": 0.975,
                "n": 4,
                "df": 3,
                "evidence_class": "DERIVED_FROM_MEASURED",
                "decision_vocabulary": "d1-contrast/v1",
                "decision": "INCONCLUSIVE",
                "evidence_hash": evidence_hash(201),
            },
            {
                "contrast_id": "interaction",
                "contrast_kind": "FACTORIAL_INTERACTION",
                "terms": [
                    {"arm_id": "none", "coefficient": 1.0},
                    {"arm_id": "sg", "coefficient": -1.0},
                    {"arm_id": "ac", "coefficient": -1.0},
                    {"arm_id": "both", "coefficient": 1.0},
                ],
                "estimate": -0.003,
                "sample_sd": 0.02,
                "interval_lower": -0.04,
                "interval_upper": 0.03,
                "confidence_level": 0.975,
                "n": 4,
                "df": 3,
                "evidence_class": "DERIVED_FROM_MEASURED",
                "decision_vocabulary": "d1-contrast/v1",
                "decision": "INTERACTION_INCONCLUSIVE",
                "evidence_hash": evidence_hash(202),
            },
        ],
        "clauses": {
            name: {
                "registered_rule": f"d1.{name}.v1",
                "decision_vocabulary": "d1-clause/v1",
                "threshold": None,
                "margin": None,
                "status": statuses[name],
                "evidence_hash": evidence_hash(210 + index),
            }
            for index, name in enumerate(("noncollapse", "future_grounding", "language"))
        },
        "integrity": {
            "status": "PASS",
            "roster_complete": True,
            "pairing_verified": True,
            "data_order_verified": True,
            "completion_verified": True,
            "evidence_hash": evidence_hash(220),
        },
        "exclusions": [],
        "inputs": [{"path": "results/captured/d1.jsonl", "md5": digest(221, width=32)}],
        "review": {
            "open_major_findings": [],
            "adjudication": {"status": "NOT_REQUIRED", "major_findings": []},
        },
        "verdict": verdict,
        "successor": {
            "authorization": "AUTHORIZED" if verdict == "SUPPORTED" else "VALID_NEGATIVE",
            "eligible_strategies": ["d2-pullback"] if verdict == "SUPPORTED" else [],
        },
    }
