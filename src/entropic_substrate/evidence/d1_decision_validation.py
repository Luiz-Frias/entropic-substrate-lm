"""Review adjudication and preregistered decision-table validation for D1 v2."""

from __future__ import annotations

from collections.abc import Mapping

from entropic_substrate.evidence.d1_contract import D1DecisionContract, constants_match
from entropic_substrate.evidence.d1_parsing import error, object_exact, string
from entropic_substrate.evidence.d1_types import (
    D1ArmEvidence,
    D1Clause,
    D1Contrast,
    D1Integrity,
    D1Review,
    D1ReviewAdjudication,
    D1Successor,
    JsonValue,
)


def parse_review(value: object) -> D1Review:
    review = object_exact(
        value,
        path="review",
        fields=frozenset({"open_major_findings", "adjudication"}),
    )
    findings = review["open_major_findings"]
    if not isinstance(findings, list) or any(
        not isinstance(item, str) or not item for item in findings
    ):
        raise error(
            "invalid_major_findings",
            "review.open_major_findings",
            "must be a list of nonempty identifiers, including [] when none are open",
        )
    adjudication = object_exact(
        review["adjudication"],
        path="review.adjudication",
        fields=frozenset({"status", "major_findings"}),
    )
    adjudication_status = string(adjudication["status"], path="review.adjudication.status")
    if adjudication_status not in {"OPEN", "RESOLVED", "NOT_REQUIRED"}:
        raise error(
            "invalid_adjudication_status",
            "review.adjudication.status",
            "must be OPEN, RESOLVED, or NOT_REQUIRED",
        )
    major_findings = adjudication["major_findings"]
    if not isinstance(major_findings, list) or any(
        not isinstance(item, str) or not item for item in major_findings
    ):
        raise error(
            "invalid_adjudication",
            "review.adjudication.major_findings",
            "must be a list of nonempty identifiers",
        )
    open_findings = tuple(findings)
    adjudicated_findings = tuple(major_findings)
    if adjudication_status == "OPEN" and (
        not open_findings or set(open_findings) != set(adjudicated_findings)
    ):
        raise error(
            "inconsistent_adjudication",
            "review.adjudication",
            "OPEN adjudication must list exactly the open MAJOR findings",
        )
    if adjudication_status == "RESOLVED" and (open_findings or not adjudicated_findings):
        raise error(
            "inconsistent_adjudication",
            "review.adjudication",
            "RESOLVED adjudication requires no open MAJOR and at least one resolved MAJOR",
        )
    if adjudication_status == "NOT_REQUIRED" and (open_findings or adjudicated_findings):
        raise error(
            "inconsistent_adjudication",
            "review.adjudication",
            "NOT_REQUIRED adjudication requires no MAJOR findings",
        )
    return D1Review(
        open_major_findings=open_findings,
        adjudication=D1ReviewAdjudication(
            status=adjudication_status,
            major_findings=adjudicated_findings,
        ),
    )


def validate_decision_contract(
    *,
    contract: D1DecisionContract,
    constants: Mapping[str, JsonValue],
    arms: tuple[D1ArmEvidence, ...],
    contrasts: tuple[D1Contrast, ...],
    clauses: Mapping[str, D1Clause],
    integrity: D1Integrity,
    review: D1Review,
    verdict: str,
    successor: D1Successor,
) -> None:
    if not constants_match(constants, contract):
        raise error(
            "registered_constants_mismatch",
            "registered_constants",
            "must exactly match the preregistered decision contract",
        )
    if {arm.arm_id for arm in arms} != set(contract.arm_ids):
        raise error("arm_set_mismatch", "arms", "must exactly match the registered arm set")
    if {contrast.contrast_id for contrast in contrasts} != set(contract.contrast_ids):
        raise error(
            "contrast_set_mismatch",
            "contrasts",
            "must exactly match the registered contrast set",
        )
    if set(clauses) != set(contract.clause_ids):
        raise error(
            "clause_set_mismatch", "clauses", "must exactly match the registered clause set"
        )
    try:
        rule = contract.rule_for(verdict)
    except ValueError as exc:
        raise error("unregistered_decision", "verdict", str(exc)) from exc
    if successor.authorization is not rule.successor_authorization:
        raise error(
            "decision_authorization_mismatch",
            "successor.authorization",
            "does not match the registered decision-table rule",
        )
    if {name: clause.status for name, clause in clauses.items()} != dict(
        rule.required_clause_statuses
    ):
        raise error(
            "decision_clause_mismatch",
            "clauses",
            "do not satisfy the registered decision-table clause statuses",
        )
    if rule.require_integrity_pass and integrity.status != "PASS":
        raise error(
            "decision_integrity_mismatch",
            "integrity.status",
            "registered decision rule requires PASS integrity",
        )
    if rule.require_review_clearance and review.open_major_findings:
        raise error(
            "decision_review_mismatch",
            "review.open_major_findings",
            "registered decision rule requires review clearance",
        )
    if rule.require_eligible_strategies != bool(successor.eligible_strategies):
        raise error(
            "decision_strategy_mismatch",
            "successor.eligible_strategies",
            "does not match the registered decision-table rule",
        )
