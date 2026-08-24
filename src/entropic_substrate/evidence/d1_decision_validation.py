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
from entropic_substrate.evidence.lifecycle import ContentHash


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


def _validate_registered_contrast(contract: D1DecisionContract, contrast: D1Contrast) -> None:
    rule = contract.contrast_rules[contrast.contrast_id]
    if contrast.contrast_kind != rule.contrast_kind or contrast.terms != rule.terms:
        raise error(
            "registered_contrast_terms_mismatch",
            f"contrasts.{contrast.contrast_id}.terms",
            "contrast kind, term order, coefficients, and signs must match registration",
        )
    if (
        contrast.endpoint.metric != rule.endpoint.metric
        or contrast.endpoint.estimator != rule.endpoint.estimator
    ):
        raise error(
            "registered_endpoint_mismatch",
            f"contrasts.{contrast.contrast_id}.endpoint",
            "endpoint metric and estimator must match registration",
        )
    if contrast.endpoint.smoothing_window != rule.endpoint.smoothing_window:
        raise error(
            "registered_smoothing_mismatch",
            f"contrasts.{contrast.contrast_id}.endpoint.smoothing_window",
            "smoothing window must match registration",
        )
    if rule.n_rule == "EXACT" and contrast.n != rule.registered_n:
        raise error(
            "registered_sample_size_mismatch",
            f"contrasts.{contrast.contrast_id}.n",
            "sample size must match the exact registered sample size",
        )
    if rule.df_rule == "N_MINUS_ONE" and contrast.df != contrast.n - 1:
        raise error(
            "registered_df_mismatch",
            f"contrasts.{contrast.contrast_id}.df",
            "degrees of freedom must equal n - 1",
        )
    if rule.df_rule == "NOT_APPLICABLE" and contrast.df is not None:
        raise error(
            "registered_df_mismatch",
            f"contrasts.{contrast.contrast_id}.df",
            "degrees of freedom must be null under the registered rule",
        )
    if contrast.confidence_level != rule.confidence_level:
        raise error(
            "registered_confidence_mismatch",
            f"contrasts.{contrast.contrast_id}.confidence_level",
            "confidence level must match registration",
        )
    if contrast.multiplicity != rule.multiplicity:
        raise error(
            "registered_multiplicity_mismatch",
            f"contrasts.{contrast.contrast_id}.multiplicity",
            "multiplicity method must match registration",
        )
    if contrast.margins != rule.margins:
        raise error(
            "registered_margins_mismatch",
            f"contrasts.{contrast.contrast_id}.margins",
            "all practical margins must match registration",
        )
    if contrast.decision_vocabulary != rule.decision_vocabulary:
        raise error(
            "registered_contrast_vocabulary_mismatch",
            f"contrasts.{contrast.contrast_id}.decision_vocabulary",
            "contrast decision vocabulary must match registration",
        )


def validate_decision_contract(
    *,
    contract: D1DecisionContract,
    constants: Mapping[str, JsonValue],
    arms: tuple[D1ArmEvidence, ...],
    contrasts: tuple[D1Contrast, ...],
    clauses: Mapping[str, D1Clause],
    integrity: D1Integrity,
    exclusions: tuple[tuple[str, str, ContentHash], ...],
    review: D1Review,
    clause_precedence: tuple[str, ...],
    verdict_vocabulary: str,
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
    for contrast in contrasts:
        _validate_registered_contrast(contract, contrast)
    if set(clauses) != set(contract.clause_ids):
        raise error(
            "clause_set_mismatch", "clauses", "must exactly match the registered clause set"
        )
    for clause_id, clause in clauses.items():
        registered = contract.clause_rules[clause_id]
        if (
            clause.registered_rule != registered.registered_rule
            or clause.decision_vocabulary != registered.decision_vocabulary
            or clause.threshold != registered.threshold
            or clause.margin != registered.margin
        ):
            raise error(
                "registered_clause_mismatch",
                f"clauses.{clause_id}",
                "clause rule, vocabulary, threshold, and margin must match registration",
            )
    if clause_precedence != contract.clause_precedence:
        raise error(
            "registered_clause_precedence_mismatch",
            "clause_precedence",
            "clause precedence must match registration exactly",
        )
    if verdict_vocabulary != contract.verdict_vocabulary:
        raise error(
            "registered_verdict_vocabulary_mismatch",
            "verdict_vocabulary",
            "verdict vocabulary must match registration exactly",
        )
    allowed_exclusions = {registered.reason_id for registered in contract.exclusion_rules}
    for index, (_, reason, _) in enumerate(exclusions):
        if reason not in allowed_exclusions:
            raise error(
                "unregistered_exclusion",
                f"exclusions[{index}].reason",
                "exclusion reason is not present in the registered exclusion policy",
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
    if successor.eligible_strategies != rule.eligible_strategies:
        raise error(
            "decision_strategy_mismatch",
            "successor.eligible_strategies",
            "strategy list does not match the registered successor mapping",
        )
