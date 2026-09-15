"""Review adjudication and preregistered decision-table validation for D1 v2."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType

from entropic_substrate.evidence.d1_contract import D1DecisionContract, constants_match
from entropic_substrate.evidence.d1_parsing import (
    error,
    object_exact,
    sha256_digest,
    string,
)
from entropic_substrate.evidence.d1_types import (
    D1_STABILIZATION_ARM_IDS,
    D1ArmEvidence,
    D1Clause,
    D1Contrast,
    D1Integrity,
    D1Review,
    D1ReviewAdjudication,
    D1ReviewFindingDisposition,
    D1Successor,
    JsonValue,
)
from entropic_substrate.evidence.lifecycle import ContentHash, SuccessorAuthorization

_MAJOR_ID = re.compile(r"MAJOR-[1-9][0-9]*")


def _finding_ids(value: object, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or _MAJOR_ID.fullmatch(item) is None for item in value
    ):
        raise error("invalid_major_findings", path, "must be canonical MAJOR-N identifiers")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise error("invalid_major_findings", path, "must be unique")
    return result


def parse_review(value: object) -> D1Review:
    review = object_exact(
        value,
        path="review",
        fields=frozenset({"open_major_findings", "adjudication"}),
    )
    open_findings = _finding_ids(review["open_major_findings"], path="review.open_major_findings")
    adjudication = object_exact(
        review["adjudication"],
        path="review.adjudication",
        fields=frozenset(
            {
                "dispositions",
                "execution_receipt_sha256",
                "major_finding_ids",
                "open_major_finding_ids",
                "operator_record_sha256",
                "report_sha256",
                "review_packet_sha256",
                "review_request_sha256",
                "schema_version",
                "status",
            }
        ),
    )
    if adjudication["schema_version"] != "d1-review-adjudication-v1":
        raise error(
            "invalid_adjudication_schema",
            "review.adjudication.schema_version",
            "must be d1-review-adjudication-v1",
        )
    adjudication_status = string(adjudication["status"], path="review.adjudication.status")
    if adjudication_status not in {"OPEN", "RESOLVED", "NOT_REQUIRED"}:
        raise error(
            "invalid_adjudication_status",
            "review.adjudication.status",
            "must be OPEN, RESOLVED, or NOT_REQUIRED",
        )
    major_findings = _finding_ids(
        adjudication["major_finding_ids"], path="review.adjudication.major_finding_ids"
    )
    declared_open = _finding_ids(
        adjudication["open_major_finding_ids"],
        path="review.adjudication.open_major_finding_ids",
    )
    raw_dispositions = adjudication["dispositions"]
    if not isinstance(raw_dispositions, list):
        raise error(
            "invalid_adjudication",
            "review.adjudication.dispositions",
            "must be a list",
        )
    dispositions: dict[str, D1ReviewFindingDisposition] = {}
    for index, raw in enumerate(raw_dispositions):
        path = f"review.adjudication.dispositions[{index}]"
        item = object_exact(
            raw,
            path=path,
            fields=frozenset({"evidence_sha256", "finding_id", "rationale", "status"}),
        )
        finding_id = string(item["finding_id"], path=f"{path}.finding_id")
        if _MAJOR_ID.fullmatch(finding_id) is None or finding_id in dispositions:
            raise error(
                "invalid_adjudication", f"{path}.finding_id", "must be canonical and unique"
            )
        finding_status = string(item["status"], path=f"{path}.status")
        if finding_status not in {"OPEN", "RESOLVED"}:
            raise error("invalid_adjudication", f"{path}.status", "must be OPEN or RESOLVED")
        evidence = item["evidence_sha256"]
        evidence_sha256 = (
            None if evidence is None else sha256_digest(evidence, path=f"{path}.evidence_sha256")
        )
        if finding_status == "RESOLVED" and evidence_sha256 is None:
            raise error("invalid_adjudication", path, "resolved MAJOR requires evidence SHA-256")
        rationale = string(item["rationale"], path=f"{path}.rationale")
        dispositions[finding_id] = D1ReviewFindingDisposition(
            finding_id, finding_status, rationale, evidence_sha256
        )
    if tuple(dispositions) != major_findings:
        raise error(
            "inconsistent_adjudication",
            "review.adjudication.dispositions",
            "must exactly cover MAJOR findings in order",
        )
    expected_open = tuple(
        finding_id for finding_id, item in dispositions.items() if item.status == "OPEN"
    )
    expected_status = (
        "NOT_REQUIRED" if not major_findings else "OPEN" if expected_open else "RESOLVED"
    )
    if (
        declared_open != expected_open
        or open_findings != expected_open
        or adjudication_status != expected_status
    ):
        raise error(
            "inconsistent_adjudication",
            "review.adjudication",
            "status and open findings must derive exactly from dispositions",
        )
    return D1Review(
        open_major_findings=open_findings,
        adjudication=D1ReviewAdjudication(
            status=adjudication_status,
            major_findings=major_findings,
            dispositions=MappingProxyType(dispositions),
            execution_receipt_sha256=sha256_digest(
                adjudication["execution_receipt_sha256"],
                path="review.adjudication.execution_receipt_sha256",
            ),
            operator_record_sha256=sha256_digest(
                adjudication["operator_record_sha256"],
                path="review.adjudication.operator_record_sha256",
            ),
            report_sha256=sha256_digest(
                adjudication["report_sha256"], path="review.adjudication.report_sha256"
            ),
            review_packet_sha256=sha256_digest(
                adjudication["review_packet_sha256"],
                path="review.adjudication.review_packet_sha256",
            ),
            review_request_sha256=sha256_digest(
                adjudication["review_request_sha256"],
                path="review.adjudication.review_request_sha256",
            ),
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
    feasible_arms: tuple[str, ...],
    contrasts: tuple[D1Contrast, ...],
    clauses_by_arm: Mapping[str, Mapping[str, D1Clause]],
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
    if set(clauses_by_arm) != set(D1_STABILIZATION_ARM_IDS):
        raise error(
            "clause_arm_set_mismatch",
            "clauses_by_arm",
            "must exactly match the D1 stabilization arms",
        )
    if {contrast.contrast_id for contrast in contrasts} != set(contract.contrast_ids):
        raise error(
            "contrast_set_mismatch",
            "contrasts",
            "must exactly match the registered contrast set",
        )
    for contrast in contrasts:
        _validate_registered_contrast(contract, contrast)
    for arm_id, clauses in clauses_by_arm.items():
        if set(clauses) != set(contract.clause_ids):
            raise error(
                "clause_set_mismatch",
                f"clauses_by_arm.{arm_id}",
                "must exactly match the registered clause set",
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
                    f"clauses_by_arm.{arm_id}.{clause_id}",
                    "clause rule, vocabulary, threshold, and margin must match registration",
                )
    expected_feasible = tuple(
        arm_id
        for arm_id in D1_STABILIZATION_ARM_IDS
        if all(clause.status == "PASS" for clause in clauses_by_arm[arm_id].values())
    )
    if feasible_arms != expected_feasible:
        raise error(
            "feasible_arm_mismatch",
            "feasible_arms",
            "must equal every stabilization arm whose registered clauses all pass",
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
    matching_arms = tuple(
        arm_id
        for arm_id in D1_STABILIZATION_ARM_IDS
        if {name: clause.status for name, clause in clauses_by_arm[arm_id].items()}
        == dict(rule.required_clause_statuses)
    )
    if not matching_arms:
        raise error(
            "decision_clause_mismatch",
            "clauses_by_arm",
            "no stabilization arm satisfies the registered decision-table clause statuses",
        )
    if successor.authorization is not SuccessorAuthorization.AUTHORIZED:
        for candidate in contract.decision_rules.values():
            candidate_matches = any(
                {name: clause.status for name, clause in clauses_by_arm[arm_id].items()}
                == dict(candidate.required_clause_statuses)
                for arm_id in D1_STABILIZATION_ARM_IDS
            )
            integrity_ready = not candidate.require_integrity_pass or integrity.status == "PASS"
            review_ready = not candidate.require_review_clearance or not review.open_major_findings
            if (
                candidate.successor_authorization is SuccessorAuthorization.AUTHORIZED
                and candidate_matches
                and integrity_ready
                and review_ready
            ):
                raise error(
                    "decision_authorization_downgrade",
                    "verdict",
                    "cannot downgrade while a registered authorized arm is ready",
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
    expected_eligible_arms = (
        feasible_arms if successor.authorization is SuccessorAuthorization.AUTHORIZED else ()
    )
    if successor.eligible_arms != expected_eligible_arms:
        raise error(
            "decision_eligible_arm_mismatch",
            "successor.eligible_arms",
            "must carry the complete feasible frontier only when authorization is granted",
        )
    if successor.authorization is SuccessorAuthorization.AUTHORIZED and (
        not feasible_arms or set(matching_arms) != set(feasible_arms)
    ):
        raise error(
            "decision_feasible_arm_mismatch",
            "feasible_arms",
            "authorized decision statuses must identify the complete feasible frontier",
        )
