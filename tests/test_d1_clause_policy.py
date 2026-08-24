"""D1 clause labels are mechanical projections of registered condition data."""

from __future__ import annotations

import pytest

from d1_launch_admission_fixtures import SEEDS, contract
from entropic_substrate.analysis.d1_clause_policy import (
    D1ClausePolicyError,
    D1ConditionStatus,
    evaluate_d1_clauses,
    parse_d1_clause_policy,
)
from entropic_substrate.analysis.d1_factorial import compute_d1_factorial
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.experiments.d1_primitives import D1Arm

from .test_d1_factorial_analysis import _cells, _design

pytestmark = pytest.mark.unit


def _condition(
    *,
    threshold: float,
    operator: str,
    source: str = "ARM_ENDPOINT_MEAN",
    contrast_id: str | None = None,
) -> dict[str, object]:
    return {
        "contrast_id": contrast_id,
        "estimator": "last_k_mean" if source == "ARM_ENDPOINT_MEAN" else None,
        "metric": "nats_per_byte" if source == "ARM_ENDPOINT_MEAN" else None,
        "operator": operator,
        "smoothing_window": 5 if source == "ARM_ENDPOINT_MEAN" else None,
        "source": source,
        "threshold": threshold,
    }


def _policy(*, future_threshold: float = 15.0) -> dict[str, object]:
    candidates = (D1Arm.SG, D1Arm.AC, D1Arm.BOTH)
    conditions = {
        "noncollapse": {
            arm.value: [_condition(threshold=13.0, operator="AT_LEAST")] for arm in candidates
        },
        "future_grounding": {
            arm.value: [_condition(threshold=future_threshold, operator="AT_LEAST")]
            for arm in candidates
        },
        "language": {
            arm.value: [_condition(threshold=20.0, operator="AT_MOST")] for arm in candidates
        },
    }
    return {
        "schema_version": "d1-analysis-policy-v1",
        "candidate_arms": [arm.value for arm in candidates],
        "clauses": {
            clause_id: {
                "registered_rule": f"d1.{clause_id}.v1",
                "conditions_by_arm": by_arm,
            }
            for clause_id, by_arm in conditions.items()
        },
    }


def _contract_with_policy(policy: dict[str, object]) -> D1DecisionContract:
    base = contract()
    constants = {"analysis_policy": policy, "primary": {"alpha": 0.025}, "seeds": list(SEEDS)}
    return D1DecisionContract(
        family_identity=base.family_identity,
        registered_constants=constants,
        arm_ids=base.arm_ids,
        contrast_ids=base.contrast_ids,
        clause_ids=base.clause_ids,
        contrast_rules=base.contrast_rules,
        clause_rules=base.clause_rules,
        exclusion_rules=base.exclusion_rules,
        clause_precedence=base.clause_precedence,
        verdict_vocabulary=base.verdict_vocabulary,
        decision_rules=base.decision_rules,
    )


def _analysis(decision: D1DecisionContract):
    return compute_d1_factorial(
        design=_design(decision),
        decision_contract=decision,
        cells=_cells(),
    )


def test_every_coequal_clause_is_evaluated_before_feasible_set_intersection() -> None:
    decision = _contract_with_policy(_policy())
    policy = parse_d1_clause_policy(decision)
    result = evaluate_d1_clauses(
        policy=policy,
        decision_contract=decision,
        factorial=_analysis(decision),
        cells=_cells(),
    )

    assert result.clause_precedence == decision.clause_precedence
    assert tuple(result.by_arm) == (D1Arm.SG, D1Arm.AC, D1Arm.BOTH)
    assert result.by_arm[D1Arm.SG]["noncollapse"].status is D1ConditionStatus.PASS
    assert result.by_arm[D1Arm.SG]["future_grounding"].status is D1ConditionStatus.FAIL
    assert result.by_arm[D1Arm.SG]["language"].status is D1ConditionStatus.PASS
    assert result.by_arm[D1Arm.BOTH]["future_grounding"].status is D1ConditionStatus.PASS
    assert result.feasible_arms == (D1Arm.BOTH,)
    assert all(
        value.evidence_class == "DERIVED_FROM_MEASURED"
        for clauses in result.by_arm.values()
        for value in clauses.values()
    )


def test_empty_feasible_set_is_a_first_class_result_not_an_exception() -> None:
    decision = _contract_with_policy(_policy(future_threshold=30.0))
    result = evaluate_d1_clauses(
        policy=parse_d1_clause_policy(decision),
        decision_contract=decision,
        factorial=_analysis(decision),
        cells=_cells(),
    )

    assert result.feasible_arms == ()
    assert all(
        clauses["future_grounding"].status is D1ConditionStatus.FAIL
        for clauses in result.by_arm.values()
    )


def test_registered_contrast_interval_can_return_inconclusive() -> None:
    policy = _policy()
    clauses = policy["clauses"]
    assert isinstance(clauses, dict)
    future = clauses["future_grounding"]
    assert isinstance(future, dict)
    by_arm = future["conditions_by_arm"]
    assert isinstance(by_arm, dict)
    by_arm["sg"] = [
        _condition(
            threshold=0.0,
            operator="AT_LEAST",
            source="REGISTERED_CONTRAST_INTERVAL",
            contrast_id="sg-main",
        )
    ]
    decision = _contract_with_policy(policy)
    result = evaluate_d1_clauses(
        policy=parse_d1_clause_policy(decision),
        decision_contract=decision,
        factorial=_analysis(decision),
        cells=_cells(),
    )

    clause = result.by_arm[D1Arm.SG]["future_grounding"]
    assert clause.status is D1ConditionStatus.INCONCLUSIVE
    assert clause.conditions[0].observed_lower == pytest.approx(-0.195941650636283)
    assert clause.conditions[0].observed_upper == pytest.approx(5.195941650636283)


def test_missing_or_drifted_registered_policy_fails_without_defaults() -> None:
    with pytest.raises(D1ClausePolicyError, match="analysis_policy"):
        parse_d1_clause_policy(contract())

    policy = _policy()
    clauses = policy["clauses"]
    assert isinstance(clauses, dict)
    rule = clauses["noncollapse"]
    assert isinstance(rule, dict)
    rule["registered_rule"] = "different.rule"
    with pytest.raises(D1ClausePolicyError, match="registered_rule"):
        parse_d1_clause_policy(_contract_with_policy(policy))


def test_unregistered_contrast_reference_is_refused_at_parse_time() -> None:
    policy = _policy()
    clauses = policy["clauses"]
    assert isinstance(clauses, dict)
    rule = clauses["future_grounding"]
    assert isinstance(rule, dict)
    by_arm = rule["conditions_by_arm"]
    assert isinstance(by_arm, dict)
    by_arm["sg"] = [
        _condition(
            threshold=0.0,
            operator="AT_LEAST",
            source="REGISTERED_CONTRAST_INTERVAL",
            contrast_id="not-registered",
        )
    ]
    with pytest.raises(D1ClausePolicyError, match="contrast_id"):
        parse_d1_clause_policy(_contract_with_policy(policy))
