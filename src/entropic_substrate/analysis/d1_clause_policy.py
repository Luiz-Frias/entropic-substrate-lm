"""Registered, data-driven D1 clause mechanics with no scientific defaults."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import NoReturn, cast

from entropic_substrate.analysis.d1_factorial import (
    D1AnalysisCell,
    D1FactorialAnalysis,
    d1_cell_endpoint,
)
from entropic_substrate.evidence.d1_contract import D1DecisionContract, D1RegisteredEndpoint
from entropic_substrate.evidence.d1_statistical_gaps import D1EndpointGap
from entropic_substrate.evidence.d1_telemetry_values import (
    D1_REPORT_ONLY_METRICS,
    D1_SCALAR_METRICS,
)
from entropic_substrate.evidence.d1_types import D1_STABILIZATION_ARM_IDS
from entropic_substrate.experiments.d1_primitives import D1Arm

__all__ = [
    "D1ArmClauseResult",
    "D1ClauseAnalysis",
    "D1ClausePolicy",
    "D1ClausePolicyError",
    "D1ConditionResult",
    "D1ConditionSource",
    "D1ConditionStatus",
    "D1RegisteredCondition",
    "D1ThresholdOperator",
    "classify_d1_condition_status",
    "evaluate_d1_clauses",
    "parse_d1_clause_policy",
]

_POLICY_SCHEMA = "d1-analysis-policy-v1"
_EVIDENCE_CLASS = "DERIVED_FROM_MEASURED"
_CONDITION_FIELDS = frozenset(
    {
        "contrast_id",
        "estimator",
        "metric",
        "operator",
        "smoothing_window",
        "source",
        "threshold",
    }
)


class D1ClausePolicyError(ValueError):
    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        super().__init__(f"D1 clause policy {code} at {path}: {detail}")


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise D1ClausePolicyError(code=code, path=path, detail=detail)


class D1ConditionSource(StrEnum):
    ARM_ENDPOINT_MEAN = "ARM_ENDPOINT_MEAN"
    REGISTERED_CONTRAST_INTERVAL = "REGISTERED_CONTRAST_INTERVAL"


class D1ThresholdOperator(StrEnum):
    AT_LEAST = "AT_LEAST"
    AT_MOST = "AT_MOST"
    WITHIN_ABS = "WITHIN_ABS"
    OUTSIDE_ABS = "OUTSIDE_ABS"


class D1ConditionStatus(StrEnum):
    PASS = "PASS"  # nosec B105 -- registered decision vocabulary, not a credential
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class D1RegisteredCondition:
    source: D1ConditionSource
    operator: D1ThresholdOperator
    threshold: float
    metric: str | None
    estimator: str | None
    smoothing_window: int | None
    contrast_id: str | None

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "estimator": self.estimator,
            "metric": self.metric,
            "operator": self.operator.value,
            "smoothing_window": self.smoothing_window,
            "source": self.source.value,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class D1ClausePolicy:
    candidate_arms: tuple[D1Arm, ...]
    conditions: Mapping[str, Mapping[D1Arm, tuple[D1RegisteredCondition, ...]]]


@dataclass(frozen=True, slots=True)
class D1ConditionResult:
    condition: D1RegisteredCondition
    status: D1ConditionStatus
    observed_point: float | None
    observed_lower: float | None
    observed_upper: float | None
    evidence_class: str = _EVIDENCE_CLASS


@dataclass(frozen=True, slots=True)
class D1ArmClauseResult:
    clause_id: str
    registered_rule: str
    status: D1ConditionStatus
    conditions: tuple[D1ConditionResult, ...]
    evidence_class: str = _EVIDENCE_CLASS


@dataclass(frozen=True, slots=True)
class D1ClauseAnalysis:
    clause_precedence: tuple[str, ...]
    by_arm: Mapping[D1Arm, Mapping[str, D1ArmClauseResult]]
    feasible_arms: tuple[D1Arm, ...]


def _object(value: object, *, path: str, fields: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        _fail("type", path, "must be an object with string keys")
    typed = cast(Mapping[str, object], value)
    if frozenset(typed) != fields:
        _fail("fields", path, f"must contain exactly {sorted(fields)}")
    return typed


def _condition(
    value: object,
    *,
    path: str,
    contract: D1DecisionContract,
) -> D1RegisteredCondition:
    item = _object(value, path=path, fields=_CONDITION_FIELDS)
    try:
        source = D1ConditionSource(item["source"])
        operator = D1ThresholdOperator(item["operator"])
    except (TypeError, ValueError) as error:
        _fail("vocabulary", path, str(error))
    threshold = item["threshold"]
    if isinstance(threshold, bool) or type(threshold) not in (int, float):
        _fail("threshold", f"{path}.threshold", "must be numeric")
    threshold = float(cast(int | float, threshold))
    if not math.isfinite(threshold) or threshold < 0.0:
        _fail("threshold", f"{path}.threshold", "must be finite and nonnegative")
    if source is D1ConditionSource.ARM_ENDPOINT_MEAN:
        metric, estimator, window = item["metric"], item["estimator"], item["smoothing_window"]
        if type(metric) is not str or not metric or estimator != "last_k_mean":
            _fail("endpoint", path, "requires metric and estimator='last_k_mean'")
        if metric not in D1_SCALAR_METRICS or metric in D1_REPORT_ONLY_METRICS:
            _fail("endpoint", path, "metric must be a registered deciding scalar")
        if type(window) is not int or window < 1 or item["contrast_id"] is not None:
            _fail("endpoint", path, "requires positive smoothing_window and null contrast_id")
        return D1RegisteredCondition(
            source,
            operator,
            threshold,
            metric,
            "last_k_mean",
            window,
            None,
        )
    contrast_id = item["contrast_id"]
    if (
        type(contrast_id) is not str
        or contrast_id not in contract.contrast_rules
        or any(item[field] is not None for field in ("metric", "estimator", "smoothing_window"))
    ):
        _fail("contrast_id", path, "must name one registered contrast and no endpoint fields")
    contrast_metric = contract.contrast_rules[contrast_id].endpoint.metric
    if contrast_metric not in D1_SCALAR_METRICS or contrast_metric in D1_REPORT_ONLY_METRICS:
        _fail("contrast_id", path, "contrast endpoint must use a registered deciding scalar")
    return D1RegisteredCondition(source, operator, threshold, None, None, None, contrast_id)


def parse_d1_clause_policy(contract: D1DecisionContract) -> D1ClausePolicy:
    """Parse the policy already covered by the decision-contract digest."""
    if type(contract) is not D1DecisionContract:
        raise TypeError("D1 clause policy requires D1DecisionContract")
    raw = contract.registered_constants.get("analysis_policy")
    root = _object(
        raw,
        path="registered_constants.analysis_policy",
        fields=frozenset({"candidate_arms", "clauses", "schema_version"}),
    )
    if root["schema_version"] != _POLICY_SCHEMA:
        _fail("schema", "analysis_policy.schema_version", f"must equal {_POLICY_SCHEMA}")
    raw_arms = root["candidate_arms"]
    if type(raw_arms) is not tuple or not raw_arms:
        _fail("candidate_arms", "analysis_policy.candidate_arms", "must be a nonempty list")
    try:
        arms = tuple(D1Arm(value) for value in raw_arms)
    except (TypeError, ValueError) as error:
        _fail("candidate_arms", "analysis_policy.candidate_arms", str(error))
    if len(set(arms)) != len(arms):
        _fail("candidate_arms", "analysis_policy.candidate_arms", "must be unique")
    expected_arms = tuple(D1Arm(value) for value in D1_STABILIZATION_ARM_IDS)
    if arms != expected_arms:
        _fail(
            "candidate_arms",
            "analysis_policy.candidate_arms",
            f"must equal canonical stabilization sequence {D1_STABILIZATION_ARM_IDS}",
        )
    raw_clauses = root["clauses"]
    if not isinstance(raw_clauses, Mapping) or set(raw_clauses) != set(contract.clause_ids):
        _fail("clauses", "analysis_policy.clauses", "must bind every registered clause")
    clauses: dict[str, Mapping[D1Arm, tuple[D1RegisteredCondition, ...]]] = {}
    for clause_id in contract.clause_precedence:
        entry = _object(
            raw_clauses[clause_id],
            path=f"analysis_policy.clauses.{clause_id}",
            fields=frozenset({"conditions_by_arm", "registered_rule"}),
        )
        registered = contract.clause_rules[clause_id]
        if entry["registered_rule"] != registered.registered_rule:
            _fail("registered_rule", clause_id, "differs from the registered clause")
        by_arm = entry["conditions_by_arm"]
        if not isinstance(by_arm, Mapping) or set(by_arm) != {arm.value for arm in arms}:
            _fail("conditions_by_arm", clause_id, "must bind every candidate arm exactly once")
        parsed: dict[D1Arm, tuple[D1RegisteredCondition, ...]] = {}
        for arm in arms:
            values = by_arm[arm.value]
            if type(values) is not tuple or not values:
                _fail("conditions", f"{clause_id}.{arm.value}", "must be a nonempty list")
            parsed[arm] = tuple(
                _condition(item, path=f"{clause_id}.{arm.value}[{index}]", contract=contract)
                for index, item in enumerate(values)
            )
        declared = {condition.threshold for values in parsed.values() for condition in values}
        for name in ("threshold", "margin"):
            scalar = getattr(registered, name)
            if scalar is not None and float(scalar) not in declared:
                _fail("registered_rule", clause_id, f"{name} is absent from policy conditions")
        clauses[clause_id] = MappingProxyType(parsed)
    return D1ClausePolicy(arms, MappingProxyType(clauses))


def _endpoint(
    condition: D1RegisteredCondition, arm: D1Arm, cells: tuple[D1AnalysisCell, ...]
) -> float | None:
    metric = condition.metric
    smoothing_window = condition.smoothing_window
    if metric is None or smoothing_window is None:
        _fail("condition_state", arm.value, "endpoint condition is incomplete")
    values: list[float] = []
    for cell in cells:
        if cell.arm is not arm:
            continue
        point = d1_cell_endpoint(
            cell, D1RegisteredEndpoint(metric, "last_k_mean", smoothing_window)
        )
        if type(point) is D1EndpointGap:
            return None
        values.append(cast(float, point))
    return None if not values else math.fsum(values) / len(values)


def classify_d1_condition_status(
    operator: D1ThresholdOperator,
    threshold: float,
    lower: float | None,
    upper: float | None,
) -> D1ConditionStatus:
    if lower is None or upper is None:
        return D1ConditionStatus.INCONCLUSIVE
    if operator is D1ThresholdOperator.AT_LEAST:
        return (
            D1ConditionStatus.PASS
            if lower >= threshold
            else (D1ConditionStatus.FAIL if upper < threshold else D1ConditionStatus.INCONCLUSIVE)
        )
    if operator is D1ThresholdOperator.AT_MOST:
        return (
            D1ConditionStatus.PASS
            if upper <= threshold
            else (D1ConditionStatus.FAIL if lower > threshold else D1ConditionStatus.INCONCLUSIVE)
        )
    within = lower >= -threshold and upper <= threshold
    outside = lower > threshold or upper < -threshold
    if operator is D1ThresholdOperator.WITHIN_ABS:
        return (
            D1ConditionStatus.PASS
            if within
            else (D1ConditionStatus.FAIL if outside else D1ConditionStatus.INCONCLUSIVE)
        )
    return (
        D1ConditionStatus.PASS
        if outside
        else (D1ConditionStatus.FAIL if within else D1ConditionStatus.INCONCLUSIVE)
    )


def _evaluate_condition(
    condition: D1RegisteredCondition,
    arm: D1Arm,
    factorial: D1FactorialAnalysis | None,
    cells: tuple[D1AnalysisCell, ...],
) -> D1ConditionResult:
    if condition.source is D1ConditionSource.ARM_ENDPOINT_MEAN:
        point = _endpoint(condition, arm, cells)
        lower = upper = point
    else:
        if condition.contrast_id is None:
            _fail("condition_state", arm.value, "contrast condition is incomplete")
        contrast = (
            None if factorial is None else factorial.registered_contrasts.get(condition.contrast_id)
        )
        point = None if contrast is None else contrast.estimate
        lower = None if contrast is None else contrast.interval_lower
        upper = None if contrast is None else contrast.interval_upper
    return D1ConditionResult(
        condition=condition,
        status=classify_d1_condition_status(
            condition.operator,
            condition.threshold,
            lower,
            upper,
        ),
        observed_point=point,
        observed_lower=lower,
        observed_upper=upper,
    )


def evaluate_d1_clauses(
    *,
    policy: D1ClausePolicy,
    decision_contract: D1DecisionContract,
    factorial: D1FactorialAnalysis | None,
    cells: tuple[D1AnalysisCell, ...],
) -> D1ClauseAnalysis:
    """Evaluate every condition and retain all co-equal clause riders."""
    if type(policy) is not D1ClausePolicy or type(decision_contract) is not D1DecisionContract:
        raise TypeError("D1 clause evaluation requires typed policy and decision contract")
    observed = {(cell.arm, cell.seed) for cell in cells}
    if len(observed) != len(cells):
        _fail("cell_population", "cells", "contains duplicate arm-by-seed identities")
    if factorial is None:
        if cells:
            _fail("cell_population", "cells", "finite cells require factorial analysis")
    else:
        expected = {(arm, seed) for seed in factorial.included_seeds for arm in D1Arm}
        if observed != expected:
            _fail("cell_population", "cells", "does not equal the factorial survivor grid")
        supplied = {(cell.arm, cell.seed): cell for cell in cells}
        if supplied != dict(factorial.cell_commitment):
            _fail("cell_population", "cells", "does not equal the factorial cell commitment")
    by_arm: dict[D1Arm, Mapping[str, D1ArmClauseResult]] = {}
    for arm in policy.candidate_arms:
        clauses: dict[str, D1ArmClauseResult] = {}
        for clause_id in decision_contract.clause_precedence:
            results = tuple(
                _evaluate_condition(condition, arm, factorial, cells)
                for condition in policy.conditions[clause_id][arm]
            )
            statuses = {item.status for item in results}
            status = (
                D1ConditionStatus.FAIL
                if D1ConditionStatus.FAIL in statuses
                else D1ConditionStatus.PASS
                if statuses == {D1ConditionStatus.PASS}
                else D1ConditionStatus.INCONCLUSIVE
            )
            clauses[clause_id] = D1ArmClauseResult(
                clause_id,
                decision_contract.clause_rules[clause_id].registered_rule,
                status,
                results,
            )
        by_arm[arm] = MappingProxyType(clauses)
    feasible = tuple(
        arm
        for arm in policy.candidate_arms
        if all(value.status is D1ConditionStatus.PASS for value in by_arm[arm].values())
    )
    return D1ClauseAnalysis(
        decision_contract.clause_precedence,
        MappingProxyType(by_arm),
        feasible,
    )
