"""Registered report-only D1 trajectories and measured work-to-quality views."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from types import MappingProxyType
from typing import NoReturn, cast

from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.kernel.verdict.d1_contract import D1DecisionContract
from entropic_substrate.ladder.analyze.d1_capture import D1AdmittedAnalysisCapture
from entropic_substrate.ladder.register.d1_registered_design import decision_contract_digest

__all__ = [
    "D1DiagnosticPolicy",
    "D1DiagnosticPolicyError",
    "D1QualityTargetResult",
    "D1ReportRunDiagnostic",
    "D1TargetOperator",
    "D1TrajectoryPoint",
    "build_d1_report_diagnostics",
    "parse_d1_diagnostic_policy",
]

_SCHEMA = "d1-diagnostic-policy-v1"
_EVIDENCE_CLASS = "DERIVED_FROM_MEASURED"


class D1DiagnosticPolicyError(ValueError):
    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = MappingProxyType({"code": code, "path": path})
        super().__init__(f"D1 diagnostic policy {code} at {path}: {detail}")


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise D1DiagnosticPolicyError(code=code, path=path, detail=detail)


class D1TargetOperator(StrEnum):
    AT_MOST = "AT_MOST"
    AT_LEAST = "AT_LEAST"


@dataclass(frozen=True, slots=True)
class D1DiagnosticPolicy:
    aulc_metric: str
    aulc_normalization: str
    target_metric: str
    target_operator: D1TargetOperator
    target_threshold: float


@dataclass(frozen=True, slots=True)
class D1TrajectoryPoint:
    step: int
    eval_index: int
    value: float


@dataclass(frozen=True, slots=True)
class D1QualityTargetResult:
    reached: bool
    step: int | None
    eval_index: int | None
    metric_value: float | None
    trainer_wall_seconds: float | None
    source_bytes: int | None
    model_tokens: int | None
    evidence_class: str = _EVIDENCE_CLASS


@dataclass(frozen=True, slots=True)
class D1ReportRunDiagnostic:
    arm: D1Arm
    seed: int
    aulc_metric: str
    aulc: float
    aulc_normalization: str
    aulc_deciding: bool
    trajectory: tuple[D1TrajectoryPoint, ...]
    quality_target: D1QualityTargetResult
    evidence_class: str = _EVIDENCE_CLASS


def _object(value: object, *, path: str, fields: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        _fail("type", path, "must be an object with string keys")
    typed = cast(Mapping[str, object], value)
    if frozenset(typed) != fields:
        _fail("fields", path, f"must contain exactly {sorted(fields)}")
    return typed


def _metric(value: object, *, path: str) -> str:
    if type(value) is not str or not value:
        _fail("metric", path, "must be a nonempty string")
    return value


def parse_d1_diagnostic_policy(contract: D1DecisionContract) -> D1DiagnosticPolicy:
    """Read the exact diagnostic choices already frozen in registered constants."""
    if type(contract) is not D1DecisionContract:
        raise TypeError("D1 diagnostic policy requires D1DecisionContract")
    root = _object(
        contract.registered_constants.get("diagnostic_policy"),
        path="registered_constants.diagnostic_policy",
        fields=frozenset({"aulc", "quality_target", "schema_version"}),
    )
    if root["schema_version"] != _SCHEMA:
        _fail("schema", "diagnostic_policy.schema_version", f"must equal {_SCHEMA}")
    aulc = _object(
        root["aulc"],
        path="diagnostic_policy.aulc",
        fields=frozenset({"metric", "normalization"}),
    )
    if aulc["normalization"] != "RAW_TRAPEZOID":
        _fail("normalization", "diagnostic_policy.aulc.normalization", "is unsupported")
    target = _object(
        root["quality_target"],
        path="diagnostic_policy.quality_target",
        fields=frozenset({"metric", "operator", "threshold"}),
    )
    try:
        operator = D1TargetOperator(target["operator"])
    except (TypeError, ValueError) as error:
        _fail("operator", "diagnostic_policy.quality_target.operator", str(error))
    threshold = target["threshold"]
    if isinstance(threshold, bool) or type(threshold) not in (int, float):
        _fail("threshold", "diagnostic_policy.quality_target.threshold", "must be numeric")
    threshold_float = float(cast(int | float, threshold))
    if not math.isfinite(threshold_float):
        _fail("threshold", "diagnostic_policy.quality_target.threshold", "must be finite")
    return D1DiagnosticPolicy(
        aulc_metric=_metric(aulc["metric"], path="diagnostic_policy.aulc.metric"),
        aulc_normalization="RAW_TRAPEZOID",
        target_metric=_metric(target["metric"], path="diagnostic_policy.quality_target.metric"),
        target_operator=operator,
        target_threshold=threshold_float,
    )


def _reached(policy: D1DiagnosticPolicy, value: float) -> bool:
    if policy.target_operator is D1TargetOperator.AT_MOST:
        return value <= policy.target_threshold
    return value >= policy.target_threshold


def _identity(run: object) -> tuple[D1Arm, int]:
    from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1RunEvidence

    if type(run) is not D1RunEvidence:
        raise TypeError("D1 report diagnostic requires D1RunEvidence")
    plan = cast(Mapping[str, object], run.plan_manifest)
    try:
        arm = D1Arm(plan["arm"])
    except (KeyError, TypeError, ValueError) as error:
        _fail("identity", "run.plan.arm", str(error))
    seed = plan.get("seed")
    if type(seed) is not int or seed < 0:
        _fail("identity", "run.plan.seed", "must be a nonnegative integer")
    return arm, seed


def _target(run: object, policy: D1DiagnosticPolicy) -> D1QualityTargetResult:
    from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1RunEvidence

    if type(run) is not D1RunEvidence:
        raise TypeError("D1 report diagnostic requires D1RunEvidence")
    for item in run.evals:
        value = item.quality.scalar_metric(policy.target_metric)
        if not _reached(policy, value):
            continue
        work = item.resource_work
        return D1QualityTargetResult(
            True,
            item.step,
            item.eval_index,
            value,
            work.trainer_wall_seconds,
            work.committed_quality_source_bytes,
            work.committed_quality_model_tokens,
        )
    return D1QualityTargetResult(False, None, None, None, None, None, None)


def build_d1_report_diagnostics(
    *,
    admitted: D1AdmittedAnalysisCapture,
    policy: D1DiagnosticPolicy,
    decision_contract: D1DecisionContract,
) -> tuple[D1ReportRunDiagnostic, ...]:
    """Compute report-only probes; none participates in clause evaluation."""
    if (
        type(admitted) is not D1AdmittedAnalysisCapture
        or type(policy) is not D1DiagnosticPolicy
        or type(decision_contract) is not D1DecisionContract
    ):
        raise TypeError("D1 diagnostics require admitted capture, policy, and contract")
    if admitted.decision_contract_digest != decision_contract_digest(decision_contract):
        _fail("identity", "decision_contract", "differs from capture admission")
    results: list[D1ReportRunDiagnostic] = []
    for run in admitted.runs:
        arm, seed = _identity(run)
        trajectory = tuple(
            D1TrajectoryPoint(
                item.step,
                item.eval_index,
                item.quality.scalar_metric(policy.aulc_metric),
            )
            for item in run.evals
        )
        if not trajectory:
            _fail("trajectory", f"{arm.value}:{seed}", "has no evaluation")
        aulc = math.fsum(
            (right.step - left.step) * (left.value + right.value) / 2.0
            for left, right in pairwise(trajectory)
        )
        results.append(
            D1ReportRunDiagnostic(
                arm=arm,
                seed=seed,
                aulc_metric=policy.aulc_metric,
                aulc=aulc,
                aulc_normalization=policy.aulc_normalization,
                aulc_deciding=False,
                trajectory=trajectory,
                quality_target=_target(run, policy),
            )
        )
    return tuple(results)
