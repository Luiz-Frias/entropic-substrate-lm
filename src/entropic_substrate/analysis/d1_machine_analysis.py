"""Construction-blind machine serialization for already-computed D1 evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from entropic_substrate.analysis.d1_capture import D1AdmittedAnalysisCapture
from entropic_substrate.analysis.d1_clause_policy import D1ClauseAnalysis
from entropic_substrate.analysis.d1_diagnostics import D1ReportRunDiagnostic
from entropic_substrate.analysis.d1_factorial import D1FactorialAnalysis
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.experiments.d1_primitives import D1Arm, require_sha256
from entropic_substrate.experiments.d1_registered_design import (
    D1RegisteredDesign,
    decision_contract_digest,
)

__all__ = [
    "D1AliasError",
    "D1AliasedMachineAnalysis",
    "D1ArmAliasMap",
    "build_d1_aliased_machine_analysis",
]

_SCHEMA = "d1-aliased-machine-analysis-v1"


class D1AliasError(ValueError):
    """The sealed alias map or its application is incomplete or revealing."""


@dataclass(frozen=True, slots=True)
class D1ArmAliasMap:
    aliases: Mapping[D1Arm, str]
    sealed_commitment: str

    def __post_init__(self) -> None:
        if set(self.aliases) != set(D1Arm):
            raise D1AliasError("alias map must bind every D1 arm exactly once")
        frozen: dict[D1Arm, str] = {}
        for arm in D1Arm:
            alias = self.aliases[arm]
            if type(alias) is not str or not alias or alias.strip() != alias:
                raise D1AliasError("alias labels must be nonempty trimmed strings")
            if alias.casefold() in {value.value.casefold() for value in D1Arm}:
                raise D1AliasError("alias labels must be opaque, not true arm identifiers")
            frozen[arm] = alias
        if len(set(frozen.values())) != len(D1Arm):
            raise D1AliasError("alias labels must be unique")
        try:
            require_sha256(self.sealed_commitment, label="D1 alias sealed commitment")
        except ValueError as error:
            raise D1AliasError("alias map requires a SHA-256 sealed commitment") from error
        object.__setattr__(self, "aliases", MappingProxyType(frozen))

    def for_arm(self, arm: D1Arm) -> str:
        return self.aliases[arm]


@dataclass(frozen=True, slots=True)
class D1AliasedMachineAnalysis:
    canonical_json: str

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> D1AliasedMachineAnalysis:
        return cls(
            json.dumps(
                manifest,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    @property
    def manifest(self) -> dict[str, object]:
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):  # pragma: no cover - factory always writes an object
            raise AssertionError("aliased machine analysis lost its object root")
        return value

    @property
    def analysis_digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _contrast_manifest(
    factorial: D1FactorialAnalysis,
    contract: D1DecisionContract,
    aliases: D1ArmAliasMap,
    contrast_aliases: Mapping[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for contrast_id in contract.contrast_ids:
        rule = contract.contrast_rules[contrast_id]
        result = factorial.registered_contrasts[contrast_id]
        rows.append(
            {
                "confidence_level": result.confidence_level,
                "contrast_alias": contrast_aliases[contrast_id],
                "df": result.df,
                "estimate": result.estimate,
                "evidence_class": result.evidence_class,
                "interval_lower": result.interval_lower,
                "interval_upper": result.interval_upper,
                "margins": dict(rule.margins),
                "multiplicity": result.multiplicity,
                "n": result.n,
                "paired_estimates": list(result.paired_estimates),
                "sample_sd": result.sample_sd,
                "standard_error": result.standard_error,
                "t_critical": result.t_critical,
                "terms": [
                    {
                        "arm_alias": aliases.for_arm(D1Arm(term.arm_id)),
                        "coefficient": term.coefficient,
                    }
                    for term in rule.terms
                ],
            }
        )
    return rows


def _effect_manifest(
    factorial: D1FactorialAnalysis,
    aliases: D1ArmAliasMap,
) -> list[dict[str, object]]:
    return [
        {
            "deciding": effect.deciding,
            "effect_alias": f"F{index:02d}",
            "estimate": effect.estimate,
            "evidence_class": effect.evidence_class,
            "n": effect.n,
            "paired_estimates": list(effect.paired_estimates),
            "terms": [
                {"arm_alias": aliases.for_arm(arm), "coefficient": coefficient}
                for arm, coefficient in effect.terms
            ],
        }
        for index, effect in enumerate(factorial.factorial_effects.values(), start=1)
    ]


def _condition_manifest(
    value: object,
    contrast_aliases: Mapping[str, str],
) -> dict[str, object]:
    from entropic_substrate.analysis.d1_clause_policy import D1ConditionResult

    if type(value) is not D1ConditionResult:
        raise TypeError("aliased clause serialization requires D1ConditionResult")
    condition = value.condition
    return {
        "contrast_alias": (
            None if condition.contrast_id is None else contrast_aliases[condition.contrast_id]
        ),
        "estimator": condition.estimator,
        "evidence_class": value.evidence_class,
        "metric": condition.metric,
        "observed_lower": value.observed_lower,
        "observed_point": value.observed_point,
        "observed_upper": value.observed_upper,
        "operator": condition.operator.value,
        "smoothing_window": condition.smoothing_window,
        "source": condition.source.value,
        "status": value.status.value,
        "threshold": condition.threshold,
    }


def _clauses_manifest(
    clauses: D1ClauseAnalysis,
    aliases: D1ArmAliasMap,
    contrast_aliases: Mapping[str, str],
) -> dict[str, object]:
    return {
        aliases.for_arm(arm): {
            clause_id: {
                "conditions": [
                    _condition_manifest(item, contrast_aliases) for item in result.conditions
                ],
                "evidence_class": result.evidence_class,
                "status": result.status.value,
            }
            for clause_id, result in by_clause.items()
        }
        for arm, by_clause in clauses.by_arm.items()
    }


def _diagnostic_manifest(
    value: D1ReportRunDiagnostic,
    aliases: D1ArmAliasMap,
) -> dict[str, object]:
    target = value.quality_target
    return {
        "arm_alias": aliases.for_arm(value.arm),
        "aulc": value.aulc,
        "aulc_deciding": value.aulc_deciding,
        "aulc_metric": value.aulc_metric,
        "aulc_normalization": value.aulc_normalization,
        "evidence_class": value.evidence_class,
        "quality_target": {
            "eval_index": target.eval_index,
            "evidence_class": target.evidence_class,
            "metric_value": target.metric_value,
            "model_tokens": target.model_tokens,
            "reached": target.reached,
            "source_bytes": target.source_bytes,
            "step": target.step,
            "trainer_wall_seconds": target.trainer_wall_seconds,
        },
        "seed": value.seed,
        "trajectory": [
            {"eval_index": item.eval_index, "step": item.step, "value": item.value}
            for item in value.trajectory
        ],
    }


def _validate_inputs(
    design: D1RegisteredDesign,
    contract: D1DecisionContract,
    admitted: D1AdmittedAnalysisCapture,
    factorial: D1FactorialAnalysis | None,
    clauses: D1ClauseAnalysis,
    diagnostics: tuple[D1ReportRunDiagnostic, ...],
) -> None:
    if admitted.design_digest != design.design_digest:
        raise D1AliasError("analysis capture differs from registered design")
    digest = decision_contract_digest(contract)
    if admitted.decision_contract_digest != digest or digest != design.decision_contract_digest:
        raise D1AliasError("analysis capture differs from decision contract")
    cell_ids = {(item.arm, item.seed) for item in admitted.cells}
    diagnostic_ids = {(item.arm, item.seed) for item in diagnostics}
    if len(diagnostic_ids) != len(diagnostics) or diagnostic_ids != cell_ids:
        raise D1AliasError("diagnostics must equal retained finite analysis cells")
    if factorial is None:
        if admitted.cells:
            raise D1AliasError("finite cells require factorial analysis")
    elif set(factorial.included_seeds) != {item.seed for item in admitted.cells}:
        raise D1AliasError("factorial seed identities differ from admitted cells")
    if tuple(clauses.by_arm) == ():
        raise D1AliasError("clause analysis must retain registered candidates")


def build_d1_aliased_machine_analysis(
    *,
    design: D1RegisteredDesign,
    decision_contract: D1DecisionContract,
    admitted: D1AdmittedAnalysisCapture,
    factorial: D1FactorialAnalysis | None,
    clauses: D1ClauseAnalysis,
    diagnostics: tuple[D1ReportRunDiagnostic, ...],
    aliases: D1ArmAliasMap,
) -> D1AliasedMachineAnalysis:
    """Serialize evidence using aliases only; no narrative verdict or gate."""
    if (
        type(design) is not D1RegisteredDesign
        or type(decision_contract) is not D1DecisionContract
        or type(admitted) is not D1AdmittedAnalysisCapture
        or type(clauses) is not D1ClauseAnalysis
        or type(diagnostics) is not tuple
        or type(aliases) is not D1ArmAliasMap
    ):
        raise TypeError("aliased D1 analysis requires nominal evidence values")
    _validate_inputs(design, decision_contract, admitted, factorial, clauses, diagnostics)
    contrast_aliases = {
        contrast_id: f"C{index:02d}"
        for index, contrast_id in enumerate(decision_contract.contrast_ids, start=1)
    }
    endpoint = None
    contrasts: list[dict[str, object]] = []
    effects: list[dict[str, object]] = []
    included: list[int] = []
    excluded = list(design.registered_seeds)
    if factorial is not None:
        endpoint = {
            "estimator": factorial.endpoint.estimator,
            "evidence_class": factorial.endpoint.evidence_class,
            "metric": factorial.endpoint.metric,
            "smoothing_window": factorial.endpoint.smoothing_window,
            "values_by_alias": {
                aliases.for_arm(arm): list(values)
                for arm, values in factorial.endpoint.values_by_arm.items()
            },
        }
        contrasts = _contrast_manifest(factorial, decision_contract, aliases, contrast_aliases)
        effects = _effect_manifest(factorial, aliases)
        included = list(factorial.included_seeds)
        excluded = list(factorial.excluded_seeds)
    manifest: dict[str, object] = {
        "alias_commitment": aliases.sealed_commitment,
        "bindings": {
            "capture_digest": admitted.capture_digest,
            "decision_contract_digest": admitted.decision_contract_digest,
            "design_digest": admitted.design_digest,
        },
        "clauses_by_alias": _clauses_manifest(clauses, aliases, contrast_aliases),
        "clause_precedence": list(clauses.clause_precedence),
        "contrasts": contrasts,
        "diagnostics": [_diagnostic_manifest(item, aliases) for item in diagnostics],
        "endpoint": endpoint,
        "excluded_seeds": excluded,
        "exclusions": [
            {
                "evidence_digest": item.evidence_digest,
                "exclusion_alias": f"X{index:02d}",
                "reason_id": item.reason_id,
            }
            for index, item in enumerate(admitted.exclusions, start=1)
        ],
        "factorial_effects": effects,
        "feasible_aliases": [aliases.for_arm(arm) for arm in clauses.feasible_arms],
        "included_seeds": included,
        "schema_version": _SCHEMA,
    }
    return D1AliasedMachineAnalysis.from_manifest(manifest)
