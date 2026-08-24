"""Bind structurally valid D1 machine output to registered control values."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NoReturn, cast

from entropic_substrate.analysis.d1_clause_policy import (
    D1ClausePolicy,
    D1RegisteredCondition,
    parse_d1_clause_policy,
)
from entropic_substrate.analysis.d1_external_analysis_control import (
    D1ExternalAnalysisInputControl,
)
from entropic_substrate.analysis.d1_factorial import d1_factorial_effect_terms
from entropic_substrate.analysis.d1_machine_analysis import D1AliasedMachineAnalysis
from entropic_substrate.analysis.d1_machine_analysis_codec import (
    decode_d1_machine_analysis_declaration,
)
from entropic_substrate.evidence.d1_contract import (
    D1DecisionContract,
    D1RegisteredContrast,
    D1RegisteredEndpoint,
)
from entropic_substrate.evidence.d1_types import D1ContrastTerm
from entropic_substrate.experiments.d1_primitives import D1Arm
from entropic_substrate.experiments.d1_registered_design import decision_contract_digest

__all__ = [
    "D1ContractBoundMachineAnalysis",
    "D1MachineAnalysisBindingError",
    "bind_d1_machine_analysis_contract",
]

_TOKEN = object()


class D1MachineAnalysisBindingError(ValueError):
    """A structurally valid machine analysis differs from registered controls."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1MachineAnalysisBindingError(message)
    raise D1MachineAnalysisBindingError(message) from error


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail(f"machine analysis {label} must be an object")
    return cast(dict[str, object], value)


def _rows(value: object, label: str) -> list[object]:
    if type(value) is not list:
        _fail(f"machine analysis {label} must be an array")
    return cast(list[object], value)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _require_exact(actual: object, expected: object, label: str) -> None:
    if _canonical(actual) != _canonical(expected):
        _fail(f"machine analysis {label} differs from registered control")


def _alias(expected: D1ExternalAnalysisInputControl, arm: D1Arm) -> str:
    return expected.alias_application.aliases.for_arm(arm)


def _bind_identities(root: dict[str, object], expected: D1ExternalAnalysisInputControl) -> None:
    bindings = _mapping(root["bindings"], "bindings")
    required = {
        "capture_digest": expected.aggregate.manifest_digest,
        "decision_contract_digest": decision_contract_digest(expected.decision_contract),
        "design_digest": expected.design.design_digest,
    }
    if bindings != required:
        _fail("machine analysis top binding differs from verified input")
    if root["alias_commitment"] != expected.alias_application.commitment_sha256:
        _fail("machine analysis alias commitment differs from verified input")


def _bind_seed_partition(
    root: dict[str, object], expected: D1ExternalAnalysisInputControl
) -> tuple[int, ...]:
    included = tuple(cast(list[int], root["included_seeds"]))
    excluded = tuple(cast(list[int], root["excluded_seeds"]))
    registered = expected.design.registered_seeds
    if (
        set(included) | set(excluded) != set(registered)
        or tuple(seed for seed in registered if seed in set(included)) != included
        or tuple(seed for seed in registered if seed in set(excluded)) != excluded
    ):
        _fail("machine analysis seed partition differs from registered design")
    return included


def _shared_endpoint(contract: D1DecisionContract) -> D1RegisteredEndpoint:
    endpoints = tuple(contract.contrast_rules[name].endpoint for name in contract.contrast_ids)
    endpoint = endpoints[0]
    if any(value != endpoint for value in endpoints[1:]):
        _fail("registered contrast endpoints do not define one factorial endpoint")
    return endpoint


def _bind_endpoint(
    root: dict[str, object], contract: D1DecisionContract, included: tuple[int, ...]
) -> None:
    if not included:
        return
    endpoint = _mapping(root["endpoint"], "endpoint")
    registered = _shared_endpoint(contract)
    _require_exact(endpoint["metric"], registered.metric, "endpoint metric")
    _require_exact(endpoint["estimator"], registered.estimator, "endpoint estimator")
    _require_exact(
        endpoint["smoothing_window"],
        registered.smoothing_window,
        "endpoint smoothing_window",
    )


def _expected_n(rule: D1RegisteredContrast, included: tuple[int, ...]) -> int:
    n = len(included)
    if rule.n_rule == "EXACT" and rule.registered_n != n:
        _fail("machine analysis contrast n differs from exact registered n")
    return n


def _expected_df(rule: D1RegisteredContrast, n: int) -> int | None:
    if rule.df_rule == "NOT_APPLICABLE":
        return None
    if n < 2:
        _fail("machine analysis contrast n cannot satisfy registered df rule")
    return n - 1


def _term_rows(
    terms: tuple[D1ContrastTerm, ...],
    expected: D1ExternalAnalysisInputControl,
) -> list[dict[str, object]]:
    return [
        {
            "arm_alias": _alias(expected, D1Arm(term.arm_id)),
            "coefficient": term.coefficient,
        }
        for term in terms
    ]


def _bind_contrasts(
    root: dict[str, object],
    expected: D1ExternalAnalysisInputControl,
    included: tuple[int, ...],
) -> None:
    rows = _rows(root["contrasts"], "contrasts")
    contract = expected.decision_contract
    if len(rows) != (len(contract.contrast_ids) if included else 0):
        _fail("machine analysis contrast count differs from registered contract")
    for index, (raw, contrast_id) in enumerate(zip(rows, contract.contrast_ids, strict=True)):
        row = _mapping(raw, f"contrast[{index}]")
        rule = contract.contrast_rules[contrast_id]
        n = _expected_n(rule, included)
        required = {
            "confidence_level": rule.confidence_level,
            "contrast_alias": f"C{index + 1:02d}",
            "df": _expected_df(rule, n),
            "margins": dict(rule.margins),
            "multiplicity": rule.multiplicity,
            "n": n,
            "terms": _term_rows(rule.terms, expected),
        }
        for name, value in required.items():
            _require_exact(row[name], value, f"contrast[{index}].{name}")


def _bind_effects(
    root: dict[str, object],
    expected: D1ExternalAnalysisInputControl,
    included: tuple[int, ...],
) -> None:
    rows = _rows(root["factorial_effects"], "factorial_effects")
    effects = tuple(d1_factorial_effect_terms().values()) if included else ()
    if len(rows) != len(effects):
        _fail("machine analysis effect count differs from standard factorial effects")
    for index, (raw, terms) in enumerate(zip(rows, effects, strict=True)):
        row = _mapping(raw, f"effect[{index}]")
        expected_terms = [
            {"arm_alias": _alias(expected, arm), "coefficient": coefficient}
            for arm, coefficient in terms
        ]
        _require_exact(row["effect_alias"], f"F{index + 1:02d}", f"effect[{index}].alias")
        _require_exact(row["terms"], expected_terms, f"effect[{index}].terms")
        _require_exact(row["n"], len(included), f"effect[{index}].n")


def _expected_condition(
    condition: D1RegisteredCondition,
    contrast_aliases: Mapping[str, str],
) -> dict[str, object]:
    return {
        "contrast_alias": (
            None if condition.contrast_id is None else contrast_aliases[condition.contrast_id]
        ),
        "estimator": condition.estimator,
        "metric": condition.metric,
        "operator": condition.operator.value,
        "smoothing_window": condition.smoothing_window,
        "source": condition.source.value,
        "threshold": condition.threshold,
    }


def _bind_clauses(
    root: dict[str, object],
    expected: D1ExternalAnalysisInputControl,
    policy: D1ClausePolicy,
) -> None:
    contract = expected.decision_contract
    _require_exact(root["clause_precedence"], list(contract.clause_precedence), "clause precedence")
    by_alias = _mapping(root["clauses_by_alias"], "clauses_by_alias")
    candidate_aliases = tuple(_alias(expected, arm) for arm in policy.candidate_arms)
    if set(by_alias) != set(candidate_aliases):
        _fail("machine analysis clause candidates differ from registered policy")
    contrast_aliases = {
        name: f"C{index:02d}" for index, name in enumerate(contract.contrast_ids, start=1)
    }
    for arm in policy.candidate_arms:
        clauses = _mapping(by_alias[_alias(expected, arm)], "candidate clauses")
        for clause_id in contract.clause_precedence:
            result = _mapping(clauses[clause_id], f"clause {clause_id}")
            actual = _rows(result["conditions"], f"clause {clause_id} conditions")
            registered = [
                _expected_condition(condition, contrast_aliases)
                for condition in policy.conditions[clause_id][arm]
            ]
            if len(actual) != len(registered):
                _fail("machine analysis condition count differs from registered policy")
            for condition_index, (row, construction) in enumerate(
                zip(actual, registered, strict=True)
            ):
                values = _mapping(row, "condition")
                for name, value in construction.items():
                    _require_exact(
                        values[name],
                        value,
                        f"condition[{condition_index}].{name}",
                    )
    feasible = cast(list[str], root["feasible_aliases"])
    expected_order = [alias for alias in candidate_aliases if alias in set(feasible)]
    if feasible != expected_order:
        _fail("machine analysis feasible clause aliases differ from candidate order")


def _bind_contract(
    analysis: D1AliasedMachineAnalysis,
    expected: D1ExternalAnalysisInputControl,
) -> None:
    root = analysis.manifest
    _bind_identities(root, expected)
    included = _bind_seed_partition(root, expected)
    _bind_endpoint(root, expected.decision_contract, included)
    _bind_contrasts(root, expected, included)
    _bind_effects(root, expected, included)
    try:
        policy = parse_d1_clause_policy(expected.decision_contract)
    except (TypeError, ValueError) as error:
        _fail("verified input lacks a valid registered clause policy", error)
    _bind_clauses(root, expected, policy)


@dataclass(frozen=True, slots=True)
class D1ContractBoundMachineAnalysis:
    """Partial typestate: registered contract bound; capture diagnostics remain unbound."""

    canonical_bytes: bytes = field(repr=False)
    analysis: D1AliasedMachineAnalysis
    expected: D1ExternalAnalysisInputControl = field(repr=False)
    input_manifest_sha256: str
    _factory_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _TOKEN:
            raise TypeError(
                "D1ContractBoundMachineAnalysis is factory-only; "
                "use bind_d1_machine_analysis_contract"
            )
        if self.analysis.canonical_json.encode("utf-8") != self.canonical_bytes:
            raise ValueError("contract-bound analysis differs from canonical bytes")
        if self.expected.input_manifest_sha256 != self.input_manifest_sha256:
            raise ValueError("contract-bound analysis differs from verified input")
        _bind_contract(self.analysis, self.expected)

    @property
    def analysis_digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def bind_d1_machine_analysis_contract(
    payload: bytes,
    *,
    expected: D1ExternalAnalysisInputControl,
) -> D1ContractBoundMachineAnalysis:
    """Bind canonical machine output to registered identities without recomputing outcomes."""
    if type(expected) is not D1ExternalAnalysisInputControl:
        raise TypeError("machine analysis binding requires verified external input control")
    analysis = decode_d1_machine_analysis_declaration(payload)
    _bind_contract(analysis, expected)
    return D1ContractBoundMachineAnalysis(
        payload,
        analysis,
        expected,
        expected.input_manifest_sha256,
        _TOKEN,
    )
