"""Pre-outcome D1 decision contracts for v2 verdict validation.

The contract is supplied by the preregistration/build layer before a result is
known.  It binds registered sets and outcome rules without inventing numerical
criteria inside the verdict validator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from entropic_substrate.kernel.lifecycle import SuccessorAuthorization
from entropic_substrate.kernel.verdict.d1_parsing import freeze_json
from entropic_substrate.kernel.verdict.d1_types import (
    CLAUSE_STATUS,
    CONTRAST_KINDS,
    D1_ARM_IDS,
    D1ContrastTerm,
    D1FamilyIdentity,
    JsonValue,
)


@dataclass(frozen=True)
class D1DecisionRule:
    """The preregistered conditions for one allowed final decision."""

    successor_authorization: SuccessorAuthorization
    required_clause_statuses: Mapping[str, str]
    require_integrity_pass: bool
    require_review_clearance: bool
    eligible_strategies: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(status not in CLAUSE_STATUS for status in self.required_clause_statuses.values()):
            msg = "D1 decision rule clause statuses must use the registered clause vocabulary"
            raise ValueError(msg)
        if any(not name for name in self.required_clause_statuses):
            msg = "D1 decision rule clause identifiers must be nonempty strings"
            raise ValueError(msg)
        for name in (
            "require_integrity_pass",
            "require_review_clearance",
        ):
            if type(getattr(self, name)) is not bool:
                msg = f"D1 decision rule {name} must be bool"
                raise TypeError(msg)
        object.__setattr__(
            self,
            "required_clause_statuses",
            MappingProxyType(dict(self.required_clause_statuses)),
        )
        if any(not strategy for strategy in self.eligible_strategies):
            msg = "D1 decision rule eligible strategies must be nonempty identifiers"
            raise ValueError(msg)
        if len(set(self.eligible_strategies)) != len(self.eligible_strategies):
            msg = "D1 decision rule eligible strategies must be unique"
            raise ValueError(msg)


@dataclass(frozen=True)
class D1RegisteredEndpoint:
    """The exact metric estimator and smoothing rule used by one contrast."""

    metric: str
    estimator: str
    smoothing_window: int

    def __post_init__(self) -> None:
        if not self.metric or not self.estimator:
            raise ValueError("D1 registered endpoint fields must be nonempty")
        if type(self.smoothing_window) is not int or self.smoothing_window <= 0:
            raise ValueError("D1 registered smoothing window must be a positive integer")


@dataclass(frozen=True)
class D1RegisteredContrast:
    """Complete pre-outcome definition of one reported contrast."""

    contrast_id: str
    contrast_kind: str
    terms: tuple[D1ContrastTerm, ...]
    endpoint: D1RegisteredEndpoint
    n_rule: str
    registered_n: int | None
    df_rule: str
    confidence_level: float
    multiplicity: str
    margins: Mapping[str, float]
    decision_vocabulary: str

    def __post_init__(self) -> None:
        if not self.contrast_id or self.contrast_kind not in CONTRAST_KINDS:
            raise ValueError("D1 registered contrast identity or kind is invalid")
        if not self.terms or len({term.arm_id for term in self.terms}) != len(self.terms):
            raise ValueError("D1 registered contrast terms must be nonempty and unique")
        if any(term.arm_id not in D1_ARM_IDS or term.coefficient == 0.0 for term in self.terms):
            raise ValueError("D1 registered contrast terms require canonical arms and signs")
        if type(self.endpoint) is not D1RegisteredEndpoint:
            raise TypeError("D1 registered contrast requires a registered endpoint")
        if self.n_rule not in {"EXACT", "NONEXCLUDED_SEED_COUNT"}:
            raise ValueError("D1 registered contrast has an unknown sample-size rule")
        if self.n_rule == "EXACT":
            if type(self.registered_n) is not int or self.registered_n <= 0:
                raise ValueError("D1 exact sample-size rule requires a positive registered n")
        elif self.registered_n is not None:
            raise ValueError("D1 nonexcluded sample-size rule must not predeclare n")
        if self.df_rule not in {"N_MINUS_ONE", "NOT_APPLICABLE"}:
            raise ValueError("D1 registered contrast has an unknown degrees-of-freedom rule")
        if self.df_rule == "NOT_APPLICABLE" and self.registered_n != 1:
            raise ValueError("D1 NOT_APPLICABLE df requires exact n=1")
        if not math.isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ValueError("D1 registered confidence level must be finite and in (0, 1)")
        if not self.multiplicity or not self.decision_vocabulary:
            raise ValueError("D1 registered multiplicity and vocabulary must be nonempty")
        frozen_margins: dict[str, float] = {}
        for name, margin in self.margins.items():
            if not name or isinstance(margin, bool) or type(margin) not in (int, float):
                raise ValueError("D1 registered margins require named numeric values")
            if not math.isfinite(float(margin)) or float(margin) < 0.0:
                raise ValueError("D1 registered margins must be finite and nonnegative")
            frozen_margins[name] = float(margin)
        object.__setattr__(self, "margins", MappingProxyType(frozen_margins))


@dataclass(frozen=True)
class D1RegisteredClause:
    clause_id: str
    registered_rule: str
    decision_vocabulary: str
    threshold: float | None
    margin: float | None

    def __post_init__(self) -> None:
        if not self.clause_id or not self.registered_rule or not self.decision_vocabulary:
            raise ValueError("D1 registered clause identity and vocabulary must be nonempty")
        for name in ("threshold", "margin"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or type(value) not in (int, float)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"D1 registered clause {name} must be finite or null")


@dataclass(frozen=True)
class D1RegisteredExclusion:
    reason_id: str
    measurement: str
    predicate: str
    scope: str

    def __post_init__(self) -> None:
        if any(
            not value for value in (self.reason_id, self.measurement, self.predicate, self.scope)
        ):
            raise ValueError("D1 registered exclusion fields must be nonempty")


@dataclass(frozen=True)
class D1DecisionContract:
    """Registered D1 decision surface supplied before outcomes exist."""

    family_identity: D1FamilyIdentity
    registered_constants: Mapping[str, object]
    arm_ids: tuple[str, ...]
    contrast_ids: tuple[str, ...]
    clause_ids: tuple[str, ...]
    contrast_rules: Mapping[str, D1RegisteredContrast]
    clause_rules: Mapping[str, D1RegisteredClause]
    exclusion_rules: tuple[D1RegisteredExclusion, ...]
    clause_precedence: tuple[str, ...]
    verdict_vocabulary: str
    decision_rules: Mapping[str, D1DecisionRule]

    def __post_init__(self) -> None:
        if type(self.family_identity) is not D1FamilyIdentity:
            msg = "D1 decision contract requires a D1FamilyIdentity"
            raise TypeError(msg)
        if self.arm_ids != D1_ARM_IDS:
            msg = f"D1 decision contract requires exact arm order {D1_ARM_IDS}"
            raise ValueError(msg)
        for label, identifiers in (
            ("contrast", self.contrast_ids),
            ("clause", self.clause_ids),
        ):
            if not identifiers or any(not value for value in identifiers):
                msg = (
                    f"D1 decision contract {label} identifiers must be a nonempty tuple of strings"
                )
                raise ValueError(msg)
            if len(set(identifiers)) != len(identifiers):
                msg = f"D1 decision contract {label} identifiers must be unique"
                raise ValueError(msg)
        if set(self.contrast_rules) != set(self.contrast_ids):
            raise ValueError("D1 decision contract must bind every contrast rule exactly once")
        if set(self.clause_rules) != set(self.clause_ids):
            raise ValueError("D1 decision contract must bind every clause rule exactly once")
        if self.clause_precedence != self.clause_ids:
            raise ValueError("D1 clause precedence must list every clause in registered order")
        if not self.verdict_vocabulary:
            raise ValueError("D1 verdict vocabulary must be nonempty")
        frozen_contrasts: dict[str, D1RegisteredContrast] = {}
        for contrast_id, rule in self.contrast_rules.items():
            if type(rule) is not D1RegisteredContrast or rule.contrast_id != contrast_id:
                raise TypeError("D1 contrast rule keys must match typed contrast identities")
            frozen_contrasts[contrast_id] = rule
        frozen_clauses: dict[str, D1RegisteredClause] = {}
        for clause_id, rule in self.clause_rules.items():
            if type(rule) is not D1RegisteredClause or rule.clause_id != clause_id:
                raise TypeError("D1 clause rule keys must match typed clause identities")
            frozen_clauses[clause_id] = rule
        reasons = tuple(rule.reason_id for rule in self.exclusion_rules)
        if len(set(reasons)) != len(reasons) or any(
            type(rule) is not D1RegisteredExclusion for rule in self.exclusion_rules
        ):
            raise ValueError("D1 exclusion rules must have unique typed reason identifiers")
        frozen_constants = freeze_json(self.registered_constants, path="registered_constants")
        if not isinstance(frozen_constants, Mapping) or not frozen_constants:
            msg = "D1 decision contract registered_constants must be a nonempty object"
            raise ValueError(msg)
        if not self.decision_rules:
            msg = "D1 decision contract requires at least one decision rule"
            raise ValueError(msg)
        frozen_rules: dict[str, D1DecisionRule] = {}
        for verdict, rule in self.decision_rules.items():
            if not verdict:
                msg = "D1 decision contract verdict labels must be nonempty strings"
                raise ValueError(msg)
            if type(rule) is not D1DecisionRule:
                msg = "D1 decision contract rules must be D1DecisionRule values"
                raise TypeError(msg)
            if set(rule.required_clause_statuses) != set(self.clause_ids):
                msg = "D1 decision rule must bind every registered clause exactly once"
                raise ValueError(msg)
            frozen_rules[verdict] = rule
        object.__setattr__(self, "registered_constants", frozen_constants)
        object.__setattr__(self, "contrast_rules", MappingProxyType(frozen_contrasts))
        object.__setattr__(self, "clause_rules", MappingProxyType(frozen_clauses))
        object.__setattr__(self, "decision_rules", MappingProxyType(frozen_rules))

    def rule_for(self, verdict: str) -> D1DecisionRule:
        try:
            return self.decision_rules[verdict]
        except KeyError as exc:
            msg = f"D1 decision contract has no registered rule for verdict {verdict!r}"
            raise ValueError(msg) from exc


def constants_match(registered: Mapping[str, JsonValue], contract: D1DecisionContract) -> bool:
    """Compare fully frozen JSON values without coercion or inferred defaults."""

    return registered == contract.registered_constants
