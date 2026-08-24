"""Pre-outcome D1 decision contracts for v2 verdict validation.

The contract is supplied by the preregistration/build layer before a result is
known.  It binds registered sets and outcome rules without inventing numerical
criteria inside the verdict validator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from entropic_substrate.evidence.d1_parsing import freeze_json
from entropic_substrate.evidence.d1_types import (
    CLAUSE_STATUS,
    D1_ARM_IDS,
    D1FamilyIdentity,
    JsonValue,
)
from entropic_substrate.evidence.lifecycle import SuccessorAuthorization


@dataclass(frozen=True)
class D1DecisionRule:
    """The preregistered conditions for one allowed final decision."""

    successor_authorization: SuccessorAuthorization
    required_clause_statuses: Mapping[str, str]
    require_integrity_pass: bool
    require_review_clearance: bool
    require_eligible_strategies: bool

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
            "require_eligible_strategies",
        ):
            if type(getattr(self, name)) is not bool:
                msg = f"D1 decision rule {name} must be bool"
                raise TypeError(msg)
        object.__setattr__(
            self,
            "required_clause_statuses",
            MappingProxyType(dict(self.required_clause_statuses)),
        )


@dataclass(frozen=True)
class D1DecisionContract:
    """Registered D1 decision surface supplied before outcomes exist."""

    family_identity: D1FamilyIdentity
    registered_constants: Mapping[str, object]
    arm_ids: tuple[str, ...]
    contrast_ids: tuple[str, ...]
    clause_ids: tuple[str, ...]
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
