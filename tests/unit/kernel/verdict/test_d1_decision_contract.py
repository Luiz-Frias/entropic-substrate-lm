"""Pre-outcome D1 decision contracts are exact and immutable."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from entropic_substrate.kernel.lifecycle import SuccessorAuthorization
from entropic_substrate.kernel.verdict.d1_contract import (
    D1DecisionContract,
    D1DecisionRule,
    D1RegisteredClause,
    D1RegisteredContrast,
    D1RegisteredEndpoint,
    D1RegisteredExclusion,
)
from entropic_substrate.kernel.verdict.d1_types import D1_ARM_IDS, D1ContrastTerm, D1FamilyIdentity

pytestmark = pytest.mark.unit


def _identity() -> D1FamilyIdentity:
    return D1FamilyIdentity(
        family_digest="f" * 64,
        arm_construction_digests={arm: f"{index + 1:064x}" for index, arm in enumerate(D1_ARM_IDS)},
    )


def _rule(status: str = "PASS") -> D1DecisionRule:
    return D1DecisionRule(
        successor_authorization=SuccessorAuthorization.AUTHORIZED,
        required_clause_statuses={"noncollapse": status, "grounding": status, "language": status},
        require_integrity_pass=True,
        require_review_clearance=True,
        eligible_strategies=("d2",),
    )


def _design() -> dict[str, object]:
    endpoint = D1RegisteredEndpoint("nats_per_byte", "last_k_mean", 5)
    terms = {
        "sg": (D1ContrastTerm("sg", 1.0), D1ContrastTerm("none", -1.0)),
        "ac": (D1ContrastTerm("ac", 1.0), D1ContrastTerm("none", -1.0)),
        "interaction": tuple(
            D1ContrastTerm(arm, coefficient)
            for arm, coefficient in (("none", 1.0), ("sg", -1.0), ("ac", -1.0), ("both", 1.0))
        ),
    }
    contrast_rules = {
        name: D1RegisteredContrast(
            contrast_id=name,
            contrast_kind="FACTORIAL_INTERACTION" if name == "interaction" else "PAIRWISE",
            terms=contrast_terms,
            endpoint=endpoint,
            n_rule="EXACT",
            registered_n=4,
            df_rule="N_MINUS_ONE",
            confidence_level=0.975,
            multiplicity="BONFERRONI",
            margins={"practical": 0.01},
            decision_vocabulary="d1-contrast/v1",
        )
        for name, contrast_terms in terms.items()
    }
    clause_ids = ("noncollapse", "grounding", "language")
    return {
        "contrast_rules": contrast_rules,
        "clause_rules": {
            name: D1RegisteredClause(name, f"d1.{name}.v1", "d1-clause/v1", None, None)
            for name in clause_ids
        },
        "exclusion_rules": (
            D1RegisteredExclusion("nonfinite", "final-loss", "NONFINITE", "WHOLE_SEED_BLOCK"),
        ),
        "clause_precedence": clause_ids,
        "verdict_vocabulary": "d1-verdict/vocabulary-1",
    }


def _contract() -> D1DecisionContract:
    return D1DecisionContract(
        family_identity=_identity(),
        registered_constants={"seeds": [1, 2, 3], "future_loss_weight": 1.0},
        arm_ids=D1_ARM_IDS,
        contrast_ids=("sg", "ac", "interaction"),
        clause_ids=("noncollapse", "grounding", "language"),
        decision_rules={"ADVANCE": _rule()},
        **_design(),  # type: ignore[arg-type]
    )


def test_contract_freezes_constants_rules_and_exact_registered_sets() -> None:
    contract = _contract()

    assert isinstance(contract.registered_constants, MappingProxyType)
    assert isinstance(contract.decision_rules, MappingProxyType)
    assert contract.rule_for("ADVANCE").successor_authorization is (
        SuccessorAuthorization.AUTHORIZED
    )
    with pytest.raises(TypeError):
        contract.registered_constants["late"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="no registered rule"):
        contract.rule_for("OUTCOME_SHOPPED")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("arm_ids", ("none", "sg"), "arm order"),
        ("contrast_ids", ("sg", "sg"), "unique"),
        ("clause_ids", (), "nonempty"),
        ("registered_constants", {}, "nonempty"),
        ("decision_rules", {}, "at least one"),
    ],
)
def test_contract_refuses_incomplete_or_ambiguous_registered_surfaces(
    field: str, value: object, message: str
) -> None:
    payload = {
        "family_identity": _identity(),
        "registered_constants": {"future_loss_weight": 1.0},
        "arm_ids": D1_ARM_IDS,
        "contrast_ids": ("sg", "ac", "interaction"),
        "clause_ids": ("noncollapse", "grounding", "language"),
        "decision_rules": {"ADVANCE": _rule()},
        **_design(),
    }
    payload[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        D1DecisionContract(**payload)  # type: ignore[arg-type]


def test_every_rule_binds_every_clause_exactly_once() -> None:
    with pytest.raises(ValueError, match="every registered clause"):
        D1DecisionContract(
            family_identity=_identity(),
            registered_constants={"future_loss_weight": 1.0},
            arm_ids=D1_ARM_IDS,
            contrast_ids=("sg",),
            clause_ids=("noncollapse", "grounding", "language"),
            contrast_rules={"sg": _design()["contrast_rules"]["sg"]},  # type: ignore[index]
            clause_rules=_design()["clause_rules"],  # type: ignore[arg-type]
            exclusion_rules=_design()["exclusion_rules"],  # type: ignore[arg-type]
            clause_precedence=("noncollapse", "grounding", "language"),
            verdict_vocabulary="d1-verdict/vocabulary-1",
            decision_rules={
                "ADVANCE": D1DecisionRule(
                    successor_authorization=SuccessorAuthorization.AUTHORIZED,
                    required_clause_statuses={"noncollapse": "PASS"},
                    require_integrity_pass=True,
                    require_review_clearance=True,
                    eligible_strategies=("d2",),
                )
            },
        )
