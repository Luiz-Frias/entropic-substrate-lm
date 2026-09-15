"""The D1 decision contract binds the complete preregistered design."""

from __future__ import annotations

import json

import pytest

from d1_verdict_v2_fixtures import decision_contract, machine_verdict
from entropic_substrate.evidence.d1_types import D1VerdictValidationError
from entropic_substrate.evidence.d1_validator import ValidatedD1Verdict

pytestmark = pytest.mark.unit


def _validate(payload: dict[str, object]) -> None:
    ValidatedD1Verdict.from_mapping(payload, decision_contract=decision_contract())


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("contrasts", 0, "terms", 0, "coefficient"), -1.0, "terms"),
        (("contrasts", 0, "endpoint", "metric"), "loss", "endpoint"),
        (("contrasts", 0, "endpoint", "smoothing_window"), 1, "smoothing"),
        (("contrasts", 0, "n"), 3, "sample size"),
        (("contrasts", 0, "df"), 2, "degrees of freedom"),
        (("contrasts", 0, "confidence_level"), 0.95, "confidence"),
        (("contrasts", 0, "multiplicity"), "NONE", "multiplicity"),
        (("contrasts", 0, "margins", "practical"), 0.5, "margins"),
        (("clauses_by_arm", "sg", "language", "margin"), 0.5, "clause"),
        (
            ("clauses_by_arm", "sg", "language", "decision_vocabulary"),
            "other",
            "clause",
        ),
        (("clause_precedence", 0), "language", "precedence"),
        ("verdict_vocabulary", "other", "verdict vocabulary"),
        (("successor", "eligible_arms", 0), "both", "eligible_arms"),
        (("successor", "eligible_strategies", 0), "unregistered", "strategy"),
    ],
)
def test_machine_verdict_cannot_drift_from_registered_design(
    path: str | tuple[str | int, ...], value: object, message: str
) -> None:
    payload = json.loads(json.dumps(machine_verdict()))
    parts = (path,) if isinstance(path, str) else path
    target: object = payload
    for part in parts[:-1]:
        target = target[part]  # type: ignore[index]
    target[parts[-1]] = value  # type: ignore[index]

    with pytest.raises(D1VerdictValidationError, match=message):
        _validate(payload)


def test_registered_n_minus_one_rule_is_load_bearing() -> None:
    payload = machine_verdict()
    payload["contrasts"][0]["df"] = 4  # type: ignore[index]

    with pytest.raises(D1VerdictValidationError, match="degrees of freedom"):
        _validate(payload)


def test_unregistered_exclusion_reason_is_refused() -> None:
    payload = machine_verdict()
    payload["exclusions"] = [
        {
            "subject": "seed:4",
            "reason": "analyst-discretion",
            "evidence_hash": {"algorithm": "sha256", "digest": "e" * 64},
        }
    ]

    with pytest.raises(D1VerdictValidationError, match="exclusion"):
        _validate(payload)
