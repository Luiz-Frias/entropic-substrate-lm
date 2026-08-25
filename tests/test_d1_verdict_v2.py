"""Decision-content validation for D1 verdict v2 without filing dependencies."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from d1_verdict_v2_fixtures import decision_contract, machine_verdict
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.evidence.d1_types import D1VerdictValidationError
from entropic_substrate.evidence.d1_validator import ValidatedD1Verdict

pytestmark = pytest.mark.unit


def _validated(payload: dict[str, object]) -> ValidatedD1Verdict:
    return ValidatedD1Verdict.from_mapping(payload, decision_contract=decision_contract())


def _one_seed_payload() -> tuple[D1DecisionContract, dict[str, object]]:
    contract = decision_contract()
    one_seed_contract = replace(
        contract,
        registered_constants={"seeds": [1], "primary": {"alpha": 0.025}},
        contrast_rules={
            name: replace(rule, registered_n=1, df_rule="NOT_APPLICABLE")
            for name, rule in contract.contrast_rules.items()
        },
    )
    payload = machine_verdict()
    payload["registered_constants"] = {
        "seeds": [1],
        "primary": {"alpha": 0.025},
    }
    for contrast in payload["contrasts"]:  # type: ignore[union-attr]
        contrast["n"] = 1
        contrast["df"] = None
        contrast["sample_sd"] = None
        contrast["interval_lower"] = None
        contrast["interval_upper"] = None
    return one_seed_contract, payload


def test_v2_preserves_full_precision_scientific_content_without_filing_hashes() -> None:
    verdict = _validated(machine_verdict())
    assert verdict.schema_version == "d1-verdict/v2"
    assert verdict.contrasts[0].estimate == 0.0123456789012345
    assert verdict.contrasts[0].interval_lower == -0.02
    assert verdict.review.adjudication.status == "NOT_REQUIRED"


def test_v2_represents_n1_interval_statistics_as_explicit_nulls() -> None:
    one_seed_contract, payload = _one_seed_payload()

    verdict = ValidatedD1Verdict.from_mapping(
        payload,
        decision_contract=one_seed_contract,
    )

    assert verdict.contrasts[0].interval_lower is None
    assert verdict.contrasts[0].interval_upper is None


@pytest.mark.parametrize(
    ("index", "decision"),
    [(0, "FAVORS_LEFT"), (2, "INTERACTION_DETECTED")],
)
def test_n1_null_intervals_require_inconclusive_decisions(index: int, decision: str) -> None:
    contract, payload = _one_seed_payload()
    payload["contrasts"][index]["decision"] = decision  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError, match="null interval"):
        ValidatedD1Verdict.from_mapping(payload, decision_contract=contract)


def test_interval_presence_must_match_sample_size() -> None:
    contract, payload = _one_seed_payload()
    payload["contrasts"][0]["interval_lower"] = 0.0  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError, match="both be finite"):
        ValidatedD1Verdict.from_mapping(payload, decision_contract=contract)

    contract, payload = _one_seed_payload()
    payload["contrasts"][0]["interval_upper"] = 0.0  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError, match="both be finite"):
        ValidatedD1Verdict.from_mapping(payload, decision_contract=contract)

    contract, payload = _one_seed_payload()
    payload["contrasts"][0]["interval_lower"] = 0.0  # type: ignore[index]
    payload["contrasts"][0]["interval_upper"] = 0.0  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError, match="explicit null"):
        ValidatedD1Verdict.from_mapping(payload, decision_contract=contract)

    payload = machine_verdict()
    payload["contrasts"][0]["interval_lower"] = None  # type: ignore[index]
    payload["contrasts"][0]["interval_upper"] = None  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError, match="finite interval"):
        _validated(payload)


@pytest.mark.parametrize("field", ("filed_artifacts", "machine_sha256", "machine_md5"))
def test_v2_refuses_any_machine_self_hash_or_v1_artifact_field(field: str) -> None:
    payload = machine_verdict()
    payload[field] = "not-admissible"
    with pytest.raises(D1VerdictValidationError, match="unknown"):
        _validated(payload)


def test_v2_rejects_v1_schema_before_decision_interpretation() -> None:
    payload = machine_verdict()
    payload["schema_version"] = "d1-verdict/v1"
    with pytest.raises(D1VerdictValidationError, match="schema_version"):
        _validated(payload)


def test_contract_binds_sets_constants_integrity_and_decision_table() -> None:
    changed: dict[str, object] = json.loads(json.dumps(machine_verdict()))
    changed["registered_constants"]["seeds"] = [1, 2]
    with pytest.raises(D1VerdictValidationError, match="registered_constants"):
        _validated(changed)

    changed = json.loads(json.dumps(machine_verdict()))
    changed["integrity"]["pairing_verified"] = False
    with pytest.raises(D1VerdictValidationError, match="integrity"):
        _validated(changed)

    changed = json.loads(json.dumps(machine_verdict()))
    changed["clauses"]["language"]["status"] = "FAIL"
    with pytest.raises(D1VerdictValidationError, match="decision"):
        _validated(changed)

    changed = json.loads(json.dumps(machine_verdict()))
    changed["arms"] = changed["arms"][:-1]
    with pytest.raises(D1VerdictValidationError, match="arm"):
        _validated(changed)

    changed = json.loads(json.dumps(machine_verdict()))
    changed["contrasts"] = changed["contrasts"][:-1]
    with pytest.raises(D1VerdictValidationError, match="contrast_set"):
        _validated(changed)

    changed = json.loads(json.dumps(machine_verdict()))
    del changed["clauses"]["language"]
    with pytest.raises(D1VerdictValidationError, match="clause"):
        _validated(changed)


def test_review_major_requires_adjudication_consistent_with_registered_clearance() -> None:
    payload = machine_verdict()
    payload["review"] = {
        "open_major_findings": ["MAJOR-1"],
        "adjudication": {"status": "NOT_REQUIRED", "major_findings": []},
    }
    with pytest.raises(D1VerdictValidationError, match="adjudication"):
        _validated(payload)
