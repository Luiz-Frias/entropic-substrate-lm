"""Forward-only D1 v3 absent estimates cannot masquerade as historical v2."""

from dataclasses import replace

import pytest

from entropic_substrate.kernel.identity.d1_primitives import thaw_for_digest
from entropic_substrate.kernel.verdict.d1_evaluation_values import D1CosineGap
from entropic_substrate.kernel.verdict.d1_statistical_gaps import (
    D1EndpointGap,
    D1EstimateGap,
    D1GapCell,
)
from entropic_substrate.ladder.file.d1_validator import ValidatedD1Verdict

from .d1_verdict_v2_fixtures import decision_contract, machine_verdict

pytestmark = pytest.mark.unit


def unavailable_verdict():
    base = decision_contract()
    contract = replace(
        base,
        registered_constants=thaw_for_digest(base.registered_constants),
        contrast_rules={
            key: replace(
                rule, endpoint=replace(rule.endpoint, metric="mean_pairwise_cosine_distance")
            )
            for key, rule in base.contrast_rules.items()
        },
    )
    value = machine_verdict(verdict="VALID_NEGATIVE")
    value["schema_version"] = "d1-verdict/v3"
    gap = D1EstimateGap((D1GapCell("none", 1, D1EndpointGap(((20000, 40, D1CosineGap(1, 4)),))),))
    for row in value["contrasts"]:
        row["endpoint"]["metric"] = "mean_pairwise_cosine_distance"
        row.update(
            estimate=None,
            sample_sd=None,
            interval_lower=None,
            interval_upper=None,
            unavailable=gap.manifest(),
        )
        row["decision"] = (
            "INTERACTION_INCONCLUSIVE"
            if row["contrast_kind"] == "FACTORIAL_INTERACTION"
            else "INCONCLUSIVE"
        )
    return contract, value


def test_v3_keeps_negative_clauses_and_undefined_contrasts() -> None:
    contract, value = unavailable_verdict()
    parsed = ValidatedD1Verdict.from_mapping(value, decision_contract=contract)
    assert parsed.verdict == "VALID_NEGATIVE" and parsed.exclusions == ()
    assert all(
        row.estimate is None and row.unavailable is not None and row.n == 4
        for row in parsed.contrasts
    )
    assert parsed.clauses_by_arm["sg"]["noncollapse"].status == "FAIL"
    value["schema_version"] = "d1-verdict/v2"
    with pytest.raises(ValueError):
        ValidatedD1Verdict.from_mapping(value, decision_contract=contract)


def test_gap_does_not_permit_numeric_estimate_or_absent_witness() -> None:
    contract, value = unavailable_verdict()
    value["contrasts"][0]["estimate"] = 0.0
    with pytest.raises(ValueError):
        ValidatedD1Verdict.from_mapping(value, decision_contract=contract)
    _, value = unavailable_verdict()
    del value["contrasts"][0]["unavailable"]
    with pytest.raises(ValueError):
        ValidatedD1Verdict.from_mapping(value, decision_contract=contract)
