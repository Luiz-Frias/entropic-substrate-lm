"""Undefined cosine affects only dependent estimates, never seed membership."""

from dataclasses import replace

import pytest

from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.kernel.verdict.d1_evaluation_values import D1CosineGap
from entropic_substrate.ladder.analyze.d1_factorial import (
    D1DiagnosticGapPoint,
    D1MetricPoint,
    compute_d1_factorial,
)

from .test_d1_clause_policy import _contract_with_policy, _policy
from .test_d1_factorial_analysis import _authority, _cells, _design
from .test_d1_report_diagnostics import _policy as diagnostic_policy

pytestmark = pytest.mark.unit
METRIC = "mean_pairwise_cosine_distance"


def cosine_case(*, gap_step: int = 2500):
    contract = _contract_with_policy(
        _policy(), contrast_metric=METRIC, diagnostic_policy=diagnostic_policy()
    )
    design = _design(contract)
    cells = []
    for cell in _cells():
        points = tuple(
            D1MetricPoint(p.step, p.eval_index, 0.5) for p in cell.metrics["nats_per_byte"]
        )
        gaps = {}
        if cell.arm is D1Arm.NONE and cell.seed == design.registered_seeds[0]:
            points = tuple(p for p in points if p.step != gap_step)
            gaps = {METRIC: (D1DiagnosticGapPoint(gap_step, gap_step // 500, D1CosineGap(1, 4)),)}
        cells.append(replace(cell, metrics={**cell.metrics, METRIC: points}, diagnostic_gaps=gaps))
    return contract, design, tuple(cells)


def test_gap_retains_roster_and_only_blocks_required_terms() -> None:
    contract, design, cells = cosine_case()
    result = compute_d1_factorial(
        design=design,
        decision_contract=contract,
        cells=cells,
        exclusion_authority=_authority(design),
    )
    assert result.included_seeds == design.registered_seeds and result.excluded_seeds == ()
    for name, rule in contract.contrast_rules.items():
        estimate = result.registered_contrasts[name]
        if any(term.arm_id == D1Arm.NONE.value for term in rule.terms):
            assert estimate.estimate is None and estimate.unavailable is not None
            assert estimate.paired_estimates[0] is None
            assert estimate.n == len(design.registered_seeds)
        else:
            assert estimate.estimate == 0.0 and estimate.unavailable is None
    for effect in result.factorial_effects.values():
        depends = any(arm is D1Arm.NONE for arm, _ in effect.terms)
        assert (effect.estimate is None) == depends


def test_early_gap_outside_registered_window_does_not_erase_later_measurements() -> None:
    contract, design, cells = cosine_case(gap_step=0)
    result = compute_d1_factorial(
        design=design,
        decision_contract=contract,
        cells=cells,
        exclusion_authority=_authority(design),
    )
    assert all(
        item.estimate == 0.0 and item.unavailable is None
        for item in result.registered_contrasts.values()
    )


def test_measured_and_gap_cursors_cannot_alias_one_eval_index() -> None:
    _, _, cells = cosine_case()
    with pytest.raises(ValueError, match="cursors"):
        replace(
            cells[0], diagnostic_gaps={METRIC: (D1DiagnosticGapPoint(2500, 4, D1CosineGap(1, 4)),)}
        )
