"""Registered D1 endpoint arithmetic keeps seed blocks as the sampling unit."""

from __future__ import annotations

from dataclasses import replace

import pytest

from d1_launch_admission_fixtures import (
    PREREG_BLOB,
    PREREG_COMMIT,
    PREREG_PATH,
    SEEDS,
    contract,
    family,
)
from entropic_substrate.analysis.d1_factorial import (
    D1AnalysisCell,
    D1MetricPoint,
    compute_d1_factorial,
)
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.experiments.d1_primitives import D1Arm
from entropic_substrate.experiments.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)

pytestmark = pytest.mark.unit


def _design(decision_contract: D1DecisionContract | None = None) -> D1RegisteredDesign:
    selected_contract = contract() if decision_contract is None else decision_contract
    admitted = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family=family(),
            decision_contract=selected_contract,
            registered_seeds=SEEDS,
            preregistration=D1PreregistrationArtifact(
                path=PREREG_PATH,
                blob_sha256=PREREG_BLOB,
                commit=PREREG_COMMIT,
            ),
        )
    )
    assert type(admitted) is D1RegisteredDesign
    return admitted


def _with_contrast_rules(base: D1DecisionContract, rules: dict[str, object]) -> D1DecisionContract:
    return D1DecisionContract(
        family_identity=base.family_identity,
        registered_constants={"seeds": list(SEEDS), "primary": {"alpha": 0.025}},
        arm_ids=base.arm_ids,
        contrast_ids=base.contrast_ids,
        clause_ids=base.clause_ids,
        contrast_rules=rules,  # type: ignore[arg-type]
        clause_rules=base.clause_rules,
        exclusion_rules=base.exclusion_rules,
        clause_precedence=base.clause_precedence,
        verdict_vocabulary=base.verdict_vocabulary,
        decision_rules=base.decision_rules,
    )


_ENDPOINTS = {
    D1Arm.NONE: (10.0, 11.0, 12.0, 13.0),
    D1Arm.SG: (11.0, 13.0, 15.0, 17.0),
    D1Arm.AC: (12.0, 13.0, 14.0, 15.0),
    D1Arm.BOTH: (14.0, 17.0, 20.0, 23.0),
}


def _cells() -> tuple[D1AnalysisCell, ...]:
    cells: list[D1AnalysisCell] = []
    for seed_index, seed in enumerate(SEEDS):
        for arm in D1Arm:
            endpoint = _ENDPOINTS[arm][seed_index]
            points = tuple(
                D1MetricPoint(step=index * 500, eval_index=index, value=endpoint)
                for index in range(6)
            )
            cells.append(
                D1AnalysisCell(
                    arm=arm,
                    seed=seed,
                    pairing_id=f"pair-{seed}",
                    metrics={"nats_per_byte": points},
                )
            )
    return tuple(cells)


def test_registered_contrasts_use_seed_blocks_not_eval_rows() -> None:
    result = compute_d1_factorial(
        design=_design(),
        decision_contract=contract(),
        cells=_cells(),
    )

    sg = result.registered_contrasts["sg-main"]
    assert sg.paired_estimates == (1.0, 2.0, 3.0, 4.0)
    assert sg.estimate == 2.5
    assert sg.n == 4
    assert sg.df == 3
    assert sg.sample_sd == pytest.approx(1.2909944487358056)
    assert sg.standard_error == pytest.approx(0.6454972243679028)
    assert sg.t_critical == pytest.approx(4.176534846104503, rel=1e-11)
    assert sg.interval_lower == pytest.approx(-0.195941650636283)
    assert sg.interval_upper == pytest.approx(5.195941650636283)
    assert sg.evidence_class == "DERIVED_FROM_MEASURED"

    interaction = result.registered_contrasts["interaction"]
    assert interaction.paired_estimates == (1.0, 2.0, 3.0, 4.0)
    assert result.factorial_effects["sg_at_ac_off"].paired_estimates == (1.0, 2.0, 3.0, 4.0)
    assert result.factorial_effects["sg_at_ac_on"].paired_estimates == (2.0, 4.0, 6.0, 8.0)
    assert result.factorial_effects["sg_main"].paired_estimates == (1.5, 3.0, 4.5, 6.0)
    assert result.factorial_effects["interaction"].paired_estimates == (1.0, 2.0, 3.0, 4.0)


def test_roster_must_be_the_exact_complete_paired_four_arm_matrix() -> None:
    cells = _cells()
    with pytest.raises(ValueError, match="missing registered cells"):
        compute_d1_factorial(design=_design(), decision_contract=contract(), cells=cells[:-1])
    with pytest.raises(ValueError, match="duplicate cell"):
        compute_d1_factorial(
            design=_design(), decision_contract=contract(), cells=(*cells, cells[0])
        )
    mismatched = replace(cells[1], pairing_id="wrong-pair")
    with pytest.raises(ValueError, match="pairing_id"):
        compute_d1_factorial(
            design=_design(),
            decision_contract=contract(),
            cells=(cells[0], mismatched, *cells[2:]),
        )


def test_registered_endpoint_and_sample_size_rules_fail_loudly() -> None:
    broken_cells = list(_cells())
    broken_cells[0] = replace(
        broken_cells[0],
        metrics={"nats_per_byte": broken_cells[0].metrics["nats_per_byte"][-4:]},
    )
    with pytest.raises(ValueError, match="smoothing window"):
        compute_d1_factorial(
            design=_design(),
            decision_contract=contract(),
            cells=tuple(broken_cells),
        )

    wrong_n = contract()
    wrong_rules = {
        name: replace(rule, registered_n=5) for name, rule in wrong_n.contrast_rules.items()
    }
    wrong_n = _with_contrast_rules(wrong_n, wrong_rules)
    with pytest.raises(ValueError, match="registered n"):
        compute_d1_factorial(design=_design(wrong_n), decision_contract=wrong_n, cells=_cells())


def test_unknown_estimator_and_nonfinite_extracted_value_are_refused() -> None:
    unknown = contract()
    rules = {
        name: replace(rule, endpoint=replace(rule.endpoint, estimator="median_tail"))
        for name, rule in unknown.contrast_rules.items()
    }
    unknown = _with_contrast_rules(unknown, rules)
    with pytest.raises(ValueError, match="unsupported registered endpoint estimator"):
        compute_d1_factorial(design=_design(unknown), decision_contract=unknown, cells=_cells())

    with pytest.raises(ValueError, match="finite"):
        D1MetricPoint(step=1, eval_index=0, value=float("nan"))


def test_registered_nonexcluded_n_accepts_only_complete_surviving_seed_blocks() -> None:
    decision = contract()
    rules = {
        name: replace(rule, n_rule="NONEXCLUDED_SEED_COUNT", registered_n=None)
        for name, rule in decision.contrast_rules.items()
    }
    decision = _with_contrast_rules(decision, rules)

    result = compute_d1_factorial(
        design=_design(decision),
        decision_contract=decision,
        cells=_cells()[:-4],
    )

    assert result.included_seeds == SEEDS[:-1]
    assert result.excluded_seeds == SEEDS[-1:]
    assert all(value.n == 3 and value.df == 2 for value in result.registered_contrasts.values())
