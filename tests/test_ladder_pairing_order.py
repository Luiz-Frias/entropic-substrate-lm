"""Paired scientific statistics must compare the same measured source order."""

from dataclasses import replace

import pytest

from entropic_substrate.ladder.analyze.ladder_registration import (
    StatisticalEvaluation,
    StatisticalRun,
    paired_contrasts,
    read_statistical_design,
)

from .test_ladder_registration import sealed, specification

pytestmark = pytest.mark.unit


def cohort():
    raw = specification()
    payload, digest = sealed(raw)
    design = read_statistical_design(payload, expected_sha256=digest)
    runs = tuple(
        StatisticalRun(
            arm,
            seed,
            "pilot",
            "4" * 64,
            construction,
            (
                StatisticalEvaluation(0, 0, (("nats_per_byte", 1.0),)),
                StatisticalEvaluation(1, 4, (("nats_per_byte", 0.9),)),
                StatisticalEvaluation(2, 8, (("nats_per_byte", 0.8),)),
            ),
            data_order=((1, "a" * 64, 1), (2, "b" * 64, 2)),
        )
        for arm, construction in (("both", "3" * 64), ("sg", "5" * 64))
        for seed in (7, 8)
    )
    return runs, design


def test_measured_paired_order_and_read_old_d1_cohorts_both_work() -> None:
    runs, design = cohort()
    assert len(paired_contrasts(runs, design)[0].pairs) == 2
    legacy = tuple(replace(run, data_order=None) for run in runs)
    assert paired_contrasts(legacy, design) == paired_contrasts(runs, design)


def test_a_seed_may_have_its_own_order_but_all_its_arms_must_agree() -> None:
    runs, design = cohort()
    other = ((1, "c" * 64, 1), (2, "d" * 64, 2))
    paired = tuple(replace(run, data_order=other) if run.seed == 8 else run for run in runs)
    assert len(paired_contrasts(paired, design)[0].pairs) == 2
    with pytest.raises(ValueError, match="data-order"):
        paired_contrasts((replace(runs[0], data_order=other), *runs[1:]), design)


@pytest.mark.parametrize(
    "order",
    [
        None,
        (),
        ((1, "a" * 64, 1),),
        ((1, "a" * 64, 1), (2, "b" * 64, 2), (3, "c" * 64, 3)),
        ((0, "a" * 64, 0), (1, "a" * 64, 1), (2, "b" * 64, 2)),
        ((2, "b" * 64, 2), (1, "a" * 64, 1)),
        ((1, "a" * 63, 1), (2, "b" * 64, 2)),
        ((1, "A" * 64, 1), (2, "b" * 64, 2)),
        ((1, "a" * 64, 1), (2, "b" * 64, 1)),
        ((1, "a" * 64, True), (2, "b" * 64, 2)),
        ((True, "a" * 64, 1), (2, "b" * 64, 2)),
        [(1, "a" * 64, 1), (2, "b" * 64, 2)],
    ],
)
def test_missing_mixed_malformed_or_wrong_count_attestation_is_refused(order) -> None:
    runs, design = cohort()
    with pytest.raises((ValueError, TypeError)):
        paired_contrasts((replace(runs[0], data_order=order), *runs[1:]), design)


def test_audit_batch_count_saturates_at_registered_prefix_length() -> None:
    runs, design = cohort()
    later = StatisticalEvaluation(33, 132, (("nats_per_byte", 0.7),))
    runs = tuple(
        replace(run, evals=(*run.evals, later), data_order=(*run.data_order, (33, "c" * 64, 32)))
        for run in runs
    )
    assert paired_contrasts(runs, design)
    wrong = tuple(
        replace(run, data_order=(*run.data_order[:-1], (33, "c" * 64, 33))) for run in runs
    )
    with pytest.raises(ValueError, match="data-order"):
        paired_contrasts(wrong, design)


def test_frozen_prefix_cannot_drift_even_when_every_arm_repeats_the_same_error() -> None:
    runs, design = cohort()
    runs = tuple(
        replace(
            run,
            evals=(
                *run.evals,
                StatisticalEvaluation(32, 128, ()),
                StatisticalEvaluation(33, 132, ()),
            ),
            data_order=(*run.data_order, (32, "c" * 64, 32), (33, "d" * 64, 32)),
        )
        for run in runs
    )
    with pytest.raises(ValueError, match="prefix"):
        paired_contrasts(runs, design)
