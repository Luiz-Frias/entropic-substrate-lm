"""D1 launch-roster and non-success attempt lanes are immutable and total."""

from __future__ import annotations

from dataclasses import replace

import pytest

from entropic_substrate.evidence.d1_capture_roster import (
    D1AttemptLane,
    D1DeclaredAttempt,
    D1ExpectedRun,
    D1LaunchRoster,
    D1PairExclusion,
    D1PairExclusionReason,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass

pytestmark = pytest.mark.unit


def _run(arm: D1Arm, *, seed: int = 7, pairing_id: str = "pair-7") -> D1ExpectedRun:
    digit = str(list(D1Arm).index(arm) + 1)
    return D1ExpectedRun(
        logical_run_id=f"d1-{arm.value}-s{seed}",
        arm=arm,
        seed=seed,
        pairing_id=pairing_id,
        family_digest="a" * 64,
        construction_digest=digit * 64,
        execution_digest=(digit * 63) + "a",
        aggregation_digest=(digit * 63) + "b",
        plan_digest=(digit * 63) + "c",
        code_commit="abcdef0",
        repository_digest="d" * 64,
        image_digest="sha256:" + "e" * 64,
        dataset_digest="f" * 64,
        split_digest="0" * 64,
        sampler_construction_digest="9" * 64,
    )


def _family() -> tuple[D1ExpectedRun, ...]:
    return tuple(_run(arm) for arm in D1Arm)


def test_roster_requires_the_exact_four_arm_pairing_block() -> None:
    roster = D1LaunchRoster(
        preregistration_path="docs/d1-preregistration.md",
        preregistration_commit="abcdef0",
        family_digest="a" * 64,
        expected_confirmatory=_family(),
        attempts=(),
    )

    assert roster.expected_run_ids == frozenset(run.logical_run_id for run in _family())
    assert roster.pairing_ids == ("pair-7",)
    assert roster.manifest["schema_version"] == "d1-launch-roster-v1"
    with pytest.raises(TypeError):
        roster.expected_confirmatory[0] = _run(D1Arm.NONE)  # type: ignore[index]


@pytest.mark.parametrize(
    "runs",
    [
        _family()[:-1],
        (*_family(), _family()[0]),
        (*_family()[:-1], replace(_family()[-1], pairing_id="pair-8")),
        (*_family()[:-1], replace(_family()[-1], seed=8)),
    ],
)
def test_roster_refuses_incomplete_duplicate_or_misaligned_pairing_blocks(
    runs: tuple[D1ExpectedRun, ...],
) -> None:
    with pytest.raises(ValueError):
        D1LaunchRoster(
            preregistration_path="docs/d1-preregistration.md",
            preregistration_commit="abcdef0",
            family_digest="a" * 64,
            expected_confirmatory=runs,
            attempts=(),
        )


def test_unsuccessful_and_pilot_attempts_are_declared_in_separate_lanes() -> None:
    attempts = (
        D1DeclaredAttempt(
            logical_run_id="d1-pilot-both-s42",
            arm=D1Arm.BOTH,
            seed=42,
            pairing_id="pilot-42",
            run_class=RunClass.PILOT,
            lane=D1AttemptLane.EXCLUDED_PILOT,
            disposition="registered numerical-range pilot",
        ),
        D1DeclaredAttempt(
            logical_run_id="d1-sg-s7-attempt-0",
            arm=D1Arm.SG,
            seed=7,
            pairing_id="pair-7",
            run_class=RunClass.CONFIRMATORY,
            lane=D1AttemptLane.ABORTED,
            disposition="trainer recorded a numerical defect",
        ),
        D1DeclaredAttempt(
            logical_run_id="d1-ac-s7-attempt-0",
            arm=D1Arm.AC,
            seed=7,
            pairing_id="pair-7",
            run_class=RunClass.CONFIRMATORY,
            lane=D1AttemptLane.ABANDONED,
            disposition="pod never reached trainer start",
        ),
    )
    roster = D1LaunchRoster(
        preregistration_path="docs/d1-preregistration.md",
        preregistration_commit="abcdef0",
        family_digest="a" * 64,
        expected_confirmatory=_family(),
        attempts=attempts,
    )

    assert tuple(item.lane for item in roster.attempts) == (
        D1AttemptLane.EXCLUDED_PILOT,
        D1AttemptLane.ABORTED,
        D1AttemptLane.ABANDONED,
    )
    with pytest.raises(ValueError, match="pilot lane"):
        replace(attempts[0], run_class=RunClass.CONFIRMATORY)
    with pytest.raises(ValueError, match="confirmatory"):
        replace(attempts[1], run_class=RunClass.PILOT)


def test_whole_pair_exclusion_hook_refuses_partial_pair_or_unregistered_reason() -> None:
    roster = D1LaunchRoster(
        preregistration_path="docs/d1-preregistration.md",
        preregistration_commit="abcdef0",
        family_digest="a" * 64,
        expected_confirmatory=_family(),
        attempts=(),
    )
    exclusion = D1PairExclusion(
        pairing_id="pair-7",
        member_run_ids=tuple(run.logical_run_id for run in _family()),
        reason=D1PairExclusionReason.NONFINITE_FINAL_LOSS,
        evidence_paths=("results/raw/d1-both-s7.jsonl",),
    )

    assert roster.validate_pair_exclusions((exclusion,)) == (exclusion,)
    with pytest.raises(ValueError, match="whole pairing block"):
        roster.validate_pair_exclusions((replace(exclusion, member_run_ids=("d1-both-s7",)),))
