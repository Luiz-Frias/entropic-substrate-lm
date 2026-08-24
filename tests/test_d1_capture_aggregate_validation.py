"""Heterogeneous D1 capture admission derives whole-pair exclusions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.evidence.d1_capture_aggregate_validation import (
    D1CaptureAggregateError,
    build_d1_capture_aggregate,
)
from entropic_substrate.evidence.d1_capture_roster import D1PairExclusionReason

from .d1_capture_aggregate_fixtures import heterogeneous_family

pytestmark = pytest.mark.unit


def test_aggregate_derives_exclusion_and_proves_exact_disjoint_roster_partition(
    tmp_path: Path,
) -> None:
    roster, complete, defects, attempts, root = heterogeneous_family(tmp_path)

    aggregate = build_d1_capture_aggregate(
        root,
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )

    assert aggregate.partition_proof.is_exact
    assert aggregate.partition_proof.expected_logical_run_ids == roster.expected_run_ids
    assert aggregate.partition_proof.complete_run_ids.isdisjoint(
        aggregate.partition_proof.terminal_defect_run_ids
    )
    assert len(aggregate.complete_runs) == 3
    assert len(aggregate.terminal_defects) == 1
    assert aggregate.attempts == attempts
    (exclusion,) = aggregate.pair_exclusions
    assert exclusion.reason is D1PairExclusionReason.NONFINITE_FINAL_LOSS
    assert frozenset(exclusion.member_run_ids) == roster.expected_run_ids
    assert exclusion.evidence_paths == (defects[0].journal.normalized_path,)
    assert aggregate.dvc_handoff.analysis_ready is False
    assert len(aggregate.dvc_handoff.artifacts) == 12
    assert aggregate.manifest["authority"]["mlflow"] == "optional_mirror_not_input"  # type: ignore[index]


@pytest.mark.parametrize("mutation", ["missing", "overlap", "duplicate"])
def test_expected_roster_must_equal_exact_disjoint_complete_and_defect_union(
    tmp_path: Path, mutation: str
) -> None:
    roster, complete, defects, attempts, root = heterogeneous_family(tmp_path)
    if mutation == "missing":
        defects = ()
    elif mutation == "overlap":
        complete = (*complete, replace(complete[0], logical_run_id=defects[0].logical_run_id))
    else:
        defects = (*defects, defects[0])

    with pytest.raises(D1CaptureAggregateError) as captured:
        build_d1_capture_aggregate(
            root,
            roster=roster,
            complete_inputs=complete,
            terminal_defect_inputs=defects,
            attempt_evidence=attempts,
        )

    assert captured.value.code == "capture_partition"


def test_defect_artifact_drift_and_nondefect_abort_refuse_aggregate(tmp_path: Path) -> None:
    roster, complete, defects, attempts, root = heterogeneous_family(tmp_path)
    (root / defects[0].journal.normalized_path).write_text("drift\n", encoding="utf-8")

    with pytest.raises(D1CaptureAggregateError) as drift:
        build_d1_capture_aggregate(
            root,
            roster=roster,
            complete_inputs=complete,
            terminal_defect_inputs=defects,
            attempt_evidence=attempts,
        )

    assert drift.value.code == "defect_artifact"
