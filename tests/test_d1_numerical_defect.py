"""Terminal numerical defects serialize classification, never NaN or infinity."""

from __future__ import annotations

import json

import pytest

from d1_trainer_fixtures import micro_plan, run_start
from entropic_substrate.kernel.identity.d1_primitives import D1Arm, EvidenceClass
from entropic_substrate.ladder.capture.reader.d1_numerical_defect import (
    D1CheckpointReferenceKind,
    D1DefectCheckpointReference,
    D1NonfiniteClassification,
    D1NonfiniteMetricError,
    D1TerminalNumericalDefectRecord,
    d1_nonfinite_metric_error,
    terminal_numerical_defect_from_mapping,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "classification"),
    [
        (float("nan"), D1NonfiniteClassification.NAN),
        (float("inf"), D1NonfiniteClassification.POSITIVE_INFINITY),
        (float("-inf"), D1NonfiniteClassification.NEGATIVE_INFINITY),
    ],
)
def test_nonfinite_source_guard_has_closed_classification_without_raw_value(
    value: float, classification: D1NonfiniteClassification
) -> None:
    error = d1_nonfinite_metric_error(
        metric_name="nats_per_byte",
        value=value,
        source_exception_code="d1.eval.nats_per_byte.nonfinite.v1",
    )

    assert type(error) is D1NonfiniteMetricError
    assert error.classification is classification
    assert not hasattr(error, "value")
    assert "nan" not in error.__dict__ and "inf" not in error.__dict__


def test_source_guard_refuses_finite_values_and_unregistered_names() -> None:
    with pytest.raises(ValueError, match="must be nonfinite"):
        d1_nonfinite_metric_error(
            metric_name="nats_per_byte",
            value=0.8,
            source_exception_code="d1.eval.nats_per_byte.nonfinite.v1",
        )
    with pytest.raises(ValueError, match="registered terminal metric"):
        d1_nonfinite_metric_error(
            metric_name="loss",
            value=float("nan"),
            source_exception_code="d1.eval.loss.nonfinite.v1",
        )


def test_terminal_record_binds_exact_cursor_run_checkpoint_and_evidence_class() -> None:
    plan = micro_plan(D1Arm.BOTH, seed=23)
    start = run_start(plan)
    error = d1_nonfinite_metric_error(
        metric_name="nats_per_byte",
        value=float("inf"),
        source_exception_code="d1.eval.nats_per_byte.nonfinite.v1",
    )
    record = D1TerminalNumericalDefectRecord.from_error(
        error,
        plan=plan,
        start=start,
        successful_step=2,
        eval_index=2,
        checkpoint=D1DefectCheckpointReference.receipt("c" * 64),
    )

    assert record.arm is D1Arm.BOTH
    assert record.seed == 23
    assert record.pairing_id == "d1-smoke-pair-23"
    assert record.logical_run_id == plan.logical_run_id
    assert record.successful_step == record.max_steps == 2
    assert record.eval_index == 2
    assert record.execution_segment_id == f"{plan.logical_run_id}:segment:0"
    assert record.checkpoint.kind is D1CheckpointReferenceKind.RECEIPT
    assert record.evidence_class is EvidenceClass.MEASURED
    assert record.classification is D1NonfiniteClassification.POSITIVE_INFINITY
    encoded = json.dumps(record.manifest, allow_nan=False, sort_keys=True)
    assert "Infinity" not in encoded and "NaN" not in encoded
    assert terminal_numerical_defect_from_mapping(record.manifest) == record


def test_checkpoint_gap_is_honest_but_distinct_from_a_receipt() -> None:
    gap = D1DefectCheckpointReference.gap("no_prior_checkpoint")

    assert gap.kind is D1CheckpointReferenceKind.GAP
    assert gap.sha256 is None
    with pytest.raises(ValueError, match="gap"):
        D1DefectCheckpointReference(
            kind=D1CheckpointReferenceKind.GAP,
            sha256="d" * 64,
            gap_code="no_prior_checkpoint",
        )


def test_terminal_record_refuses_nonterminal_cursor() -> None:
    plan = micro_plan(D1Arm.NONE)
    error = d1_nonfinite_metric_error(
        metric_name="nats_per_byte",
        value=float("nan"),
        source_exception_code="d1.eval.nats_per_byte.nonfinite.v1",
    )
    with pytest.raises(ValueError, match="terminal cursor"):
        D1TerminalNumericalDefectRecord.from_error(
            error,
            plan=plan,
            start=run_start(plan),
            successful_step=1,
            eval_index=1,
            checkpoint=D1DefectCheckpointReference.receipt("e" * 64),
        )
