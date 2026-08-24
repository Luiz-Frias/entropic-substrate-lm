"""Immutable values for mixed complete-run and terminal-defect D1 capture."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.d1_numerical_defect import (
    D1DefectCheckpointReference,
    D1NonfiniteClassification,
    D1TerminalNumericalDefectRecord,
)
from entropic_substrate.evidence.d1_capture_aggregate import (
    D1CapturePartitionProof,
    D1HeterogeneousCaptureAggregate,
    D1TerminalDefectInput,
    D1ValidatedTerminalDefect,
)
from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactRole,
    identify_d1_artifact,
)
from entropic_substrate.evidence.d1_capture_roster import D1ExpectedRun
from entropic_substrate.evidence.resource_provenance import execution_segment_id
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass

pytestmark = pytest.mark.unit


def _record(expected) -> D1TerminalNumericalDefectRecord:
    return D1TerminalNumericalDefectRecord(
        metric_name="nats_per_byte",
        classification=D1NonfiniteClassification.NAN,
        source_exception_code="d1.eval.nats_per_byte.nonfinite.v1",
        successful_step=2,
        max_steps=2,
        eval_index=2,
        arm=expected.arm,
        seed=expected.seed,
        pairing_id=expected.pairing_id,
        logical_run_id=expected.logical_run_id,
        run_class=RunClass.CONFIRMATORY,
        execution_segment_index=0,
        execution_segment_id=execution_segment_id(expected.logical_run_id, 0),
        plan_digest=expected.plan_digest,
        family_digest=expected.family_digest,
        construction_digest=expected.construction_digest,
        execution_digest=expected.execution_digest,
        aggregation_digest=expected.aggregation_digest,
        checkpoint=D1DefectCheckpointReference.gap("no_prior_checkpoint"),
    )


def _expected() -> D1ExpectedRun:
    return D1ExpectedRun(
        logical_run_id="d1-sg-s7",
        arm=D1Arm.SG,
        seed=7,
        pairing_id="pair-7",
        family_digest="a" * 64,
        construction_digest="2" * 64,
        execution_digest="6" * 64,
        aggregation_digest="7" * 64,
        plan_digest="8" * 64,
        code_commit="abcdef0",
        repository_digest="d" * 64,
        image_digest="sha256:" + "5" * 64,
        dataset_digest="f" * 64,
        split_digest="0" * 64,
        sampler_construction_digest="9" * 64,
    )


def _defect_input(tmp_path: Path, logical_run_id: str) -> D1TerminalDefectInput:
    tmp_path.mkdir(parents=True, exist_ok=True)
    journal = tmp_path / "defect.jsonl"
    billing = tmp_path / "defect.billing.json"
    journal.write_text('{"event":"trainer_aborted"}\n', encoding="utf-8")
    billing.write_text('{"amount":"1.00"}\n', encoding="utf-8")
    return D1TerminalDefectInput(
        logical_run_id=logical_run_id,
        journal=identify_d1_artifact(
            tmp_path,
            journal,
            role=D1ArtifactRole.JOURNAL,
            dvc_pointer_path="results/captured.dvc",
        ),
        supporting_artifacts=(
            identify_d1_artifact(
                tmp_path,
                billing,
                role=D1ArtifactRole.BILLING_SIDECAR,
                dvc_pointer_path="results/captured.dvc",
            ),
        ),
    )


def test_partition_proof_requires_exact_disjoint_union() -> None:
    proof = D1CapturePartitionProof(
        expected_logical_run_ids=frozenset({"a", "b", "c"}),
        complete_run_ids=frozenset({"a", "b"}),
        terminal_defect_run_ids=frozenset({"c"}),
    )

    assert proof.is_exact is True
    with pytest.raises(ValueError, match="disjoint"):
        replace(proof, terminal_defect_run_ids=frozenset({"b", "c"}))
    with pytest.raises(ValueError, match="union"):
        replace(proof, terminal_defect_run_ids=frozenset())


def test_terminal_defect_input_requires_journal_billing_and_unique_artifact_paths(
    tmp_path: Path,
) -> None:
    value = _defect_input(tmp_path, "d1-sg-s7")

    assert value.artifacts == (value.journal, *value.supporting_artifacts)
    with pytest.raises(ValueError, match="billing"):
        replace(value, supporting_artifacts=())
    with pytest.raises(ValueError, match="role"):
        replace(value, journal=value.supporting_artifacts[0])
    duplicate = replace(value.journal, role=D1ArtifactRole.BILLING_SIDECAR)
    with pytest.raises(ValueError, match="unique"):
        replace(value, supporting_artifacts=(value.supporting_artifacts[0], duplicate))


def test_validated_defect_binds_expected_run_and_canonical_record(tmp_path: Path) -> None:
    expected = _expected()
    capture = _defect_input(tmp_path / "defect-input", expected.logical_run_id)
    value = D1ValidatedTerminalDefect(expected, capture, _record(expected), ((0, "e" * 64, 2),))

    assert value.expected is expected
    with pytest.raises(ValueError, match="identity"):
        replace(value, record=replace(value.record, logical_run_id="different-run"))


def test_aggregate_is_factory_only() -> None:
    with pytest.raises(TypeError, match="factory-only"):
        D1HeterogeneousCaptureAggregate(  # type: ignore[arg-type]
            roster=None,
            partition_proof=None,
            complete_runs=(),
            terminal_defects=(),
            attempts=(),
            pair_exclusions=(),
            dvc_handoff=None,
        )
