"""Fresh D1 analysis re-admits captured evidence through public telemetry."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import entropic_substrate.ladder.capture.reader.telemetry as telemetry_api
from d1_launch_admission_fixtures import (
    PREREG_BLOB,
    PREREG_COMMIT,
    PREREG_PATH,
    contract,
    family,
)
from entropic_substrate.kernel.identity.d1_primitives import (
    D1Arm,
    RunClass,
    canonical_digest,
    thaw_for_digest,
)
from entropic_substrate.ladder.analyze.d1_capture import (
    D1AnalysisAdmissionError,
    admit_d1_analysis_capture,
    issue_d1_factorial_exclusion_authority,
)
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate import (
    D1CapturePartitionProof,
    D1HeterogeneousCaptureAggregate,
)
from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import D1ArtifactIdentity
from entropic_substrate.ladder.capture.custody.d1_capture_bundle import (
    D1DvcCaptureHandoff,
    D1ValidatedRunCapture,
)
from entropic_substrate.ladder.capture.custody.d1_capture_roster import (
    D1ExpectedRun,
    D1LaunchRoster,
    D1PairExclusion,
    D1PairExclusionReason,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import JournalIdentity
from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1RunEvidence
from entropic_substrate.ladder.register.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)

from .d1_capture_fixtures import capture_family
from .test_d1_factorial_analysis import _whole_pair_exclusion

pytestmark = pytest.mark.unit


def _design_and_contract():
    decision = contract(family(), seeds=(7,))
    admitted = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family=family(),
            decision_contract=decision,
            registered_seeds=(7,),
            preregistration=D1PreregistrationArtifact(
                path=PREREG_PATH,
                blob_sha256=PREREG_BLOB,
                commit=PREREG_COMMIT,
            ),
        )
    )
    assert type(admitted) is D1RegisteredDesign
    return admitted, decision


def _bound(artifact: D1ArtifactIdentity) -> D1ArtifactIdentity:
    return replace(
        artifact,
        dvc=artifact.dvc.bind(pointer_sha256="9" * 64, object_id="8" * 32 + ".dir"),
    )


def _analysis_case(tmp_path: Path):
    old_roster, inputs, root = capture_family(tmp_path)
    real_runs = {
        item.logical_run_id: telemetry_api.read_d1_run_evidence(root / item.journal.normalized_path)
        for item in inputs
    }
    design, decision = _design_and_contract()
    candidates = tuple(
        replace(
            item,
            family_digest=design.family_digest,
            construction_digest=design.arm_construction_digests[item.arm.value],
        )
        for item in old_roster.expected_confirmatory
    )
    run_evidence: dict[str, D1RunEvidence] = {}
    expected: list[D1ExpectedRun] = []
    for item in candidates:
        original = real_runs[item.logical_run_id]
        plan = cast(dict[str, object], thaw_for_digest(original.plan_manifest))
        plan.update(
            {
                "arm": item.arm.value,
                "construction_digest": item.construction_digest,
                "family_digest": item.family_digest,
                "pairing_id": item.pairing_id,
                "preregistration": {
                    "commit": design.preregistration.commit,
                    "family_digest": design.family_digest,
                    "path": design.preregistration.path,
                },
                "seed": item.seed,
            }
        )
        plan_digest = canonical_digest(plan)
        finalized = replace(item, plan_digest=plan_digest)
        expected.append(finalized)
        run_evidence[finalized.logical_run_id] = replace(
            original,
            identity=JournalIdentity(
                logical_run_id=finalized.logical_run_id,
                execution_digest=finalized.execution_digest,
                aggregation_digest=finalized.aggregation_digest,
                construction_digest=finalized.construction_digest,
                family_digest=finalized.family_digest,
                run_class=RunClass.CONFIRMATORY,
            ),
            plan_digest=plan_digest,
            plan_manifest=plan,  # type: ignore[arg-type]
        )
    expected_tuple = tuple(expected)
    roster = D1LaunchRoster(
        preregistration_path=design.preregistration.path,
        preregistration_commit=design.preregistration.commit,
        family_digest=design.family_digest,
        expected_confirmatory=expected_tuple,
        attempts=(),
    )
    inputs_by_id = {}
    artifacts: list[D1ArtifactIdentity] = []
    for item in inputs:
        bound = replace(
            item,
            journal=_bound(item.journal),
            checkpoint=_bound(item.checkpoint),
            billing_sidecars=tuple(_bound(value) for value in item.billing_sidecars),
        )
        inputs_by_id[item.logical_run_id] = bound
        artifacts.extend(bound.artifacts)
    complete = tuple(
        D1ValidatedRunCapture(
            expected=item,
            capture_input=inputs_by_id[item.logical_run_id],
            terminal_step=real_runs[item.logical_run_id].evals[-1].step,
            data_order_segments=real_runs[item.logical_run_id].data_order_segments,
            checkpoint_receipts=real_runs[item.logical_run_id].checkpoint_receipts,
        )
        for item in expected_tuple
    )
    ids = roster.expected_run_ids
    aggregate = D1HeterogeneousCaptureAggregate.from_validated_evidence(
        roster=roster,
        partition_proof=D1CapturePartitionProof(ids, ids, frozenset()),
        complete_runs=complete,
        terminal_defects=(),
        attempts=(),
        pair_exclusions=(),
        dvc_handoff=D1DvcCaptureHandoff(tuple(artifacts)),
    )
    return root, design, decision, aggregate, run_evidence


def test_analysis_reopens_every_finite_cell_through_public_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, design, decision, aggregate, runs = _analysis_case(tmp_path)
    calls: list[str] = []

    def read(_path: Path, **expected: object) -> D1RunEvidence:
        logical_run_id = str(expected["expected_logical_run_id"])
        calls.append(logical_run_id)
        return runs[logical_run_id]

    monkeypatch.setattr(telemetry_api, "read_d1_run_evidence", read)
    admitted = admit_d1_analysis_capture(
        root=root,
        design=design,
        decision_contract=decision,
        aggregate=aggregate,
    )

    assert calls == [item.expected.logical_run_id for item in aggregate.complete_runs]
    assert tuple((cell.seed, cell.arm) for cell in admitted.cells) == tuple(
        (7, arm) for arm in D1Arm
    )
    assert all("nats_per_byte" in cell.metrics for cell in admitted.cells)
    assert admitted.exclusions == ()
    assert admitted.capture_digest == aggregate.manifest_digest
    assert len(admitted.input_artifacts) == 12


def test_factorial_authority_is_sealed_to_the_exact_admitted_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, design, decision, aggregate, runs = _analysis_case(tmp_path)
    monkeypatch.setattr(
        telemetry_api,
        "read_d1_run_evidence",
        lambda _path, **expected: runs[str(expected["expected_logical_run_id"])],
    )
    admitted = admit_d1_analysis_capture(
        root=root,
        design=design,
        decision_contract=decision,
        aggregate=aggregate,
    )
    authority = issue_d1_factorial_exclusion_authority(
        design=design,
        decision_contract=decision,
        aggregate=aggregate,
        admitted=admitted,
    )
    assert authority.capture_digest == aggregate.manifest_digest
    assert authority.exclusions == ()

    for forged in (
        replace(admitted, capture_digest="0" * 64),
        replace(admitted, exclusions=(_whole_pair_exclusion(7),)),
    ):
        with pytest.raises(D1AnalysisAdmissionError, match="factorial_authority"):
            issue_d1_factorial_exclusion_authority(
                design=design,
                decision_contract=decision,
                aggregate=aggregate,
                admitted=forged,
            )


def test_analysis_rejects_inherited_exclusion_without_terminal_defect_partition(
    tmp_path: Path,
) -> None:
    root, design, decision, aggregate, _ = _analysis_case(tmp_path)
    claim = D1PairExclusion(
        pairing_id=aggregate.roster.pairing_ids[0],
        member_run_ids=tuple(
            item.logical_run_id for item in aggregate.roster.expected_confirmatory
        ),
        reason=D1PairExclusionReason.NONFINITE_FINAL_LOSS,
        evidence_paths=(aggregate.complete_runs[0].capture_input.journal.normalized_path,),
    )
    forged = D1HeterogeneousCaptureAggregate.from_validated_evidence(
        roster=aggregate.roster,
        partition_proof=aggregate.partition_proof,
        complete_runs=aggregate.complete_runs,
        terminal_defects=(),
        attempts=(),
        pair_exclusions=(claim,),
        dvc_handoff=aggregate.dvc_handoff,
    )

    with pytest.raises(D1AnalysisAdmissionError, match="recomputed exclusions"):
        admit_d1_analysis_capture(
            root=root,
            design=design,
            decision_contract=decision,
            aggregate=forged,
        )


def test_analysis_refuses_pending_dvc_or_public_reader_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, design, decision, aggregate, _ = _analysis_case(tmp_path)
    pending = replace(
        aggregate,
        dvc_handoff=D1DvcCaptureHandoff(
            tuple(
                replace(
                    item,
                    dvc=replace(
                        item.dvc, state=item.dvc.state.PENDING, pointer_sha256=None, object_id=None
                    ),
                )
                for item in aggregate.dvc_handoff.artifacts
            )
        ),
    )
    with pytest.raises(D1AnalysisAdmissionError, match="DVC"):
        admit_d1_analysis_capture(
            root=root, design=design, decision_contract=decision, aggregate=pending
        )

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise ValueError("public reader refused")

    monkeypatch.setattr(telemetry_api, "read_d1_run_evidence", refuse)
    with pytest.raises(D1AnalysisAdmissionError, match="public_reader") as captured:
        admit_d1_analysis_capture(
            root=root, design=design, decision_contract=decision, aggregate=aggregate
        )
    assert isinstance(captured.value.__cause__, ValueError)
