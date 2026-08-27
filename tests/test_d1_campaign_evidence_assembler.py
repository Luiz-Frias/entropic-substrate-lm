"""Durable finalized-cell custody becomes real campaign capture material."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from entropic_substrate.d1_campaign_capture import D1CampaignCaptureService
from entropic_substrate.d1_campaign_evidence_assembler import D1CampaignEvidenceAssembler
from entropic_substrate.d1_finalized_cell_custody import parse_d1_finalized_cell_custody
from entropic_substrate.d1_finalized_cell_custody_repository import (
    D1FinalizedCellCustodyRepository,
)
from entropic_substrate.d1_numerical_defect import (
    D1DefectCheckpointReference,
    D1NonfiniteClassification,
    D1TerminalNumericalDefectRecord,
)
from entropic_substrate.evidence.d1_capture_artifacts import D1ArtifactRole
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalEvent,
    JournalCheckpointState,
    JournalEventKind,
)
from entropic_substrate.evidence.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.experiments.d1_launch_attempt_codec import (
    launch_attempt_ledger_to_mapping,
)
from entropic_substrate.experiments.d1_primitives import EvidenceClass, RunClass, canonical_json

from .test_d1_campaign_capture import _completed
from .test_d1_campaign_queue import _campaign, _campaign_with_defect

pytestmark = pytest.mark.unit


def _envelope(value: dict[str, object]) -> dict[str, object]:
    text = canonical_json(value)
    return {"manifest": value, "sha256": hashlib.sha256(text.encode()).hexdigest()}


def _record(intent: object, start: object, ledger: object, root: Path, *, failed: bool = False):
    plan = intent.plan
    run_root = root / "runs" / plan.logical_run_id
    run_root.mkdir(parents=True)
    journal = run_root / "run.jsonl"
    checkpoint = run_root / "run.pt"
    journal.write_bytes(f"finalized journal {plan.logical_run_id}\n".encode())
    checkpoint.write_bytes(f"checkpoint {plan.logical_run_id}\n".encode())
    journal_sha = hashlib.sha256(journal.read_bytes()).hexdigest()
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    defect = (
        D1TerminalNumericalDefectRecord(
            "nats_per_byte",
            D1NonfiniteClassification.NAN,
            "d1.eval.nats_per_byte.nonfinite.v1",
            2,
            2,
            2,
            plan.profile.arm,
            plan.seed,
            plan.run.execution.pairing_id,
            plan.logical_run_id,
            RunClass.CONFIRMATORY,
            0,
            start.execution_segment_id,
            plan.plan_digest,
            plan.profile.family.family_digest,
            plan.run.construction_digest,
            plan.run.execution_digest,
            plan.run.aggregation_digest,
            D1DefectCheckpointReference.receipt(checkpoint_sha),
            EvidenceClass.MEASURED,
        )
        if failed
        else None
    )
    terminal = D1JournalEvent(
        JournalEventKind.RUN_END,
        1,
        plan.logical_run_id,
        plan.run.execution_digest,
        plan.run.aggregation_digest,
        plan.run.construction_digest,
        plan.profile.family.family_digest,
        RunClass.CONFIRMATORY,
        2,
        {"outcome": "success"},
    )
    state = JournalCheckpointState(
        plan.logical_run_id, plan.run.execution_digest, 1, journal_sha, 2
    )
    finalization = {
        "billing": {"kind": "reconciled"},
        "journal_state": state.manifest,
        "quality_export_ready": True,
        "schema_version": "d1-finalized-cell-evidence-v1",
        "terminal_event": terminal.manifest,
        "trainer_outcome": "failed" if failed else "completed",
    }
    value = {
        "arm": plan.profile.arm.value,
        "attempt_ledger": _envelope(launch_attempt_ledger_to_mapping(ledger)),
        "billing_posture": "reconciled",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "finalization": _envelope(finalization),
        "intent": _envelope(intent.manifest),
        "journal": {"path": str(journal), "sha256": journal_sha},
        "logical_run_id": plan.logical_run_id,
        "numerical_defect": None if defect is None else defect.manifest,
        "plan_digest": plan.plan_digest,
        "quality_export_ready": True,
        "registered_design_digest": intent.registered_design_digest,
        "quarantined_siblings": [],
        "schema_version": "d1-finalized-cell-custody-v4",
        "seed": plan.seed,
        "segment_starts": [run_start_to_mapping(start)],
        "trainer_outcome": "failed" if failed else "completed",
    }
    return parse_d1_finalized_cell_custody(canonical_json(value))


def _case(tmp_path: Path, *, defect: bool = False):
    if defect:
        campaign, intents, starts, defect_index, _ = _campaign_with_defect(tmp_path / "lifecycle")
    else:
        campaign, _ = _completed(tmp_path / "lifecycle")
        _, intents, starts = _campaign()
        defect_index = -1
    source = tmp_path / "source"
    custody_root = source / "custody"
    custody_root.mkdir(parents=True)
    repository = D1FinalizedCellCustodyRepository(custody_root)
    artifacts = {}
    for index, (intent, start, result) in enumerate(
        zip(intents, starts, campaign.results, strict=True)
    ):
        record = _record(
            intent,
            start,
            result.attempt_ledger,
            source,
            failed=index == defect_index,
        )
        artifacts[intent.plan.logical_run_id] = repository.save(record)
    roster = D1CampaignCaptureService.expected_roster_from_matrix(campaign)
    return campaign, roster, source, repository, artifacts


def test_assembler_builds_complete_inputs_only_from_durable_cell_custody(
    tmp_path: Path,
) -> None:
    campaign, roster, source, repository, artifacts = _case(tmp_path)

    material = D1CampaignEvidenceAssembler(
        source_root=source,
        custody=repository,
        cell_artifacts=artifacts,
    ).prepare(campaign, roster)

    assert material.roster == roster
    assert material.terminal_defect_inputs == ()
    assert material.attempt_evidence == ()
    assert tuple(item.logical_run_id for item in material.complete_inputs) == tuple(
        item.logical_run_id for item in roster.expected_confirmatory
    )
    for item in material.complete_inputs:
        assert item.journal.role is D1ArtifactRole.JOURNAL
        assert item.checkpoint.role is D1ArtifactRole.CHECKPOINT
        assert len(item.billing_sidecars) == 1
        assert item.billing_sidecars[0].role is D1ArtifactRole.BILLING_SIDECAR


def test_assembler_refuses_missing_swapped_or_drifted_cell_evidence(tmp_path: Path) -> None:
    campaign, roster, source, repository, artifacts = _case(tmp_path)
    first, second = tuple(artifacts)[:2]

    with pytest.raises(ValueError, match="artifact set"):
        D1CampaignEvidenceAssembler(
            source_root=source,
            custody=repository,
            cell_artifacts={key: value for key, value in artifacts.items() if key != first},
        ).prepare(campaign, roster)

    swapped = dict(artifacts)
    swapped[first], swapped[second] = swapped[second], swapped[first]
    with pytest.raises(ValueError, match="identity"):
        D1CampaignEvidenceAssembler(
            source_root=source,
            custody=repository,
            cell_artifacts=swapped,
        ).prepare(campaign, roster)

    record = repository.load(artifacts[first])
    Path(record.journal.path).write_bytes(b"drift\n")
    with pytest.raises(ValueError, match="journal"):
        D1CampaignEvidenceAssembler(
            source_root=source,
            custody=repository,
            cell_artifacts=artifacts,
        ).prepare(campaign, roster)


def test_assembler_partitions_registered_failure_into_terminal_defect_lane(
    tmp_path: Path,
) -> None:
    campaign, roster, source, repository, artifacts = _case(tmp_path, defect=True)

    material = D1CampaignEvidenceAssembler(
        source_root=source,
        custody=repository,
        cell_artifacts=artifacts,
    ).prepare(campaign, roster)

    assert len(material.complete_inputs) == 3
    assert len(material.terminal_defect_inputs) == 1
    defect = material.terminal_defect_inputs[0]
    assert defect.logical_run_id == campaign.matrix.terminal_defect_run_ids[0]
    roles = {item.role for item in defect.supporting_artifacts}
    assert D1ArtifactRole.NUMERICAL_DEFECT_CHECKPOINT in roles
    assert D1ArtifactRole.BILLING_SIDECAR in roles
