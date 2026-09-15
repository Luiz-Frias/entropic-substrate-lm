"""Production-shaped D1 science from captured evidence through filed successor gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entropic_substrate.d1_filing_application import (
    D1LocalFilingArtifactStore,
    finalize_d1_filing,
    prepare_d1_filing_bundle,
)
from entropic_substrate.evidence.d1_filing import validate_d1_filing_manifest
from entropic_substrate.evidence.d1_filing_graph import verify_d1_filing
from entropic_substrate.evidence.d1_filing_receipts import D1FilingRevisions
from entropic_substrate.evidence.d1_lifecycle_values import D1ProcessState
from entropic_substrate.evidence.d1_review_adjudication import D1_NO_MAJOR_OUTCOME
from entropic_substrate.evidence.lifecycle import ActorRole

from .d1_capture_aggregate_fixtures import heterogeneous_family
from .test_d1_capture_service import _service as _capture_service
from .test_d1_filing_application import _filing_git, _paths
from .test_d1_process_manager import _actor, _artifact
from .test_d1_process_manager_gate import _adjudicated
from .test_d1_real_arm_verdict_assembly import _lineage
from .test_d1_registered_design_hydration import _RAW_PREREGISTRATION

pytestmark = pytest.mark.unit


def test_golden_scientific_flywheel_files_registered_evidence_and_derives_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defect_root = tmp_path / "terminal-defect"
    defect_root.mkdir()
    roster, complete, defects, attempts, source_root = heterogeneous_family(defect_root)
    transport_root = tmp_path / "terminal-defect-transport"
    transport_root.mkdir()
    capture, capture_events, _capture_repo = _capture_service(
        transport_root,
        source_root,
    )

    defect_receipt = capture.capture_heterogeneous(
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )

    assert capture_events == ["export", "dvc_add", "dvc_push", "cloud", "git"]
    assert len(defect_receipt.aggregate.terminal_defects) == 1
    assert len(defect_receipt.aggregate.complete_runs) == 3
    assert len(defect_receipt.aggregate.pair_exclusions) == 1
    exclusion = defect_receipt.aggregate.pair_exclusions[0]
    expected_members = tuple(
        item.logical_run_id
        for item in defect_receipt.aggregate.roster.expected_confirmatory
        if item.pairing_id == exclusion.pairing_id
    )
    assert exclusion.member_run_ids == expected_members
    assert len(exclusion.member_run_ids) == 4
    assert defect_receipt.aggregate.dvc_handoff.analysis_ready is True

    lineage = _lineage(tmp_path / "finite-lineage", monkeypatch)

    assert lineage.analysis_receipt.manifest["filing_eligible"] is True
    assert lineage.execution.report == f"{D1_NO_MAJOR_OUTCOME}\n".encode()
    assert lineage.adjudication.status == "NOT_REQUIRED"
    assert lineage.reveal.review_packet_sha256 == lineage.execution.request.packet.packet_digest
    lineage.execution.require_factory_attested()
    lineage.reveal.require_factory_attested()
    lineage.adjudication.require_factory_attested()

    prepared = prepare_d1_filing_bundle(
        verified=lineage.verified,
        analysis_receipt=lineage.analysis_receipt,
        execution=lineage.execution,
        reveal=lineage.reveal,
        adjudication=lineage.adjudication,
        verdict="SUPPORTED",
        narrative_body=b"# D1 stabilized latent verdict\n",
        preregistration_bytes=_RAW_PREREGISTRATION,
        paths=_paths(),
    )
    artifact_root = tmp_path / "filing-root"
    (artifact_root / "filing").mkdir(parents=True)
    store = D1LocalFilingArtifactStore(artifact_root)
    store.publish(prepared, manifest_path="filing/manifest.json")
    revisions = D1FilingRevisions(
        "b" * 40,
        "c" * 40,
        "d" * 40,
        "e" * 40,
        "1" * 40,
        "2" * 40,
    )
    git = _filing_git(prepared, revisions)
    detached = validate_d1_filing_manifest(
        prepared.verdict.manifest,
        json.loads(prepared.manifest_bytes),
        decision_contract=prepared.verdict.verified.expected.decision_contract,
        artifact_root=artifact_root,
    )
    filing = verify_d1_filing(
        detached,
        prepared.verdict.verified.aggregate,
        revisions,
        store,
        git,
    )
    manager = _adjudicated(tmp_path / "filing-process", filing)
    finalized = finalize_d1_filing(
        prepared=prepared,
        artifact_root=artifact_root,
        manifest_path="filing/manifest.json",
        revisions=revisions,
        git=git,
        manager=manager,
        actor=_actor(ActorRole.OPERATOR),
        filing_event_id="golden-verdict",
        gate_artifact=_artifact(170, "golden-gate"),
        gate_event_id="golden-gate",
    )

    assert finalized is not None
    assert finalized.filing.graph_digest == filing.graph_digest
    assert finalized.terminal.state is D1ProcessState.GATE_AUTHORIZED
    assert finalized.terminal.eligible_arms == ("sg", "ac", "both")
