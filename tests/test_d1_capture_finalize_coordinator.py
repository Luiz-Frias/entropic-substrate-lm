"""COMMIT_REQUIRED custody finalizes only after exact Git pointer proof."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.d1_campaign_capture import D1HeterogeneousCaptureAdapter
from entropic_substrate.d1_capture_finalize_coordinator import (
    D1CampaignCaptureFinalizationCoordinator,
    D1PublishedCaptureFinalization,
)
from entropic_substrate.d1_capture_prepare_coordinator import (
    D1CampaignCapturePreparationCoordinator,
)
from entropic_substrate.evidence.d1_capture_ports import D1CaptureTransportError
from entropic_substrate.evidence.d1_capture_preparation_custody_repository import (
    D1CapturePreparationCustodyRepositoryError,
)
from entropic_substrate.evidence.d1_lifecycle_repository import D1LifecycleRepository
from entropic_substrate.evidence.d1_lifecycle_values import D1ProcessState
from entropic_substrate.evidence.d1_process_manager import D1ProcessManager
from entropic_substrate.evidence.lifecycle import (
    ActorIdentity,
    ActorRole,
    ContentHash,
)

from .test_d1_campaign_evidence_assembler import _case
from .test_d1_capture_service import _service

pytestmark = pytest.mark.unit


def _prepared(tmp_path: Path):  # type: ignore[no-untyped-def]
    campaign, _roster, source, finalized_repository, artifacts = _case(tmp_path / "campaign")
    transport = tmp_path / "transport"
    transport.mkdir()
    capture, events, repo_root = _service(transport, source)
    port = D1HeterogeneousCaptureAdapter(capture)
    automation = ActorIdentity("capture", "capture-session", ActorRole.AUTOMATION)
    preparation_root = tmp_path / "preparations"
    preparation_root.mkdir()
    published = D1CampaignCapturePreparationCoordinator(
        capture=port,
        automation=automation,
    ).prepare(
        campaign=campaign,
        cell_artifacts=artifacts,
        source_root=source,
        finalized_cell_custody_root=finalized_repository.root,
        preparation_custody_root=preparation_root,
    )
    coordinator = D1CampaignCaptureFinalizationCoordinator(
        capture=port,
        automation=automation,
    )
    return campaign, coordinator, published, preparation_root, repo_root, events


def _reloaded_state(campaign: object) -> D1ProcessState:
    manager = D1ProcessManager.load(
        D1LifecycleRepository(campaign.process_manager.repository.path)  # type: ignore[attr-defined]
    )
    assert type(manager) is D1ProcessManager
    return manager.state


def test_finalize_verifies_git_then_persists_captured_lifecycle(tmp_path: Path) -> None:
    campaign, coordinator, prepared, preparation_root, repo_root, events = _prepared(tmp_path)

    finalized = coordinator.finalize(
        campaign=campaign,
        preparation_artifact=prepared.artifact,
        repo_root=repo_root,
        preparation_custody_root=preparation_root,
    )

    assert type(finalized) is D1PublishedCaptureFinalization
    assert finalized.captured.process_manager.state is D1ProcessState.CAPTURED
    assert finalized.captured.capture.pointer_commit == "abcdef1"
    assert finalized.preparation_artifact == prepared.artifact
    assert events == ["export", "dvc_add", "dvc_push", "cloud", "git"]
    assert _reloaded_state(campaign) is D1ProcessState.CAPTURED


def test_pointer_drift_refuses_before_git_and_lifecycle_capture(tmp_path: Path) -> None:
    campaign, coordinator, prepared, preparation_root, repo_root, events = _prepared(tmp_path)
    pointer = repo_root / prepared.record.pointer.pointer_path
    pointer.write_bytes(prepared.record.pointer_bytes + b"\n")

    with pytest.raises(D1CaptureTransportError, match="changed"):
        coordinator.finalize(
            campaign=campaign,
            preparation_artifact=prepared.artifact,
            repo_root=repo_root,
            preparation_custody_root=preparation_root,
        )

    assert events == ["export", "dvc_add", "dvc_push", "cloud"]
    assert _reloaded_state(campaign) is D1ProcessState.OPERATOR_FINALIZED


def test_substituted_preparation_artifact_refuses_before_git(tmp_path: Path) -> None:
    campaign, coordinator, prepared, preparation_root, repo_root, events = _prepared(tmp_path)
    substituted = replace(
        prepared.artifact,
        content_hash=ContentHash("sha256", "0" * 64),
    )

    with pytest.raises(D1CapturePreparationCustodyRepositoryError, match="artifact"):
        coordinator.finalize(
            campaign=campaign,
            preparation_artifact=substituted,
            repo_root=repo_root,
            preparation_custody_root=preparation_root,
        )

    assert events == ["export", "dvc_add", "dvc_push", "cloud"]
    assert _reloaded_state(campaign) is D1ProcessState.OPERATOR_FINALIZED


def test_finalize_requires_automation_actor() -> None:
    with pytest.raises(ValueError, match="automation"):
        D1CampaignCaptureFinalizationCoordinator(
            capture=object(),  # type: ignore[arg-type]
            automation=ActorIdentity("operator", "session", ActorRole.OPERATOR),
        )
