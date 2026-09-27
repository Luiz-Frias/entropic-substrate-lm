"""Completed campaigns publish durable COMMIT_REQUIRED capture preparations."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.kernel.lifecycle import ActorIdentity, ActorRole
from entropic_substrate.ladder.capture.custody.d1_capture_preparation_custody_repository import (
    D1CapturePreparationCustodyRepository,
)
from entropic_substrate.ladder.capture.custody.d1_capture_service import D1CapturePreparationStatus
from entropic_substrate.ladder.run.campaign.d1_campaign_capture import D1HeterogeneousCaptureAdapter
from entropic_substrate.ladder.run.ckptsync.d1_capture_prepare_coordinator import (
    D1CampaignCapturePreparationCoordinator,
    D1PublishedCapturePreparation,
)

from .test_d1_campaign_evidence_assembler import _case
from .test_d1_capture_service import _service

pytestmark = pytest.mark.unit


def _coordinator(tmp_path: Path):  # type: ignore[no-untyped-def]
    campaign, _roster, source, finalized_repository, artifacts = _case(tmp_path / "campaign")
    transport = tmp_path / "transport"
    transport.mkdir()
    capture, events, _repo_root = _service(transport, source)
    preparation_root = tmp_path / "preparations"
    preparation_root.mkdir()
    coordinator = D1CampaignCapturePreparationCoordinator(
        capture=D1HeterogeneousCaptureAdapter(capture),
        automation=ActorIdentity("capture", "capture-session", ActorRole.AUTOMATION),
    )
    return (
        coordinator,
        campaign,
        source,
        finalized_repository.root,
        preparation_root,
        artifacts,
        events,
    )


def test_prepare_publishes_content_addressed_commit_required_custody(
    tmp_path: Path,
) -> None:
    coordinator, campaign, source, cells, preparations, artifacts, events = _coordinator(tmp_path)

    published = coordinator.prepare(
        campaign=campaign,
        cell_artifacts=artifacts,
        source_root=source,
        finalized_cell_custody_root=cells,
        preparation_custody_root=preparations,
    )

    assert type(published) is D1PublishedCapturePreparation
    assert published.record.status is D1CapturePreparationStatus.COMMIT_REQUIRED
    assert published.record.authority.matrix_digest == campaign.matrix.matrix_digest
    assert published.artifact.content_hash.digest == published.record.record_sha256
    repository = D1CapturePreparationCustodyRepository(preparations)
    try:
        restored = repository.load(published.artifact)
    finally:
        repository.close()
    assert restored == published.record
    assert events == ["export", "dvc_add", "dvc_push", "cloud"]


def test_missing_cell_custody_refuses_before_dvc_or_preparation_write(
    tmp_path: Path,
) -> None:
    coordinator, campaign, source, cells, preparations, artifacts, events = _coordinator(tmp_path)
    first = next(iter(artifacts))
    incomplete = {key: value for key, value in artifacts.items() if key != first}

    with pytest.raises(ValueError, match="artifact set"):
        coordinator.prepare(
            campaign=campaign,
            cell_artifacts=incomplete,
            source_root=source,
            finalized_cell_custody_root=cells,
            preparation_custody_root=preparations,
        )

    assert events == []
    assert list(preparations.iterdir()) == []


def test_prepare_requires_automation_actor(tmp_path: Path) -> None:
    _coordinator(tmp_path)

    with pytest.raises(ValueError, match="automation"):
        D1CampaignCapturePreparationCoordinator(
            capture=object(),  # type: ignore[arg-type]
            automation=ActorIdentity("operator", "session", ActorRole.OPERATOR),
        )
