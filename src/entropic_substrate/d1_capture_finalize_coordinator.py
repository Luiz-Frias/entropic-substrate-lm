"""Finalize one persisted D1 capture preparation and append CAPTURED."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from entropic_substrate.d1_campaign_capture import (
    D1CampaignCaptureMaterial,
    D1CampaignCapturePort,
    D1CampaignCaptureService,
    D1CapturedCampaign,
)
from entropic_substrate.d1_campaign_queue import D1CompletedCampaign
from entropic_substrate.evidence.d1_capture_preparation_custody import (
    D1CampaignCaptureAuthority,
)
from entropic_substrate.evidence.d1_capture_preparation_custody_repository import (
    D1CapturePreparationCustodyRepository,
)
from entropic_substrate.evidence.d1_capture_roster import D1LaunchRoster
from entropic_substrate.evidence.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
)

__all__ = [
    "D1CampaignCaptureFinalizationCoordinator",
    "D1PublishedCaptureFinalization",
]


class _FinalizationOnlyAssembler:
    def prepare(
        self,
        campaign: D1CompletedCampaign,
        roster: D1LaunchRoster,
    ) -> D1CampaignCaptureMaterial:
        del campaign, roster
        raise RuntimeError("capture finalization must not re-run evidence assembly")


@dataclass(frozen=True, slots=True)
class D1PublishedCaptureFinalization:
    captured: D1CapturedCampaign
    preparation_artifact: ArtifactIdentity
    authority: D1CampaignCaptureAuthority

    def __post_init__(self) -> None:
        if type(self.captured) is not D1CapturedCampaign:
            raise TypeError("published capture finalization requires captured campaign")
        if type(self.preparation_artifact) is not ArtifactIdentity:
            raise TypeError("published capture finalization requires preparation artifact")
        if type(self.authority) is not D1CampaignCaptureAuthority:
            raise TypeError("published capture finalization requires campaign authority")
        campaign = self.captured.campaign
        if (
            self.authority.registered_design_digest != campaign.design_digest
            or self.authority.matrix_digest != campaign.matrix.matrix_digest
            or self.authority.attempt_ledger_digest != campaign.attempt_ledger.ledger_digest
        ):
            raise ValueError("published capture authority differs from captured campaign")


class D1CampaignCaptureFinalizationCoordinator:
    """Load exact custody, verify Git pointer evidence, then persist CAPTURED."""

    def __init__(
        self,
        *,
        capture: D1CampaignCapturePort,
        automation: ActorIdentity,
    ) -> None:
        if type(automation) is not ActorIdentity or automation.role is not ActorRole.AUTOMATION:
            raise ValueError("capture finalization coordinator requires automation actor")
        if not hasattr(capture, "prepare") or not hasattr(capture, "finalize"):
            raise TypeError("capture finalization coordinator requires capture port")
        self._capture = capture
        self._automation = automation

    def finalize(
        self,
        *,
        campaign: D1CompletedCampaign,
        preparation_artifact: ArtifactIdentity,
        repo_root: Path,
        preparation_custody_root: Path,
    ) -> D1PublishedCaptureFinalization:
        if type(campaign) is not D1CompletedCampaign:
            raise TypeError("capture finalization requires completed campaign")
        if type(preparation_artifact) is not ArtifactIdentity:
            raise TypeError("capture finalization requires preparation artifact")
        if not isinstance(cast(object, repo_root), Path) or not isinstance(
            cast(object, preparation_custody_root), Path
        ):
            raise TypeError("capture finalization roots must be Paths")
        repository = D1CapturePreparationCustodyRepository(preparation_custody_root)
        try:
            record = repository.load(preparation_artifact)
            service = D1CampaignCaptureService(
                assembler=_FinalizationOnlyAssembler(),
                capture=self._capture,
                automation=self._automation,
            )
            prepared = service.rehydrate_preparation(
                campaign=campaign,
                record=record,
                capture_root=repo_root / "results" / "captured",
            )
            captured = service.finalize(prepared)
            return D1PublishedCaptureFinalization(
                captured,
                preparation_artifact,
                record.authority,
            )
        finally:
            repository.close()
