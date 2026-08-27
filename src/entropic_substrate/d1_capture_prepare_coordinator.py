"""Publish one completed D1 campaign as durable COMMIT_REQUIRED custody."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from entropic_substrate.d1_campaign_capture import (
    D1CampaignCapturePort,
    D1CampaignCaptureService,
)
from entropic_substrate.d1_campaign_evidence_assembler import D1CampaignEvidenceAssembler
from entropic_substrate.d1_campaign_queue import D1CompletedCampaign
from entropic_substrate.d1_finalized_cell_custody_repository import (
    D1FinalizedCellCustodyRepository,
)
from entropic_substrate.evidence.d1_capture_preparation_custody import (
    D1CapturePreparationCustodyRecord,
)
from entropic_substrate.evidence.d1_capture_preparation_custody_repository import (
    D1CapturePreparationCustodyRepository,
)
from entropic_substrate.evidence.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
)

__all__ = [
    "D1CampaignCapturePreparationCoordinator",
    "D1PublishedCapturePreparation",
]


@dataclass(frozen=True, slots=True)
class D1PublishedCapturePreparation:
    record: D1CapturePreparationCustodyRecord
    artifact: ArtifactIdentity

    def __post_init__(self) -> None:
        if type(self.record) is not D1CapturePreparationCustodyRecord:
            raise TypeError("published capture preparation requires custody record")
        if type(self.artifact) is not ArtifactIdentity:
            raise TypeError("published capture preparation requires artifact identity")
        if (
            self.artifact.content_hash.algorithm != "sha256"
            or self.artifact.content_hash.digest != self.record.record_sha256
        ):
            raise ValueError("published preparation artifact differs from custody record")


class D1CampaignCapturePreparationCoordinator:
    """Compose existing campaign, capture, and custody capabilities exactly once."""

    def __init__(
        self,
        *,
        capture: D1CampaignCapturePort,
        automation: ActorIdentity,
    ) -> None:
        if type(automation) is not ActorIdentity or automation.role is not ActorRole.AUTOMATION:
            raise ValueError("capture preparation coordinator requires automation actor")
        if not hasattr(capture, "prepare") or not hasattr(capture, "finalize"):
            raise TypeError("capture preparation coordinator requires capture port")
        self._capture = capture
        self._automation = automation

    def prepare(
        self,
        *,
        campaign: D1CompletedCampaign,
        cell_artifacts: Mapping[str, ArtifactIdentity],
        source_root: Path,
        finalized_cell_custody_root: Path,
        preparation_custody_root: Path,
    ) -> D1PublishedCapturePreparation:
        if type(campaign) is not D1CompletedCampaign:
            raise TypeError("capture preparation requires completed campaign")
        if not isinstance(cast(object, cell_artifacts), Mapping):
            raise TypeError("capture preparation cell artifacts must be a mapping")
        for value in (
            source_root,
            finalized_cell_custody_root,
            preparation_custody_root,
        ):
            if not isinstance(cast(object, value), Path):
                raise TypeError("capture preparation roots must be Paths")
        finalized = D1FinalizedCellCustodyRepository(finalized_cell_custody_root)
        preparations = D1CapturePreparationCustodyRepository(preparation_custody_root)
        try:
            service = D1CampaignCaptureService(
                assembler=D1CampaignEvidenceAssembler(
                    source_root=source_root,
                    custody=finalized,
                    cell_artifacts=cell_artifacts,
                ),
                capture=self._capture,
                automation=self._automation,
            )
            prepared = service.prepare(campaign)
            record = service.preparation_custody(prepared)
            artifact = preparations.save(record)
            return D1PublishedCapturePreparation(record, artifact)
        finally:
            preparations.close()
            finalized.close()
