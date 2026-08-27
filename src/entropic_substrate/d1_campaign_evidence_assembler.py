"""Assemble completed-campaign capture lanes from durable finalized-cell custody."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast

from entropic_substrate.d1_campaign_capture import D1CampaignCaptureMaterial
from entropic_substrate.d1_campaign_queue import D1CompletedCampaign
from entropic_substrate.d1_finalized_cell_custody import D1FinalizedCellCustodyRecord
from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1ArtifactRole,
    identify_d1_artifact,
)
from entropic_substrate.evidence.d1_capture_bundle import D1CaptureInput
from entropic_substrate.evidence.d1_capture_roster import D1ExpectedRun, D1LaunchRoster
from entropic_substrate.evidence.lifecycle import ArtifactIdentity

__all__ = ["D1CampaignEvidenceAssembler", "D1FinalizedCellCustodyLoaderPort"]

_DVC_POINTER = "results/captured.dvc"


class D1FinalizedCellCustodyLoaderPort(Protocol):
    def load(self, artifact: ArtifactIdentity) -> D1FinalizedCellCustodyRecord: ...


def _source_root(value: Path) -> Path:
    if not isinstance(cast(object, value), Path):
        raise TypeError("campaign evidence source_root must be a Path")
    root = Path(os.path.abspath(value))
    if root.is_symlink() or not root.is_dir():
        raise ValueError("campaign evidence source_root must be a real directory")
    return root


def _capture_artifact(
    root: Path,
    artifact: ArtifactIdentity,
    role: D1ArtifactRole,
) -> D1ArtifactIdentity:
    if type(artifact) is not ArtifactIdentity or artifact.content_hash.algorithm != "sha256":
        raise TypeError(f"campaign {role.value} requires SHA-256 ArtifactIdentity")
    identified = identify_d1_artifact(
        root,
        artifact.path,
        role=role,
        dvc_pointer_path=_DVC_POINTER,
    )
    if identified.sha256 != artifact.content_hash.digest:
        raise ValueError(f"campaign {role.value} identity differs from its bytes")
    return identified


def _matches_expected(
    record: D1FinalizedCellCustodyRecord,
    expected: D1ExpectedRun,
    campaign: D1CompletedCampaign,
) -> None:
    record.require_factory_attested()
    if (
        record.logical_run_id != expected.logical_run_id
        or record.arm is not expected.arm
        or record.seed != expected.seed
        or record.plan_digest != expected.plan_digest
        or record.registered_design_digest != campaign.design_digest
    ):
        raise ValueError("finalized-cell custody identity differs from campaign roster")
    events = record.attempt_ledger.events
    if not events or campaign.attempt_ledger.events[: len(events)] != events:
        raise ValueError("finalized-cell custody ledger is not a campaign prefix")


class D1CampaignEvidenceAssembler:
    """Build complete capture inputs without retaining operator result objects."""

    def __init__(
        self,
        *,
        source_root: Path,
        custody: D1FinalizedCellCustodyLoaderPort,
        cell_artifacts: Mapping[str, ArtifactIdentity],
    ) -> None:
        if not hasattr(custody, "load"):
            raise TypeError("campaign evidence assembler requires a custody loader")
        if not isinstance(cast(object, cell_artifacts), Mapping):
            raise TypeError("campaign cell artifacts must be a mapping")
        self._root = _source_root(source_root)
        self._custody = custody
        self._artifacts = MappingProxyType(dict(cell_artifacts))

    def _input(
        self,
        campaign: D1CompletedCampaign,
        expected: D1ExpectedRun,
        artifact: ArtifactIdentity,
    ) -> D1CaptureInput:
        try:
            record = self._custody.load(artifact)
        except Exception as error:
            raise ValueError(f"finalized-cell custody load failed: {error}") from error
        if type(record) is not D1FinalizedCellCustodyRecord:
            raise TypeError("campaign custody loader returned a non-nominal record")
        _matches_expected(record, expected, campaign)
        if (
            artifact.content_hash.algorithm != "sha256"
            or artifact.content_hash.digest != record.record_digest
        ):
            raise ValueError("finalized-cell artifact identity differs from custody record")
        journal = _capture_artifact(self._root, record.journal, D1ArtifactRole.JOURNAL)
        checkpoint = _capture_artifact(
            self._root,
            record.checkpoint,
            D1ArtifactRole.CHECKPOINT,
        )
        billing = _capture_artifact(
            self._root,
            artifact,
            D1ArtifactRole.BILLING_SIDECAR,
        )
        quarantined = tuple(
            _capture_artifact(self._root, item, D1ArtifactRole.QUARANTINED_SIBLING)
            for item in record.quarantined_siblings
        )
        return D1CaptureInput(
            expected.logical_run_id,
            journal,
            checkpoint,
            (billing,),
            quarantined,
        )

    def prepare(
        self,
        campaign: D1CompletedCampaign,
        roster: D1LaunchRoster,
    ) -> D1CampaignCaptureMaterial:
        if type(campaign) is not D1CompletedCampaign or type(roster) is not D1LaunchRoster:
            raise TypeError("campaign evidence assembly requires nominal campaign and roster")
        expected_ids = roster.expected_run_ids
        if frozenset(self._artifacts) != expected_ids:
            raise ValueError("campaign finalized-cell artifact set differs from roster")
        if roster.attempts:
            raise ValueError("completed campaign assembler requires explicit attempt evidence")
        complete = tuple(
            self._input(campaign, expected, self._artifacts[expected.logical_run_id])
            for expected in roster.expected_confirmatory
        )
        return D1CampaignCaptureMaterial(roster, complete, (), ())
