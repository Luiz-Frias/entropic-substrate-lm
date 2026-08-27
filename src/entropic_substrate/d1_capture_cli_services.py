"""Real public CLI services for the two-phase D1 capture transaction."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from entropic_substrate.d1_campaign_authority_loader import (
    D1CampaignAuthorityLoader,
    D1CampaignAuthorityLoaderError,
    D1LoadedCampaignAuthority,
)
from entropic_substrate.d1_campaign_queue import D1CompletedCampaign
from entropic_substrate.d1_campaign_request import (
    D1CampaignCommandRequest,
    D1CampaignRequestError,
    parse_d1_campaign_request,
)
from entropic_substrate.d1_capture_finalize_coordinator import (
    D1PublishedCaptureFinalization,
)
from entropic_substrate.d1_capture_finalize_request import (
    D1CaptureFinalizeRequest,
    D1CaptureFinalizeRequestError,
    parse_d1_capture_finalize_request,
)
from entropic_substrate.d1_capture_prepare_coordinator import (
    D1PublishedCapturePreparation,
)
from entropic_substrate.d1_cli import (
    D1CliCommand,
    D1CliInvalidEvidence,
    D1CliInvocation,
    D1CliOperatorGap,
)
from entropic_substrate.evidence.d1_capture_ports import D1CaptureTransportError
from entropic_substrate.evidence.d1_capture_preparation_custody_repository import (
    D1CapturePreparationCustodyRepositoryError,
)
from entropic_substrate.evidence.d1_lifecycle_values import D1ProcessState
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.experiments.d1_config import D1LaunchProfile
from entropic_substrate.experiments.d1_confirmatory_matrix import D1GitProvenancePort
from entropic_substrate.experiments.d1_primitives import canonical_digest, canonical_json

__all__ = ["D1CaptureFinalizeCliService", "D1CapturePrepareCliService"]


class D1CampaignAuthorityLoaderPort(Protocol):
    def load_persisted(
        self,
        request: D1CampaignCommandRequest,
        *,
        manifest_path: Path,
    ) -> D1LoadedCampaignAuthority: ...


class D1CapturePreparationCoordinatorPort(Protocol):
    def prepare(
        self,
        *,
        campaign: D1CompletedCampaign,
        cell_artifacts: Mapping[str, ArtifactIdentity],
        source_root: Path,
        finalized_cell_custody_root: Path,
        preparation_custody_root: Path,
    ) -> D1PublishedCapturePreparation: ...


class D1CapturePreparationCoordinatorFactoryPort(Protocol):
    def build(
        self,
        request: D1CampaignCommandRequest,
    ) -> D1CapturePreparationCoordinatorPort: ...


class D1CaptureFinalizationCoordinatorPort(Protocol):
    def finalize(
        self,
        *,
        campaign: D1CompletedCampaign,
        preparation_artifact: ArtifactIdentity,
        repo_root: Path,
        preparation_custody_root: Path,
    ) -> D1PublishedCaptureFinalization: ...


class D1CaptureFinalizationCoordinatorFactoryPort(Protocol):
    def build(
        self,
        request: D1CampaignCommandRequest,
    ) -> D1CaptureFinalizationCoordinatorPort: ...


def _request(invocation: D1CliInvocation, catalog: tuple[D1LaunchProfile, ...]):
    try:
        return parse_d1_campaign_request(
            canonical_json(dict(invocation.payload)),
            catalog=catalog,
        )
    except (D1CampaignRequestError, TypeError, ValueError) as error:
        raise D1CliInvalidEvidence(f"D1 capture prepare request was refused: {error}") from error


def _finalize_request(
    invocation: D1CliInvocation,
    catalog: tuple[D1LaunchProfile, ...],
) -> D1CaptureFinalizeRequest:
    try:
        return parse_d1_capture_finalize_request(
            canonical_json(dict(invocation.payload)),
            catalog=catalog,
        )
    except (D1CaptureFinalizeRequestError, TypeError, ValueError) as error:
        raise D1CliInvalidEvidence(f"D1 capture finalize request was refused: {error}") from error


def _result(
    published: D1PublishedCapturePreparation,
    loaded: D1LoadedCampaignAuthority,
) -> Mapping[str, object]:
    record = published.record
    pointer = record.pointer
    checkpoint = loaded.checkpoint_receipt
    return {
        "campaign_digest": record.authority.campaign_digest,
        "checkpoint_generation": {
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "size_bytes": checkpoint.size_bytes,
        },
        "matrix_digest": record.authority.matrix_digest,
        "pointer": {
            "byte_count": len(record.pointer_bytes),
            "object_id": pointer.object_id,
            "path": pointer.pointer_path,
            "sha256": pointer.pointer_sha256,
        },
        "preparation_artifact": {
            "path": published.artifact.path,
            "sha256": published.artifact.content_hash.digest,
        },
        "request_digest": canonical_digest(loaded.request.manifest),
        "required_next_command": D1CliCommand.CAPTURE_FINALIZE.value,
        "schema_version": "d1-capture-prepare-result-v1",
        "status": "commit_required",
    }


def _finalize_result(
    published: D1PublishedCaptureFinalization,
    loaded: D1LoadedCampaignAuthority,
) -> Mapping[str, object]:
    captured = published.captured
    receipt = captured.capture
    manager = captured.process_manager
    prefix = manager.history.prefix
    return {
        "campaign_digest": published.authority.campaign_digest,
        "capture_receipt_digest": canonical_digest(receipt.manifest),
        "checkpoint_generation": loaded.checkpoint_receipt.sha256,
        "lifecycle": {
            "event_count": prefix.event_count,
            "prefix_sha256": prefix.prefix_sha256,
            "state": manager.state.value,
        },
        "matrix_digest": captured.campaign.matrix.matrix_digest,
        "pointer_commit": receipt.pointer_commit,
        "pointer_identity": receipt.manifest["pointer_identity"],
        "preparation_artifact": {
            "path": published.preparation_artifact.path,
            "sha256": published.preparation_artifact.content_hash.digest,
        },
        "schema_version": "d1-capture-finalize-result-v1",
        "status": "captured",
    }


class D1CapturePrepareCliService:
    """Parse, rehydrate, re-attest, prepare, and publish COMMIT_REQUIRED."""

    def __init__(
        self,
        *,
        catalog: tuple[D1LaunchProfile, ...],
        authority_loader: D1CampaignAuthorityLoaderPort,
        git: D1GitProvenancePort,
        coordinator_factory: D1CapturePreparationCoordinatorFactoryPort,
    ) -> None:
        if (
            type(catalog) is not tuple
            or not catalog
            or any(type(item) is not D1LaunchProfile for item in catalog)
        ):
            raise ValueError("capture prepare CLI requires profile catalog")
        if not hasattr(authority_loader, "load_persisted"):
            raise TypeError("capture prepare CLI requires authority loader")
        if any(not hasattr(git, name) for name in ("commit_time", "blob_sha256", "is_ancestor")):
            raise TypeError("capture prepare CLI requires Git provenance port")
        if not hasattr(coordinator_factory, "build"):
            raise TypeError("capture prepare CLI requires coordinator factory")
        self._catalog = catalog
        self._loader = authority_loader
        self._git = git
        self._factory = coordinator_factory

    def execute(self, invocation: D1CliInvocation) -> Mapping[str, object]:
        if type(invocation) is not D1CliInvocation:
            raise TypeError("capture prepare CLI requires D1CliInvocation")
        if invocation.command is not D1CliCommand.CAPTURE_PREPARE:
            raise D1CliInvalidEvidence("D1 capture prepare received the wrong command")
        request = _request(invocation, self._catalog)
        try:
            expected = replace(
                request,
                locations=request.locations.resolve(invocation.manifest_path.parent),
            )
            loaded = self._loader.load_persisted(
                request,
                manifest_path=invocation.manifest_path,
            )
        except D1CampaignAuthorityLoaderError as error:
            raise D1CliInvalidEvidence(str(error)) from error
        except (TypeError, ValueError) as error:
            raise D1CliInvalidEvidence(
                f"D1 capture prepare authority was refused: {error}"
            ) from error
        if type(loaded) is not D1LoadedCampaignAuthority or loaded.request != expected:
            raise D1CliInvalidEvidence("D1 capture prepare loader returned substituted authority")
        try:
            campaign = D1CampaignAuthorityLoader.completed_campaign(loaded, git=self._git)
        except D1CampaignAuthorityLoaderError as error:
            if error.stage == "complete":
                raise D1CliOperatorGap(
                    "D1 campaign is not complete; run d1-campaign-run"
                ) from error
            raise D1CliInvalidEvidence(str(error)) from error
        artifacts = {
            cell.logical_run_id: cell.finalized_cell_artifact
            for cell in loaded.checkpoint.completed_cells
        }
        try:
            coordinator = self._factory.build(loaded.request)
            published = coordinator.prepare(
                campaign=campaign,
                cell_artifacts=artifacts,
                source_root=loaded.request.locations.artifact_root,
                finalized_cell_custody_root=(loaded.request.locations.finalized_cell_custody_root),
                preparation_custody_root=(
                    loaded.request.locations.capture_preparation_custody_root
                ),
            )
        except D1CaptureTransportError as error:
            if error.code in {"dvc_failure", "dvc_command", "cloud_mismatch", "cloud_status"}:
                raise D1CliOperatorGap(str(error)) from error
            raise D1CliInvalidEvidence(str(error)) from error
        except (TypeError, ValueError) as error:
            raise D1CliInvalidEvidence(f"D1 capture prepare was refused: {error}") from error
        if type(published) is not D1PublishedCapturePreparation:
            raise TypeError("capture prepare coordinator returned a non-nominal publication")
        return _result(published, loaded)


class D1CaptureFinalizeCliService:
    """Reload authority, verify one preparation, and return durable CAPTURED."""

    def __init__(
        self,
        *,
        catalog: tuple[D1LaunchProfile, ...],
        authority_loader: D1CampaignAuthorityLoaderPort,
        git: D1GitProvenancePort,
        coordinator_factory: D1CaptureFinalizationCoordinatorFactoryPort,
    ) -> None:
        if (
            type(catalog) is not tuple
            or not catalog
            or any(type(item) is not D1LaunchProfile for item in catalog)
        ):
            raise ValueError("capture finalize CLI requires profile catalog")
        if not hasattr(authority_loader, "load_persisted"):
            raise TypeError("capture finalize CLI requires authority loader")
        if any(not hasattr(git, name) for name in ("commit_time", "blob_sha256", "is_ancestor")):
            raise TypeError("capture finalize CLI requires Git provenance port")
        if not hasattr(coordinator_factory, "build"):
            raise TypeError("capture finalize CLI requires coordinator factory")
        self._catalog = catalog
        self._loader = authority_loader
        self._git = git
        self._factory = coordinator_factory

    def execute(self, invocation: D1CliInvocation) -> Mapping[str, object]:
        if type(invocation) is not D1CliInvocation:
            raise TypeError("capture finalize CLI requires D1CliInvocation")
        if invocation.command is not D1CliCommand.CAPTURE_FINALIZE:
            raise D1CliInvalidEvidence("D1 capture finalize received the wrong command")
        request = _finalize_request(invocation, self._catalog)
        campaign_request = request.campaign
        try:
            expected = replace(
                campaign_request,
                locations=campaign_request.locations.resolve(invocation.manifest_path.parent),
            )
            loaded = self._loader.load_persisted(
                campaign_request,
                manifest_path=invocation.manifest_path,
            )
        except D1CampaignAuthorityLoaderError as error:
            raise D1CliInvalidEvidence(str(error)) from error
        except (TypeError, ValueError) as error:
            raise D1CliInvalidEvidence(
                f"D1 capture finalize authority was refused: {error}"
            ) from error
        if type(loaded) is not D1LoadedCampaignAuthority or loaded.request != expected:
            raise D1CliInvalidEvidence("D1 capture finalize loader returned substituted authority")
        try:
            campaign = D1CampaignAuthorityLoader.completed_campaign(loaded, git=self._git)
        except D1CampaignAuthorityLoaderError as error:
            if error.stage == "complete":
                raise D1CliOperatorGap(
                    "D1 campaign is not complete; run d1-campaign-run"
                ) from error
            raise D1CliInvalidEvidence(str(error)) from error
        root = loaded.request.locations.capture_preparation_custody_root
        digest = request.preparation_artifact.content_hash.digest
        expected_artifact = root / f"{digest}.json"
        artifact_path = Path(os.path.abspath(request.preparation_artifact.path))
        if artifact_path != expected_artifact:
            raise D1CliInvalidEvidence(
                "D1 capture finalize preparation artifact is outside its custody root"
            )
        try:
            coordinator = self._factory.build(loaded.request)
            published = coordinator.finalize(
                campaign=campaign,
                preparation_artifact=request.preparation_artifact,
                repo_root=loaded.request.locations.repo_root,
                preparation_custody_root=root,
            )
        except D1CapturePreparationCustodyRepositoryError as error:
            raise D1CliInvalidEvidence(str(error)) from error
        except D1CaptureTransportError as error:
            if error.code in {"pointer_uncommitted", "git_status"}:
                raise D1CliOperatorGap(str(error)) from error
            raise D1CliInvalidEvidence(str(error)) from error
        except (TypeError, ValueError) as error:
            raise D1CliInvalidEvidence(f"D1 capture finalize was refused: {error}") from error
        if (
            type(published) is not D1PublishedCaptureFinalization
            or published.preparation_artifact != request.preparation_artifact
            or published.captured.campaign.matrix.matrix_digest != campaign.matrix.matrix_digest
            or published.captured.campaign.design_digest != campaign.design_digest
            or published.authority.matrix_digest != campaign.matrix.matrix_digest
            or published.authority.registered_design_digest != campaign.design_digest
            or published.captured.process_manager.state is not D1ProcessState.CAPTURED
        ):
            raise D1CliInvalidEvidence("D1 capture finalize returned substituted authority")
        return _finalize_result(published, loaded)
