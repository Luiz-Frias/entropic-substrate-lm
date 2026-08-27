"""Real public CLI services for the two-phase D1 capture transaction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from entropic_substrate.d1_campaign_authority_loader import (
    D1CampaignAuthorityLoader,
    D1CampaignAuthorityLoaderError,
    D1LoadedCampaignAuthority,
)
from entropic_substrate.d1_campaign_request import (
    D1CampaignCommandRequest,
    D1CampaignRequestError,
    parse_d1_campaign_request,
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
from entropic_substrate.experiments.d1_config import D1LaunchProfile
from entropic_substrate.experiments.d1_confirmatory_matrix import D1GitProvenancePort
from entropic_substrate.experiments.d1_primitives import canonical_digest, canonical_json

__all__ = ["D1CapturePrepareCliService"]


class D1CampaignAuthorityLoaderPort(Protocol):
    def load_persisted(
        self,
        request: D1CampaignCommandRequest,
        *,
        manifest_path: Path,
    ) -> D1LoadedCampaignAuthority: ...


class D1CapturePreparationCoordinatorPort(Protocol):
    def prepare(self, **kwargs: object) -> D1PublishedCapturePreparation: ...


class D1CapturePreparationCoordinatorFactoryPort(Protocol):
    def build(
        self,
        request: D1CampaignCommandRequest,
    ) -> D1CapturePreparationCoordinatorPort: ...


def _request(invocation: D1CliInvocation, catalog: tuple[D1LaunchProfile, ...]):
    try:
        return parse_d1_campaign_request(
            canonical_json(dict(invocation.payload)),
            catalog=catalog,
        )
    except (D1CampaignRequestError, TypeError, ValueError) as error:
        raise D1CliInvalidEvidence(f"D1 capture prepare request was refused: {error}") from error


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
