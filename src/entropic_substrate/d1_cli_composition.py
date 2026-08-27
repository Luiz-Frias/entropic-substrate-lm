"""Production composition root for the public D1 commands."""

from __future__ import annotations

from dataclasses import dataclass, replace

from entropic_substrate.d1_artifact_snapshot import D1ArtifactSnapshotError
from entropic_substrate.d1_campaign_authority_loader import D1CampaignAuthorityLoader
from entropic_substrate.d1_campaign_capture import D1HeterogeneousCaptureAdapter
from entropic_substrate.d1_campaign_request import (
    D1CampaignCommandRequest,
    D1CampaignRequestError,
    parse_d1_campaign_request,
)
from entropic_substrate.d1_capture_cli_services import (
    D1CaptureFinalizeCliService,
    D1CapturePrepareCliService,
)
from entropic_substrate.d1_capture_finalize_coordinator import (
    D1CampaignCaptureFinalizationCoordinator,
)
from entropic_substrate.d1_capture_finalize_request import (
    D1CaptureFinalizeRequestError,
    parse_d1_capture_finalize_request,
)
from entropic_substrate.d1_capture_prepare_coordinator import (
    D1CampaignCapturePreparationCoordinator,
)
from entropic_substrate.d1_cli import (
    D1CliApplication,
    D1CliCommand,
    D1CliCommandService,
    D1CliInvalidEvidence,
    D1CliInvocation,
    D1CliOperatorGap,
    D1CliServiceUnavailable,
)
from entropic_substrate.d1_git_provenance_adapter import (
    D1GitProvenanceAdapterError,
    SubprocessD1GitProvenance,
)
from entropic_substrate.d1_inspect_service import D1InspectCliService
from entropic_substrate.d1_observability import AuthoritativeJournalOnlyObserver
from entropic_substrate.d1_trainer_cli_services import (
    D1FreshTrainerCliService,
    D1ResumeTrainerCliService,
)
from entropic_substrate.evidence.d1_capture_filesystem import LocalD1EvidenceExportSource
from entropic_substrate.evidence.d1_capture_ports import D1CaptureTransportError
from entropic_substrate.evidence.d1_capture_service import D1CaptureService
from entropic_substrate.evidence.d1_capture_subprocess import (
    SubprocessD1CloudObjectVerifier,
    SubprocessD1DvcAdapter,
    SubprocessD1GitPointerStatusVerifier,
)
from entropic_substrate.evidence.lifecycle import ActorIdentity, ActorRole
from entropic_substrate.experiments.d1_config import D1LaunchProfile, load_d1_profile_catalog
from entropic_substrate.experiments.d1_primitives import canonical_digest, canonical_json

__all__ = ["build_default_d1_cli_application"]


@dataclass(frozen=True, slots=True)
class _LazyTrainerCliService:
    command: D1CliCommand

    def execute(self, invocation: D1CliInvocation) -> dict[str, object]:
        if invocation.command is not self.command:
            raise ValueError("lazy D1 trainer service received the wrong command")
        catalog = (*load_d1_profile_catalog(), *load_d1_profile_catalog(tiny=True))
        if self.command is D1CliCommand.RUN_FRESH:
            return dict(D1FreshTrainerCliService(catalog=catalog).execute(invocation))
        if self.command is D1CliCommand.RUN_RESUME:
            return dict(
                D1ResumeTrainerCliService(
                    catalog=catalog,
                    observer=AuthoritativeJournalOnlyObserver(),
                ).execute(invocation)
            )
        raise ValueError("lazy D1 trainer service command is unsupported")


def _automation(request: D1CampaignCommandRequest) -> ActorIdentity:
    digest = canonical_digest(request.manifest)
    return ActorIdentity(
        "d1-capture-automation",
        f"request-{digest[:24]}",
        ActorRole.AUTOMATION,
    )


def _capture_port(request: D1CampaignCommandRequest) -> D1HeterogeneousCaptureAdapter:
    locations = request.locations
    service = D1CaptureService(
        repo_root=locations.repo_root,
        evidence_source=LocalD1EvidenceExportSource(locations.artifact_root),
        dvc=SubprocessD1DvcAdapter(repo_root=locations.repo_root),
        cloud=SubprocessD1CloudObjectVerifier(repo_root=locations.repo_root),
        git=SubprocessD1GitPointerStatusVerifier(repo_root=locations.repo_root),
    )
    return D1HeterogeneousCaptureAdapter(service)


class _PrepareCoordinatorFactory:
    def build(
        self,
        request: D1CampaignCommandRequest,
    ) -> D1CampaignCapturePreparationCoordinator:
        return D1CampaignCapturePreparationCoordinator(
            capture=_capture_port(request),
            automation=_automation(request),
        )


class _FinalizeCoordinatorFactory:
    def build(
        self,
        request: D1CampaignCommandRequest,
    ) -> D1CampaignCaptureFinalizationCoordinator:
        return D1CampaignCaptureFinalizationCoordinator(
            capture=_capture_port(request),
            automation=_automation(request),
        )


def _capture_request(
    invocation: D1CliInvocation,
    *,
    catalog: tuple[D1LaunchProfile, ...],
) -> D1CampaignCommandRequest:
    payload = canonical_json(dict(invocation.payload))
    try:
        if invocation.command is D1CliCommand.CAPTURE_PREPARE:
            return parse_d1_campaign_request(payload, catalog=catalog)
        if invocation.command is D1CliCommand.CAPTURE_FINALIZE:
            return parse_d1_capture_finalize_request(
                payload,
                catalog=catalog,
            ).campaign
    except (D1CampaignRequestError, D1CaptureFinalizeRequestError) as error:
        raise D1CliInvalidEvidence(f"D1 capture request was refused: {error}") from error
    raise D1CliInvalidEvidence("lazy D1 capture service received the wrong command")


@dataclass(frozen=True, slots=True)
class _LazyCaptureCliService:
    command: D1CliCommand

    def execute(self, invocation: D1CliInvocation) -> dict[str, object]:
        if invocation.command is not self.command:
            raise D1CliInvalidEvidence("lazy D1 capture service received the wrong command")
        catalog = load_d1_profile_catalog()
        request = _capture_request(invocation, catalog=catalog)
        try:
            resolved = replace(
                request,
                locations=request.locations.resolve(invocation.manifest_path.parent),
            )
            git = SubprocessD1GitProvenance(repo_root=resolved.locations.repo_root)
        except (
            D1ArtifactSnapshotError,
            D1CaptureTransportError,
            D1GitProvenanceAdapterError,
            TypeError,
            ValueError,
        ) as error:
            raise D1CliOperatorGap(f"D1 capture production root is unavailable: {error}") from error
        loader = D1CampaignAuthorityLoader(catalog=catalog)
        if self.command is D1CliCommand.CAPTURE_PREPARE:
            service: D1CliCommandService = D1CapturePrepareCliService(
                catalog=catalog,
                authority_loader=loader,
                git=git,
                coordinator_factory=_PrepareCoordinatorFactory(),
            )
        else:
            service = D1CaptureFinalizeCliService(
                catalog=catalog,
                authority_loader=loader,
                git=git,
                coordinator_factory=_FinalizeCoordinatorFactory(),
            )
        return dict(service.execute(invocation))


def build_default_d1_cli_application() -> D1CliApplication:
    """Compose connected local services and loud gaps for unfinished commands."""
    services: dict[D1CliCommand, D1CliCommandService] = {
        command: D1CliServiceUnavailable(
            f"{command.value} composition service is unavailable; "
            "the production adapter has not been connected"
        )
        for command in D1CliCommand
    }
    services[D1CliCommand.RUN_FRESH] = _LazyTrainerCliService(D1CliCommand.RUN_FRESH)
    services[D1CliCommand.RUN_RESUME] = _LazyTrainerCliService(D1CliCommand.RUN_RESUME)
    services[D1CliCommand.CAPTURE_PREPARE] = _LazyCaptureCliService(D1CliCommand.CAPTURE_PREPARE)
    services[D1CliCommand.CAPTURE_FINALIZE] = _LazyCaptureCliService(D1CliCommand.CAPTURE_FINALIZE)
    services[D1CliCommand.INSPECT] = D1InspectCliService()
    return D1CliApplication(services)
