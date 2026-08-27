"""Public D1 campaign-run service over fresh or persisted recovery."""

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
from entropic_substrate.d1_campaign_checkpoint_repository import (
    D1CampaignCheckpointRepository,
    D1CampaignCheckpointRepositoryError,
)
from entropic_substrate.d1_campaign_queue import (
    D1CampaignQueueHalt,
    D1CompletedCampaign,
)
from entropic_substrate.d1_campaign_recovery import (
    D1CampaignRecoveryError,
    D1CampaignRecoveryHalt,
    D1CampaignRecoveryStatus,
    D1HydratedCampaignRecovery,
    hydrate_d1_campaign_recovery,
)
from entropic_substrate.d1_campaign_recovery_queue import (
    D1RecoverableCampaignQueue,
    D1RecoveryAttemptControllerPort,
)
from entropic_substrate.d1_campaign_request import (
    D1CampaignCommandRequest,
    D1CampaignRequestError,
    D1FreshCampaignCursor,
    D1PersistedCampaignCursor,
    parse_d1_campaign_request,
)
from entropic_substrate.d1_cli import (
    D1CliCommand,
    D1CliInvalidEvidence,
    D1CliInvocation,
    D1CliOperatorGap,
)
from entropic_substrate.experiments.d1_config import D1LaunchProfile
from entropic_substrate.experiments.d1_confirmatory_matrix import D1GitProvenancePort
from entropic_substrate.experiments.d1_primitives import canonical_digest, canonical_json

__all__ = ["D1CampaignRunCliService"]


class D1CampaignAuthorityLoaderPort(Protocol):
    def load_fresh(
        self,
        request: D1CampaignCommandRequest,
        *,
        manifest_path: Path,
    ) -> D1LoadedCampaignAuthority: ...

    def load_persisted(
        self,
        request: D1CampaignCommandRequest,
        *,
        manifest_path: Path,
    ) -> D1LoadedCampaignAuthority: ...


class D1CampaignControllerFactoryPort(Protocol):
    def build(
        self,
        loaded: D1LoadedCampaignAuthority,
    ) -> D1RecoveryAttemptControllerPort: ...


def _parse(
    invocation: D1CliInvocation,
    catalog: tuple[D1LaunchProfile, ...],
) -> D1CampaignCommandRequest:
    try:
        return parse_d1_campaign_request(
            canonical_json(dict(invocation.payload)),
            catalog=catalog,
        )
    except (D1CampaignRequestError, TypeError, ValueError) as error:
        raise D1CliInvalidEvidence(f"D1 campaign-run request was refused: {error}") from error


def _result(
    campaign: D1CompletedCampaign,
    loaded: D1LoadedCampaignAuthority,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_size: int,
) -> Mapping[str, object]:
    manager = campaign.process_manager
    prefix = manager.history.prefix
    return {
        "campaign_request_digest": canonical_digest(loaded.request.manifest),
        "checkpoint_generation": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size,
        },
        "completed_cell_count": len(campaign.matrix.launches),
        "executed_cell_count": len(campaign.results),
        "lifecycle": {
            "event_count": prefix.event_count,
            "prefix_sha256": prefix.prefix_sha256,
            "state": manager.state.value,
        },
        "matrix_digest": campaign.matrix.matrix_digest,
        "recovered_prefix_count": campaign.recovered_prefix_count,
        "registered_design_digest": campaign.design_digest,
        "required_next_command": D1CliCommand.CAPTURE_PREPARE.value,
        "schema_version": "d1-campaign-run-result-v1",
        "status": "campaign_complete",
        "terminal_defect_run_ids": list(campaign.matrix.terminal_defect_run_ids),
    }


class D1CampaignRunCliService:
    """Bootstrap or recover, execute only the suffix, and prove final checkpoint."""

    def __init__(
        self,
        *,
        catalog: tuple[D1LaunchProfile, ...],
        authority_loader: D1CampaignAuthorityLoaderPort,
        git: D1GitProvenancePort,
        controller_factory: D1CampaignControllerFactoryPort,
    ) -> None:
        if (
            type(catalog) is not tuple
            or not catalog
            or any(type(item) is not D1LaunchProfile for item in catalog)
        ):
            raise ValueError("campaign-run CLI requires profile catalog")
        if not all(hasattr(authority_loader, name) for name in ("load_fresh", "load_persisted")):
            raise TypeError("campaign-run CLI requires authority loader")
        if any(not hasattr(git, name) for name in ("commit_time", "blob_sha256", "is_ancestor")):
            raise TypeError("campaign-run CLI requires Git provenance port")
        if not hasattr(controller_factory, "build"):
            raise TypeError("campaign-run CLI requires controller factory")
        self._catalog = catalog
        self._loader = authority_loader
        self._git = git
        self._factory = controller_factory

    def _load(
        self,
        request: D1CampaignCommandRequest,
        *,
        manifest_path: Path,
    ) -> D1LoadedCampaignAuthority:
        try:
            resolved = replace(
                request,
                locations=request.locations.resolve(manifest_path.parent),
            )
            if type(request.cursor) is D1FreshCampaignCursor:
                loaded = self._loader.load_fresh(request, manifest_path=manifest_path)
                if (
                    type(loaded.request.cursor) is not D1PersistedCampaignCursor
                    or loaded.request.cursor.sha256 != loaded.checkpoint_receipt.sha256
                    or loaded.request != replace(resolved, cursor=loaded.request.cursor)
                ):
                    raise D1CliInvalidEvidence(
                        "D1 fresh campaign loader returned substituted authority"
                    )
                return loaded
            loaded = self._loader.load_persisted(request, manifest_path=manifest_path)
            if (
                type(loaded.request.cursor) is not D1PersistedCampaignCursor
                or loaded.request.cursor.sha256 != loaded.checkpoint_receipt.sha256
                or loaded.request != replace(resolved, cursor=loaded.request.cursor)
            ):
                raise D1CliInvalidEvidence(
                    "D1 persisted campaign loader returned substituted authority"
                )
            return loaded
        except D1CliInvalidEvidence:
            raise
        except D1CampaignAuthorityLoaderError as error:
            raise D1CliInvalidEvidence(str(error)) from error
        except (TypeError, ValueError) as error:
            raise D1CliInvalidEvidence(f"D1 campaign authority was refused: {error}") from error

    def _execute(self, loaded: D1LoadedCampaignAuthority) -> D1CompletedCampaign:
        recovery = loaded.recovery
        if type(recovery) is D1CampaignRecoveryHalt:
            raise D1CliOperatorGap(f"D1 campaign recovery requires adjudication: {recovery.reason}")
        if type(recovery) is not D1HydratedCampaignRecovery:
            raise D1CliInvalidEvidence("D1 campaign loader returned unsupported recovery")
        if recovery.status is D1CampaignRecoveryStatus.COMPLETE:
            try:
                return D1CampaignAuthorityLoader.completed_campaign(loaded, git=self._git)
            except D1CampaignAuthorityLoaderError as error:
                raise D1CliInvalidEvidence(str(error)) from error
        try:
            controller = self._factory.build(loaded)
            return D1RecoverableCampaignQueue(
                design=loaded.design,
                intents=loaded.request.intents,
                controller=controller,
                git=self._git,
                repository=D1CampaignCheckpointRepository(loaded.request.locations.checkpoint_path),
            ).run(
                recovery,
                generation=loaded.checkpoint_receipt,
            )
        except D1CliOperatorGap:
            raise
        except D1CampaignQueueHalt as error:
            raise D1CliOperatorGap(str(error)) from error
        except (TypeError, ValueError) as error:
            raise D1CliInvalidEvidence(f"D1 campaign execution was refused: {error}") from error

    def execute(self, invocation: D1CliInvocation) -> Mapping[str, object]:
        if type(invocation) is not D1CliInvocation:
            raise TypeError("campaign-run CLI requires D1CliInvocation")
        if invocation.command is not D1CliCommand.CAMPAIGN_RUN:
            raise D1CliInvalidEvidence("D1 campaign-run received the wrong command")
        request = _parse(invocation, self._catalog)
        loaded = self._load(request, manifest_path=invocation.manifest_path)
        campaign = self._execute(loaded)
        repository = D1CampaignCheckpointRepository(loaded.request.locations.checkpoint_path)
        try:
            checkpoint, receipt = repository.load()
            recovery = hydrate_d1_campaign_recovery(
                design=loaded.design,
                intents=loaded.request.intents,
                checkpoint=checkpoint,
                process_manager=campaign.process_manager,
            )
        except (
            D1CampaignCheckpointRepositoryError,
            D1CampaignRecoveryError,
            TypeError,
            ValueError,
        ) as error:
            raise D1CliInvalidEvidence(
                f"D1 final campaign checkpoint was refused: {error}"
            ) from error
        prefix = campaign.process_manager.history.prefix
        if (
            type(recovery) is not D1HydratedCampaignRecovery
            or recovery.status is not D1CampaignRecoveryStatus.COMPLETE
            or checkpoint.attempt_ledger != campaign.attempt_ledger
            or checkpoint.lifecycle_prefix != prefix.prefix_sha256
            or checkpoint.lifecycle_event_count != prefix.event_count
        ):
            raise D1CliInvalidEvidence("D1 final campaign checkpoint differs from completion")
        return _result(
            campaign,
            loaded,
            checkpoint_path=receipt.path,
            checkpoint_sha256=receipt.sha256,
            checkpoint_size=receipt.size_bytes,
        )
