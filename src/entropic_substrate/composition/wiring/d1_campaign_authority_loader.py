"""Rehydrate persisted D1 campaign authorities without provider access."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NoReturn, cast

from entropic_substrate.kernel.artifacts import (
    D1ArtifactSnapshotError,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.identity.d1_primitives import canonical_json
from entropic_substrate.ladder.capture.ledger.d1_lifecycle_repository import (
    D1LifecycleCorruptionError,
    D1LifecycleRepository,
    D1LifecycleRepositoryGap,
)
from entropic_substrate.ladder.capture.ledger.d1_process_manager import D1ProcessManager
from entropic_substrate.ladder.register.d1_config import D1LaunchProfile
from entropic_substrate.ladder.register.d1_confirmatory_matrix import D1GitProvenancePort
from entropic_substrate.ladder.register.d1_family import D1ExperimentFamily
from entropic_substrate.ladder.register.d1_launch_attempts import D1LaunchAttemptLedger
from entropic_substrate.ladder.register.d1_registered_design import D1RegisteredDesign
from entropic_substrate.ladder.register.d1_registration_codec import (
    D1PreregistrationByteEvidence,
    D1RegistrationCodecError,
    hydrate_registered_design,
)
from entropic_substrate.ladder.run.campaign.d1_campaign_bootstrap import (
    D1CampaignBootstrapError,
    adopt_d1_finalized_cell_after_crash,
    bootstrap_d1_ready_checkpoint,
)
from entropic_substrate.ladder.run.campaign.d1_campaign_checkpoint import D1CampaignCheckpoint
from entropic_substrate.ladder.run.campaign.d1_campaign_checkpoint_repository import (
    D1CampaignCheckpointReceipt,
    D1CampaignCheckpointRepository,
    D1CampaignCheckpointRepositoryError,
)
from entropic_substrate.ladder.run.campaign.d1_campaign_queue import (
    D1CampaignQueueHalt,
    D1CompletedCampaign,
)
from entropic_substrate.ladder.run.campaign.d1_campaign_recovery import (
    D1CampaignRecoveryError,
    D1CampaignRecoveryHalt,
    D1CampaignRecoveryStatus,
    D1HydratedCampaignRecovery,
    hydrate_d1_campaign_recovery,
)
from entropic_substrate.ladder.run.campaign.d1_campaign_recovery_queue import (
    D1RecoverableCampaignQueue,
)
from entropic_substrate.ladder.run.campaign.d1_campaign_request import (
    D1CampaignCommandRequest,
    D1FreshCampaignCursor,
    D1PersistedCampaignCursor,
)
from entropic_substrate.ladder.run.campaign.d1_operator_controller import D1OperatorRunResult
from entropic_substrate.ladder.run.campaign.d1_operator_values import D1OperatorLaunchIntent
from entropic_substrate.ladder.run.closure.d1_finalized_cell_custody_repository import (
    D1FinalizedCellCustodyRepository,
    D1FinalizedCellCustodyRepositoryError,
)

__all__ = [
    "D1CampaignAuthorityLoader",
    "D1CampaignAuthorityLoaderError",
    "D1LoadedCampaignAuthority",
]

_MAX_PREREGISTRATION_BYTES = 16 * 1024 * 1024


class D1CampaignAuthorityLoaderError(RuntimeError):
    """A persisted campaign authority is absent, substituted, or contradictory."""

    def __init__(self, *, stage: str, detail: str) -> None:
        self.stage = stage
        self.context = {"stage": stage}
        super().__init__(f"D1 campaign authority {stage}: {detail}")


def _fail(stage: str, detail: str, error: Exception | None = None) -> NoReturn:
    failure = D1CampaignAuthorityLoaderError(stage=stage, detail=detail)
    if error is None:
        raise failure
    raise failure from error


@dataclass(frozen=True, slots=True)
class D1LoadedCampaignAuthority:
    request: D1CampaignCommandRequest
    design: D1RegisteredDesign
    process_manager: D1ProcessManager
    checkpoint: D1CampaignCheckpoint
    checkpoint_receipt: D1CampaignCheckpointReceipt
    recovery: D1HydratedCampaignRecovery | D1CampaignRecoveryHalt

    def __post_init__(self) -> None:
        if type(self.request) is not D1CampaignCommandRequest:
            raise TypeError("loaded campaign authority requires request")
        if type(self.design) is not D1RegisteredDesign:
            raise TypeError("loaded campaign authority requires design")
        if type(self.process_manager) is not D1ProcessManager:
            raise TypeError("loaded campaign authority requires process manager")
        if type(self.checkpoint) is not D1CampaignCheckpoint:
            raise TypeError("loaded campaign authority requires checkpoint")
        if type(self.checkpoint_receipt) is not D1CampaignCheckpointReceipt:
            raise TypeError("loaded campaign authority requires checkpoint receipt")
        if type(self.recovery) not in (D1HydratedCampaignRecovery, D1CampaignRecoveryHalt):
            raise TypeError("loaded campaign authority requires recovery typestate")


class D1CampaignAuthorityLoader:
    """Descriptor-read registration plus lifecycle/checkpoint recovery hydration."""

    def __init__(self, *, catalog: tuple[D1LaunchProfile, ...]) -> None:
        if (
            type(catalog) is not tuple
            or not catalog
            or any(type(item) is not D1LaunchProfile for item in catalog)
        ):
            raise ValueError("campaign authority loader requires a profile catalog")
        self._catalog = catalog

    def _family(self, family_digest: str) -> D1ExperimentFamily:
        candidates: dict[str, D1ExperimentFamily] = {}
        for profile in self._catalog:
            family = profile.family
            prior = candidates.get(family.family_digest)
            if prior is not None and prior != family:
                _fail("family", "installed families share a digest but differ")
            candidates[family.family_digest] = family
        selected = candidates.get(family_digest)
        if selected is None:
            _fail("family", "registered family is not installed")
        return selected

    @staticmethod
    def _manifest_path(value: Path) -> Path:
        if not isinstance(cast(object, value), Path):
            raise TypeError("campaign authority manifest_path must be a Path")
        path = Path(os.path.abspath(value))
        if path.is_symlink() or not path.is_file():
            _fail("manifest", "manifest must remain a regular non-symlink file")
        return path

    def _design(
        self,
        request: D1CampaignCommandRequest,
    ) -> D1RegisteredDesign:
        declaration = request.design
        locations = request.locations
        try:
            _path, payload, _sha256 = read_d1_artifact_bytes(
                locations.repo_root,
                declaration.preregistration.path,
                label="D1 preregistration",
                max_bytes=_MAX_PREREGISTRATION_BYTES,
            )
            evidence = D1PreregistrationByteEvidence.from_bytes(payload)
            return hydrate_registered_design(
                canonical_json(declaration.manifest),
                expected_design_digest=declaration.design_digest,
                family=self._family(declaration.family_identity.family_digest),
                decision_contract=request.decision_contract,
                preregistration_evidence=evidence,
            )
        except (
            D1ArtifactSnapshotError,
            D1RegistrationCodecError,
            TypeError,
            ValueError,
        ) as error:
            _fail("preregistration", str(error), error)

    @staticmethod
    def _process(request: D1CampaignCommandRequest) -> D1ProcessManager:
        try:
            loaded = D1ProcessManager.load(D1LifecycleRepository(request.locations.lifecycle_path))
        except D1LifecycleCorruptionError as error:
            _fail("lifecycle", str(error), error)
        if isinstance(loaded, D1LifecycleRepositoryGap):
            _fail("lifecycle", f"repository is {loaded.code.value}")
        return loaded

    def load_persisted(
        self,
        request: D1CampaignCommandRequest,
        *,
        manifest_path: Path,
    ) -> D1LoadedCampaignAuthority:
        if type(request) is not D1CampaignCommandRequest:
            raise TypeError("campaign authority loader requires campaign request")
        if type(request.cursor) is not D1PersistedCampaignCursor:
            _fail("cursor", "persisted campaign authority requires persisted cursor")
        manifest = self._manifest_path(manifest_path)
        try:
            resolved = replace(request, locations=request.locations.resolve(manifest.parent))
        except (TypeError, ValueError) as error:
            _fail("locations", str(error), error)
        design = self._design(resolved)
        process_manager = self._process(resolved)
        repository = D1CampaignCheckpointRepository(resolved.locations.checkpoint_path)
        try:
            checkpoint, receipt = repository.load(expected_sha256=request.cursor.sha256)
        except (D1CampaignCheckpointRepositoryError, TypeError, ValueError) as error:
            _fail("checkpoint", str(error), error)
        expected_subjects = tuple(cell.logical_run_id for cell in checkpoint.completed_cells)
        observed_subjects = process_manager.finalized_subjects
        if checkpoint.next_index < len(resolved.intents) and observed_subjects == (
            *expected_subjects,
            resolved.intents[checkpoint.next_index].plan.logical_run_id,
        ):
            custody = D1FinalizedCellCustodyRepository(
                resolved.locations.finalized_cell_custody_root
            )
            try:
                candidates = custody.discover(
                    exclude=tuple(
                        cell.finalized_cell_artifact for cell in checkpoint.completed_cells
                    )
                )
                checkpoint, receipt = adopt_d1_finalized_cell_after_crash(
                    design=design,
                    intent=resolved.intents[checkpoint.next_index],
                    checkpoint=checkpoint,
                    process_manager=process_manager,
                    candidates=candidates,
                    repository=repository,
                    generation=receipt,
                )
                resolved = replace(
                    resolved,
                    cursor=D1PersistedCampaignCursor(receipt.sha256),
                )
            except (
                D1CampaignBootstrapError,
                D1FinalizedCellCustodyRepositoryError,
                TypeError,
                ValueError,
            ) as error:
                _fail("adoption", str(error), error)
            finally:
                custody.close()
        try:
            recovery = hydrate_d1_campaign_recovery(
                design=design,
                intents=resolved.intents,
                checkpoint=checkpoint,
                process_manager=process_manager,
            )
        except (D1CampaignRecoveryError, TypeError, ValueError) as error:
            _fail("recovery", str(error), error)
        return D1LoadedCampaignAuthority(
            resolved,
            design,
            process_manager,
            checkpoint,
            receipt,
            recovery,
        )

    def load_fresh(
        self,
        request: D1CampaignCommandRequest,
        *,
        manifest_path: Path,
    ) -> D1LoadedCampaignAuthority:
        """Persist the sole empty READY checkpoint before any provider call."""
        if type(request) is not D1CampaignCommandRequest:
            raise TypeError("campaign authority loader requires campaign request")
        if type(request.cursor) is not D1FreshCampaignCursor:
            _fail("cursor", "fresh campaign authority requires fresh cursor")
        manifest = self._manifest_path(manifest_path)
        try:
            resolved = replace(request, locations=request.locations.resolve(manifest.parent))
        except (TypeError, ValueError) as error:
            _fail("locations", str(error), error)
        design = self._design(resolved)
        process_manager = self._process(resolved)
        path = resolved.locations.checkpoint_path
        if path.exists() or path.is_symlink():
            _fail("checkpoint", "fresh campaign checkpoint path is already occupied")
        try:
            repository = D1CampaignCheckpointRepository(path)
            checkpoint, receipt = bootstrap_d1_ready_checkpoint(
                design=design,
                process_manager=process_manager,
                attempt_ledger=D1LaunchAttemptLedger(),
                repository=repository,
            )
            persisted = replace(
                resolved,
                cursor=D1PersistedCampaignCursor(receipt.sha256),
            )
            recovery = hydrate_d1_campaign_recovery(
                design=design,
                intents=persisted.intents,
                checkpoint=checkpoint,
                process_manager=process_manager,
            )
        except (
            D1CampaignBootstrapError,
            D1CampaignCheckpointRepositoryError,
            D1CampaignRecoveryError,
            TypeError,
            ValueError,
        ) as error:
            _fail("checkpoint", str(error), error)
        return D1LoadedCampaignAuthority(
            persisted,
            design,
            process_manager,
            checkpoint,
            receipt,
            recovery,
        )

    @staticmethod
    def completed_campaign(
        loaded: D1LoadedCampaignAuthority,
        *,
        git: D1GitProvenancePort,
    ) -> D1CompletedCampaign:
        """Re-attest a COMPLETE recovery without exposing a provider method."""
        if type(loaded) is not D1LoadedCampaignAuthority:
            raise TypeError("completed campaign reconstruction requires loaded authority")
        recovery = loaded.recovery
        if (
            type(recovery) is not D1HydratedCampaignRecovery
            or recovery.status is not D1CampaignRecoveryStatus.COMPLETE
        ):
            _fail("complete", "campaign recovery is not complete")

        @dataclass
        class _NoProviderController:
            process_manager: D1ProcessManager
            attempt_ledger: D1LaunchAttemptLedger

            def execute_fresh(self, intent: D1OperatorLaunchIntent) -> D1OperatorRunResult:
                del intent
                _fail("complete", "complete recovery attempted a provider call")

        try:
            return D1RecoverableCampaignQueue(
                design=loaded.design,
                intents=loaded.request.intents,
                controller=_NoProviderController(
                    loaded.process_manager,
                    loaded.checkpoint.attempt_ledger,
                ),
                git=git,
                repository=D1CampaignCheckpointRepository(loaded.request.locations.checkpoint_path),
            ).run(
                recovery,
                generation=loaded.checkpoint_receipt,
            )
        except (D1CampaignQueueHalt, TypeError, ValueError) as error:
            _fail("complete", str(error), error)
