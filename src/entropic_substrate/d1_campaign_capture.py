"""Completed D1 campaign to authoritative heterogeneous DVC capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Protocol

from entropic_substrate.d1_campaign_queue import D1CompletedCampaign
from entropic_substrate.evidence.d1_capture_aggregate import D1TerminalDefectInput
from entropic_substrate.evidence.d1_capture_bundle import (
    D1AttemptEvidence,
    D1CaptureInput,
)
from entropic_substrate.evidence.d1_capture_roster import (
    D1ExpectedRun,
    D1LaunchRoster,
)
from entropic_substrate.evidence.d1_capture_service import (
    D1HeterogeneousCapturePreparation,
    D1HeterogeneousCaptureReceipt,
)
from entropic_substrate.evidence.d1_lifecycle_values import D1ProcessState
from entropic_substrate.evidence.d1_process_manager import D1ProcessManager
from entropic_substrate.evidence.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
)
from entropic_substrate.experiments.d1_primitives import canonical_digest

_CAPTURE_TOKEN: Final = object()


@dataclass(frozen=True, slots=True)
class D1CampaignCaptureMaterial:
    roster: D1LaunchRoster
    complete_inputs: tuple[D1CaptureInput, ...]
    terminal_defect_inputs: tuple[D1TerminalDefectInput, ...]
    attempt_evidence: tuple[D1AttemptEvidence, ...]

    def __post_init__(self) -> None:
        if type(self.roster) is not D1LaunchRoster:
            raise TypeError("campaign capture material requires D1LaunchRoster")
        if any(
            type(value) is not tuple
            for value in (
                self.complete_inputs,
                self.terminal_defect_inputs,
                self.attempt_evidence,
            )
        ):
            raise TypeError("campaign capture lanes must be immutable tuples")


class D1CampaignEvidenceAssemblerPort(Protocol):
    def prepare(
        self,
        campaign: D1CompletedCampaign,
        roster: D1LaunchRoster,
    ) -> D1CampaignCaptureMaterial: ...


class D1CampaignCapturePort(Protocol):
    def prepare(self, material: D1CampaignCaptureMaterial) -> object: ...

    def finalize(self, preparation: object) -> D1HeterogeneousCaptureReceipt: ...


class D1HeterogeneousCaptureServicePort(Protocol):
    def prepare_heterogeneous(
        self,
        *,
        roster: D1LaunchRoster,
        complete_inputs: tuple[D1CaptureInput, ...],
        terminal_defect_inputs: tuple[D1TerminalDefectInput, ...],
        attempt_evidence: tuple[D1AttemptEvidence, ...] = (),
    ) -> D1HeterogeneousCapturePreparation: ...

    def finalize_heterogeneous(
        self, preparation: D1HeterogeneousCapturePreparation
    ) -> D1HeterogeneousCaptureReceipt: ...


class D1HeterogeneousCaptureAdapter:
    """Thin adapter over the already-validated D1 capture transaction."""

    def __init__(self, service: D1HeterogeneousCaptureServicePort) -> None:
        self._service = service

    def prepare(self, material: D1CampaignCaptureMaterial) -> D1HeterogeneousCapturePreparation:
        if type(material) is not D1CampaignCaptureMaterial:
            raise TypeError("heterogeneous capture adapter requires campaign material")
        return self._service.prepare_heterogeneous(
            roster=material.roster,
            complete_inputs=material.complete_inputs,
            terminal_defect_inputs=material.terminal_defect_inputs,
            attempt_evidence=material.attempt_evidence,
        )

    def finalize(
        self, preparation: D1HeterogeneousCapturePreparation
    ) -> D1HeterogeneousCaptureReceipt:
        return self._service.finalize_heterogeneous(preparation)


def _campaign_identity(campaign: D1CompletedCampaign) -> str:
    manager = campaign.process_manager
    if type(manager) is not D1ProcessManager:
        raise TypeError("completed campaign lacks nominal lifecycle authority")
    subjects = tuple(launch.plan.logical_run_id for launch in campaign.matrix.launches)
    if (
        campaign.design_digest != campaign.matrix.registered_design_digest
        or (
            bool(campaign.results)
            and campaign.attempt_ledger != campaign.results[-1].attempt_ledger
        )
        or manager.state is not D1ProcessState.OPERATOR_FINALIZED
        or manager.active_subject is not None
        or manager.finalized_subjects != subjects
    ):
        raise ValueError("completed campaign lifecycle identity differs from its matrix")
    prefix = manager.history.prefix
    return canonical_digest(
        {
            "attempt_ledger_digest": campaign.attempt_ledger.ledger_digest,
            "design_digest": campaign.design_digest,
            "finalized_subjects": list(subjects),
            "lifecycle_event_count": prefix.event_count,
            "lifecycle_prefix": prefix.prefix_sha256,
            "lifecycle_state": manager.state.value,
            "matrix_digest": campaign.matrix.matrix_digest,
            "schema_version": "d1-campaign-capture-identity-v1",
        }
    )


@dataclass(frozen=True, slots=True)
class D1PreparedCampaignCapture:
    campaign: D1CompletedCampaign
    material: D1CampaignCaptureMaterial
    preparation: object
    process_manager: D1ProcessManager
    campaign_digest: str
    lifecycle_prefix: str
    lifecycle_event_count: int
    _token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _CAPTURE_TOKEN:
            raise TypeError("D1PreparedCampaignCapture is factory-only")


@dataclass(frozen=True, slots=True)
class D1CapturedCampaign:
    campaign: D1CompletedCampaign
    capture: D1HeterogeneousCaptureReceipt
    process_manager: D1ProcessManager
    _token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _CAPTURE_TOKEN:
            raise TypeError("D1CapturedCampaign is factory-only")
        if self.process_manager.state.value != "captured":
            raise ValueError("captured campaign requires persisted CAPTURED lifecycle state")


def _expected_roster(campaign: D1CompletedCampaign) -> D1LaunchRoster:
    matrix = campaign.matrix
    if campaign.design_digest != matrix.registered_design_digest:
        raise ValueError("completed campaign and matrix design identities differ")
    expected: list[D1ExpectedRun] = []
    for launch in matrix.launches:
        plan = launch.plan
        code = plan.run.execution.code
        container = plan.run.execution.container
        data = plan.run.data
        if code is None or container is None or data is None:
            raise ValueError("campaign matrix launch lacks capture provenance")
        expected.append(
            D1ExpectedRun(
                logical_run_id=plan.logical_run_id,
                arm=plan.profile.arm,
                seed=plan.seed,
                pairing_id=plan.run.execution.pairing_id,
                family_digest=plan.profile.family.family_digest,
                construction_digest=plan.run.construction_digest,
                execution_digest=plan.run.execution_digest,
                aggregation_digest=plan.run.aggregation_digest,
                plan_digest=plan.plan_digest,
                code_commit=code.commit,
                repository_digest=code.repository_digest,
                image_digest=container.image_digest,
                dataset_digest=data.dataset_digest,
                split_digest=data.split_digest,
                sampler_construction_digest=data.sampler_construction_digest,
            )
        )
    preregistration = matrix.launches[0].plan.run.preregistration
    if preregistration is None:
        raise ValueError("campaign matrix lacks preregistration identity")
    return D1LaunchRoster(
        preregistration.path,
        preregistration.commit,
        expected[0].family_digest,
        tuple(expected),
        (),
    )


class D1CampaignCaptureService:
    """Capture only a complete attested campaign, then persist CAPTURED."""

    def __init__(
        self,
        *,
        assembler: D1CampaignEvidenceAssemblerPort,
        capture: D1CampaignCapturePort,
        automation: ActorIdentity,
    ) -> None:
        if automation.role is not ActorRole.AUTOMATION:
            raise ValueError("campaign capture actor must have automation role")
        self._assembler, self._capture, self._automation = assembler, capture, automation

    @staticmethod
    def expected_roster_from_matrix(campaign: D1CompletedCampaign) -> D1LaunchRoster:
        return _expected_roster(campaign)

    def expected_roster(self, campaign: D1CompletedCampaign) -> D1LaunchRoster:
        return _expected_roster(campaign)

    def prepare(self, campaign: D1CompletedCampaign) -> D1PreparedCampaignCapture:
        if type(campaign) is not D1CompletedCampaign:
            raise TypeError("campaign capture requires D1CompletedCampaign")
        identity = _campaign_identity(campaign)
        roster = _expected_roster(campaign)
        material = self._assembler.prepare(campaign, roster)
        if type(material) is not D1CampaignCaptureMaterial or material.roster != roster:
            raise ValueError("campaign evidence assembler roster differs from the matrix")
        preparation = self._capture.prepare(material)
        aggregate = getattr(preparation, "aggregate", None)
        if getattr(aggregate, "roster", None) != roster:
            raise ValueError("campaign capture preparation roster differs from the matrix")
        manager = campaign.process_manager
        prefix = manager.history.prefix
        return D1PreparedCampaignCapture(
            campaign,
            material,
            preparation,
            manager,
            identity,
            prefix.prefix_sha256,
            prefix.event_count,
            _token=_CAPTURE_TOKEN,
        )

    def finalize(self, prepared: D1PreparedCampaignCapture) -> D1CapturedCampaign:
        if type(prepared) is not D1PreparedCampaignCapture:
            raise TypeError("campaign capture finalize requires its prepared typestate")
        campaign = prepared.campaign
        manager = campaign.process_manager
        prefix = manager.history.prefix
        if (
            _campaign_identity(campaign) != prepared.campaign_digest
            or manager != prepared.process_manager
            or prefix.prefix_sha256 != prepared.lifecycle_prefix
            or prefix.event_count != prepared.lifecycle_event_count
        ):
            raise ValueError("prepared campaign or lifecycle identity differs at finalize")
        receipt = self._capture.finalize(prepared.preparation)
        roster = prepared.material.roster
        if receipt.aggregate.roster != roster:
            raise ValueError("campaign capture receipt roster differs from the matrix")
        if receipt.manifest.get("analysis_ready") is not True:
            raise ValueError("campaign capture receipt is not analysis-ready")
        artifact = ArtifactIdentity(
            f"results/captured.dvc@{receipt.manifest['pointer_commit']}",
            ContentHash("sha256", canonical_digest(receipt.manifest)),
        )
        captured = manager.capture(
            actor=self._automation,
            artifact=artifact,
            event_id=f"d1-campaign:{campaign.matrix.matrix_digest}:captured",
        )
        return D1CapturedCampaign(campaign, receipt, captured, _token=_CAPTURE_TOKEN)

    def capture(self, campaign: D1CompletedCampaign) -> D1CapturedCampaign:
        return self.finalize(self.prepare(campaign))
