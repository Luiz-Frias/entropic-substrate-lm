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
from entropic_substrate.evidence.d1_capture_service import D1HeterogeneousCaptureReceipt
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
    def capture(self, material: D1CampaignCaptureMaterial) -> D1HeterogeneousCaptureReceipt: ...


class D1HeterogeneousCaptureServicePort(Protocol):
    def capture_heterogeneous(
        self,
        *,
        roster: D1LaunchRoster,
        complete_inputs: tuple[D1CaptureInput, ...],
        terminal_defect_inputs: tuple[D1TerminalDefectInput, ...],
        attempt_evidence: tuple[D1AttemptEvidence, ...] = (),
    ) -> D1HeterogeneousCaptureReceipt: ...


class D1HeterogeneousCaptureAdapter:
    """Thin adapter over the already-validated D1 capture transaction."""

    def __init__(self, service: D1HeterogeneousCaptureServicePort) -> None:
        self._service = service

    def capture(self, material: D1CampaignCaptureMaterial) -> D1HeterogeneousCaptureReceipt:
        if type(material) is not D1CampaignCaptureMaterial:
            raise TypeError("heterogeneous capture adapter requires campaign material")
        return self._service.capture_heterogeneous(
            roster=material.roster,
            complete_inputs=material.complete_inputs,
            terminal_defect_inputs=material.terminal_defect_inputs,
            attempt_evidence=material.attempt_evidence,
        )


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

    def capture(self, campaign: D1CompletedCampaign) -> D1CapturedCampaign:
        if type(campaign) is not D1CompletedCampaign:
            raise TypeError("campaign capture requires D1CompletedCampaign")
        roster = _expected_roster(campaign)
        material = self._assembler.prepare(campaign, roster)
        if type(material) is not D1CampaignCaptureMaterial or material.roster != roster:
            raise ValueError("campaign evidence assembler roster differs from the matrix")
        receipt = self._capture.capture(material)
        if receipt.aggregate.roster != roster:
            raise ValueError("campaign capture receipt roster differs from the matrix")
        if receipt.manifest.get("analysis_ready") is not True:
            raise ValueError("campaign capture receipt is not analysis-ready")
        manager = campaign.process_manager
        if type(manager) is not D1ProcessManager:
            raise TypeError("campaign final result lacks D1ProcessManager")
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
