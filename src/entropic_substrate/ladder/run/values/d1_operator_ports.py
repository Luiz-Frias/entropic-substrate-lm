"""Ports shared by the D1 cloud-build and accelerator operator adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from entropic_substrate.__legacy.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    D1StagedTrainerRequestReceipt,
    D1TrainerStartReceipt,
)
from entropic_substrate.kernel.evidence_class.resource_provenance import RunStartProvenance
from entropic_substrate.ladder.capture.resource.resource_billing import (
    ExecutionSegmentTermination,
    ProviderBillingObservation,
    TerminalProviderBillingEvidence,
)
from entropic_substrate.ladder.run.campaign.d1_operator_values import (
    D1ImageBuildRequest,
    D1ImmutableImage,
    D1OperatorLaunchIntent,
    D1PodPlacement,
    D1RuntimeObservation,
    require_d1_pod_id,
)
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import D1TrainerRequest


class D1OperatorTransportError(RuntimeError):
    """Fail-loud external command or provider response defect."""

    def __init__(self, *, operation: str, subject: str, detail: str) -> None:
        self.operation = operation
        self.subject = subject
        self.context = MappingProxyType({"operation": operation, "subject": subject})
        super().__init__(f"D1 operator {operation} for {subject}: {detail}")


class D1PodDeletionAcknowledgedError(D1OperatorTransportError):
    """Provider deleted the pod, but complete termination evidence could not be minted."""

    deletion_acknowledged = True

    def __init__(self, *, pod_id: str, detail: str) -> None:
        pod = require_d1_pod_id(pod_id)
        if (
            type(detail) is not str
            or not detail
            or detail.strip() != detail
            or len(detail) > 512
            or any(ord(character) < 32 for character in detail)
        ):
            raise ValueError("acknowledged deletion detail must be bounded safe text")
        self.pod_id = pod
        super().__init__(
            operation="terminate_pod",
            subject=pod,
            detail=f"pod deletion was acknowledged; {detail}",
        )


@dataclass(frozen=True, slots=True)
class D1CommandResult:
    """Captured text streams from one shell-free argv invocation."""

    stdout: str
    stderr: str
    returncode: int = 0

    def __post_init__(self) -> None:
        if type(self.stdout) is not str or type(self.stderr) is not str:
            raise TypeError("D1 command result streams must be strings")
        if type(self.returncode) is not int:
            raise TypeError("D1 command result returncode must be an integer")


class D1CommandRunner(Protocol):
    """Injected process boundary; implementations must execute argv without a shell."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        stdin: str | None,
    ) -> D1CommandResult: ...


class D1BuildCloudPort(Protocol):
    def build_and_push(self, request: D1ImageBuildRequest) -> D1ImmutableImage: ...


@dataclass(frozen=True, slots=True)
class D1RunPodAuthReceipt:
    """Non-secret identity proving the configured CLI session was authenticated."""

    account_id: str

    def __post_init__(self) -> None:
        if type(self.account_id) is not str or not self.account_id.strip():
            raise ValueError("RunPod authentication account_id must be nonempty")


class D1RunPodPort(Protocol):
    def verify_authenticated(self) -> D1RunPodAuthReceipt: ...

    def create(self, intent: D1OperatorLaunchIntent) -> D1PodPlacement: ...

    def cancel(self, pod_id: str) -> datetime: ...

    def terminate(
        self,
        pod_id: str,
        *,
        start: RunStartProvenance,
        trainer_ended_at: datetime,
    ) -> ExecutionSegmentTermination: ...

    def billing(
        self, pod_id: str, *, start: RunStartProvenance
    ) -> TerminalProviderBillingEvidence: ...


class D1ProviderBillingObservationPort(Protocol):
    """One provider query; settlement policy remains above this seam."""

    def observe_billing(
        self, pod_id: str, *, start: RunStartProvenance
    ) -> ProviderBillingObservation: ...


class D1TrainerEvidencePort(Protocol):
    """Transport canonical inputs and return exported bytes, never loose metrics."""

    def bootstrap_runtime(self, pod_id: str, attempt_sha256: str) -> D1RuntimeObservation: ...

    def stage_inputs(
        self, pod_id: str, *, request: D1TrainerRequest
    ) -> D1StagedTrainerRequestReceipt: ...

    def start_trainer(self, pod_id: str) -> D1TrainerStartReceipt: ...

    def await_exported_handoff(self, pod_id: str) -> D1ExportedTrainerHandoff: ...
