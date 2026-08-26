"""Ports shared by the D1 cloud-build and accelerator operator adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from entropic_substrate.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    D1TrainerInputEnvelope,
)
from entropic_substrate.d1_operator_values import (
    D1ImageBuildRequest,
    D1ImmutableImage,
    D1OperatorLaunchIntent,
    D1PodPlacement,
    D1RuntimeObservation,
)
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.evidence.resource_billing import (
    BillingGap,
    ExecutionSegmentTermination,
    ProviderBilling,
)
from entropic_substrate.evidence.resource_provenance import RunStartProvenance


class D1OperatorTransportError(RuntimeError):
    """Fail-loud external command or provider response defect."""

    def __init__(self, *, operation: str, subject: str, detail: str) -> None:
        self.operation = operation
        self.subject = subject
        self.context = MappingProxyType({"operation": operation, "subject": subject})
        super().__init__(f"D1 operator {operation} for {subject}: {detail}")


@dataclass(frozen=True, slots=True)
class D1CommandResult:
    """Captured text streams from one shell-free argv invocation."""

    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        if type(self.stdout) is not str or type(self.stderr) is not str:
            raise TypeError("D1 command result streams must be strings")


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
    ) -> ProviderBilling | BillingGap: ...


class D1TrainerEvidencePort(Protocol):
    """Transport canonical inputs and return exported bytes, never loose metrics."""

    def observe_runtime(self, pod_id: str) -> D1RuntimeObservation: ...

    def stage_inputs(self, pod_id: str, envelope: D1TrainerInputEnvelope) -> ArtifactIdentity: ...

    def await_exported_handoff(self, pod_id: str) -> D1ExportedTrainerHandoff: ...
