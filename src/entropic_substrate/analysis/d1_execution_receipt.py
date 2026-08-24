"""Isolation port and deterministic execution receipt for D1 analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from entropic_substrate.analysis.d1_capture import D1AdmittedAnalysisCapture
from entropic_substrate.analysis.d1_machine_analysis import (
    D1AliasedMachineAnalysis,
    D1ArmAliasMap,
)
from entropic_substrate.evidence.d1_capture_artifacts import D1DvcBindingState
from entropic_substrate.experiments.d1_primitives import canonical_digest, require_sha256

__all__ = [
    "D1AnalysisExecutionReceipt",
    "D1AnalysisImplementationIdentity",
    "D1AnalysisRunnerPort",
    "D1AnalysisRunnerRequest",
    "D1AnalysisRunnerResult",
    "D1ExecutorKind",
    "D1ExternalIsolationAttestation",
    "D1InProcessIsolationRunnerFake",
    "D1IsolationRequest",
    "issue_d1_analysis_receipt",
]

_SCHEMA = "d1-analysis-execution-receipt-v1"
_COMPLETE_READER = "entropic_substrate.telemetry.read_d1_run_evidence"
_DEFECT_READER = "entropic_substrate.telemetry.read_d1_terminal_defect"


def _sha256(value: str, *, label: str) -> None:
    try:
        require_sha256(value, label=label)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest") from error


def _commit(value: str) -> None:
    if (
        type(value) is not str
        or not 7 <= len(value) <= 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("analysis code_commit must be lowercase hexadecimal")


@dataclass(frozen=True, slots=True)
class D1IsolationRequest:
    network_disabled: bool = True
    filesystem_read_only: bool = True
    live_tracking_blocked: bool = True

    def __post_init__(self) -> None:
        if not (
            self.network_disabled is True
            and self.filesystem_read_only is True
            and self.live_tracking_blocked is True
        ):
            raise ValueError(
                "D1 analysis requires network disabled, filesystem read-only, and live tracking blocked"
            )

    @property
    def manifest(self) -> dict[str, bool]:
        return {
            "filesystem_read_only": self.filesystem_read_only,
            "live_tracking_blocked": self.live_tracking_blocked,
            "network_disabled": self.network_disabled,
        }

    @property
    def request_digest(self) -> str:
        return canonical_digest(self.manifest)


class D1ExecutorKind(StrEnum):
    EXTERNAL_ENFORCED = "external_enforced"
    TEST_FAKE = "test_fake_not_execution_evidence"


@dataclass(frozen=True, slots=True)
class D1ExternalIsolationAttestation:
    request_digest: str
    executor_kind: D1ExecutorKind
    network_disabled: bool
    filesystem_read_only: bool
    live_tracking_blocked: bool
    mechanism: str

    def __post_init__(self) -> None:
        _sha256(self.request_digest, label="D1 isolation request_digest")
        if type(self.executor_kind) is not D1ExecutorKind:
            raise TypeError("D1 isolation executor_kind must be nominal")
        if not (
            self.network_disabled is True
            and self.filesystem_read_only is True
            and self.live_tracking_blocked is True
        ):
            raise ValueError("D1 isolation attestation does not satisfy the requested boundary")
        if type(self.mechanism) is not str or not self.mechanism.strip():
            raise ValueError("D1 isolation attestation mechanism must be nonempty")

    @classmethod
    def enforced(
        cls,
        *,
        request: D1IsolationRequest,
        mechanism: str,
    ) -> D1ExternalIsolationAttestation:
        if type(request) is not D1IsolationRequest:
            raise TypeError("D1 external isolation requires D1IsolationRequest")
        return cls(
            request.request_digest,
            D1ExecutorKind.EXTERNAL_ENFORCED,
            True,
            True,
            True,
            mechanism,
        )

    @classmethod
    def test_fake(cls, request: D1IsolationRequest) -> D1ExternalIsolationAttestation:
        return cls(
            request.request_digest,
            D1ExecutorKind.TEST_FAKE,
            True,
            True,
            True,
            "in-process-contract-fake",
        )

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "executor_kind": self.executor_kind.value,
            "filesystem_read_only": self.filesystem_read_only,
            "live_tracking_blocked": self.live_tracking_blocked,
            "mechanism": self.mechanism,
            "network_disabled": self.network_disabled,
            "request_digest": self.request_digest,
        }


@dataclass(frozen=True, slots=True)
class D1AnalysisRunnerRequest:
    analysis_request_digest: str
    isolation: D1IsolationRequest

    def __post_init__(self) -> None:
        _sha256(self.analysis_request_digest, label="D1 analysis request_digest")
        if type(self.isolation) is not D1IsolationRequest:
            raise TypeError("D1 analysis runner request requires D1IsolationRequest")


@dataclass(frozen=True, slots=True)
class D1AnalysisRunnerResult:
    analysis: D1AliasedMachineAnalysis
    isolation: D1ExternalIsolationAttestation


class D1AnalysisRunnerPort(Protocol):
    def run(self, request: D1AnalysisRunnerRequest) -> D1AnalysisRunnerResult: ...


class D1InProcessIsolationRunnerFake:
    """Test double for the runner port; explicitly never filing evidence."""

    def __init__(
        self,
        execute: Callable[[D1AnalysisRunnerRequest], D1AliasedMachineAnalysis],
    ) -> None:
        if not callable(execute):
            raise TypeError("D1 isolation runner fake requires a callable")
        self._execute = execute

    def run(self, request: D1AnalysisRunnerRequest) -> D1AnalysisRunnerResult:
        if type(request) is not D1AnalysisRunnerRequest:
            raise TypeError("D1 isolation runner fake requires D1AnalysisRunnerRequest")
        analysis = self._execute(request)
        if type(analysis) is not D1AliasedMachineAnalysis:
            raise TypeError("D1 isolation runner callable must return aliased machine analysis")
        return D1AnalysisRunnerResult(
            analysis,
            D1ExternalIsolationAttestation.test_fake(request.isolation),
        )


@dataclass(frozen=True, slots=True)
class D1AnalysisImplementationIdentity:
    code_commit: str
    repository_digest: str
    analysis_code_digest: str
    image_digest: str
    complete_reader_digest: str
    defect_reader_digest: str

    def __post_init__(self) -> None:
        _commit(self.code_commit)
        for field in (
            "repository_digest",
            "analysis_code_digest",
            "complete_reader_digest",
            "defect_reader_digest",
        ):
            _sha256(getattr(self, field), label=f"D1 {field}")
        if not self.image_digest.startswith("sha256:"):
            raise ValueError("D1 analysis image_digest must be immutable sha256 OCI identity")
        _sha256(self.image_digest.removeprefix("sha256:"), label="D1 analysis image_digest")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "analysis_code_digest": self.analysis_code_digest,
            "code_commit": self.code_commit,
            "image_digest": self.image_digest,
            "readers": {
                "complete": {"identity": _COMPLETE_READER, "sha256": self.complete_reader_digest},
                "terminal_defect": {
                    "identity": _DEFECT_READER,
                    "sha256": self.defect_reader_digest,
                },
            },
            "repository_digest": self.repository_digest,
        }


@dataclass(frozen=True, slots=True)
class D1AnalysisExecutionReceipt:
    canonical_json: str

    @property
    def manifest(self) -> dict[str, object]:
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):  # pragma: no cover - issuer always writes an object
            raise AssertionError("D1 analysis receipt lost its object root")
        return value

    @property
    def receipt_digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _inputs(admitted: D1AdmittedAnalysisCapture) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for artifact in sorted(admitted.input_artifacts, key=lambda item: item.normalized_path):
        dvc = artifact.dvc
        if dvc.state is not D1DvcBindingState.BOUND:
            raise ValueError("D1 analysis receipt requires DVC-bound input artifacts")
        values.append(
            {
                "dvc_object_id": dvc.object_id,
                "dvc_pointer_path": dvc.pointer_path,
                "dvc_pointer_sha256": dvc.pointer_sha256,
                "md5": artifact.md5,
                "path": artifact.normalized_path,
                "role": artifact.role.value,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
        )
    if not values:
        raise ValueError("D1 analysis receipt requires at least one input artifact")
    return values


def issue_d1_analysis_receipt(
    *,
    analysis: D1AliasedMachineAnalysis,
    admitted: D1AdmittedAnalysisCapture,
    aliases: D1ArmAliasMap,
    implementation: D1AnalysisImplementationIdentity,
    isolation: D1ExternalIsolationAttestation,
) -> D1AnalysisExecutionReceipt:
    """Bind analysis bytes to capture, readers, code, image, isolation, and inputs."""
    if (
        type(analysis) is not D1AliasedMachineAnalysis
        or type(admitted) is not D1AdmittedAnalysisCapture
        or type(aliases) is not D1ArmAliasMap
        or type(implementation) is not D1AnalysisImplementationIdentity
        or type(isolation) is not D1ExternalIsolationAttestation
    ):
        raise TypeError("D1 analysis receipt requires nominal evidence values")
    machine = analysis.manifest
    bindings = machine.get("bindings")
    expected = {
        "capture_digest": admitted.capture_digest,
        "decision_contract_digest": admitted.decision_contract_digest,
        "design_digest": admitted.design_digest,
    }
    if not isinstance(bindings, Mapping) or dict(bindings) != expected:
        raise ValueError("D1 analysis receipt machine bindings differ from admitted capture")
    if machine.get("alias_commitment") != aliases.sealed_commitment:
        raise ValueError("D1 analysis receipt alias commitment differs")
    manifest: dict[str, object] = {
        "alias_commitment": aliases.sealed_commitment,
        "analysis_digest": analysis.analysis_digest,
        "bindings": expected,
        "filing_eligible": isolation.executor_kind is D1ExecutorKind.EXTERNAL_ENFORCED,
        "implementation": implementation.manifest,
        "inputs": _inputs(admitted),
        "isolation": isolation.manifest,
        "schema_version": _SCHEMA,
    }
    canonical = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return D1AnalysisExecutionReceipt(canonical)
