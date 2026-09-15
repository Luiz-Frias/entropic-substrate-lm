"""Isolation port and deterministic execution receipt for D1 analysis."""

from __future__ import annotations

import hashlib
import json
import threading
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import NoReturn, Protocol, SupportsIndex

from entropic_substrate.analysis.d1_analysis_receipt_codec import (
    D1_ANALYSIS_EXECUTION_RECEIPT_SCHEMA,
    D1AnalysisExecutionReceipt,
    D1AnalysisFilingBindings,
    canonical_d1_analysis_receipt_bytes,
    parse_d1_analysis_execution_receipt,
)
from entropic_substrate.analysis.d1_capture import D1AdmittedAnalysisCapture
from entropic_substrate.analysis.d1_external_analysis_codec import (
    D1ExternalAnalysisRequestEnvelopeV2,
    D1ExternalAnalysisWorkerResponseV2,
    encode_d1_external_analysis_request_v2,
    encode_d1_external_analysis_worker_response_v2,
)
from entropic_substrate.analysis.d1_machine_analysis import (
    D1AliasedMachineAnalysis,
    D1ArmAliasMap,
)
from entropic_substrate.analysis.d1_machine_analysis_verification import (
    D1VerifiedMachineAnalysis,
)
from entropic_substrate.evidence.d1_capture_artifacts import D1DvcBindingState
from entropic_substrate.experiments.d1_primitives import canonical_digest, require_sha256

__all__ = [
    "D1AnalysisExecutionReceipt",
    "D1AnalysisFilingBindings",
    "D1AnalysisImplementationIdentity",
    "D1AnalysisRunnerPort",
    "D1AnalysisRunnerRequest",
    "D1AnalysisRunnerResult",
    "D1ExecutorKind",
    "D1ExternalIsolationAttestation",
    "D1InProcessIsolationRunnerFake",
    "D1IsolationExecutionEvidence",
    "D1IsolationRequest",
    "issue_d1_analysis_receipt",
    "issue_d1_verified_analysis_receipt",
    "parse_d1_analysis_execution_receipt",
]

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
class D1IsolationExecutionEvidence:
    """Observed process and output identities behind one external attestation."""

    command_sha256: str
    exit_code: int
    request_sha256: str
    analysis_sha256: str
    response_sha256: str
    stdout_sha256: str
    stderr_sha256: str
    image_digest: str
    platform: str

    def __post_init__(self) -> None:
        for name in (
            "command_sha256",
            "request_sha256",
            "analysis_sha256",
            "response_sha256",
            "stdout_sha256",
            "stderr_sha256",
        ):
            _sha256(getattr(self, name), label=f"D1 isolation {name}")
        if type(self.exit_code) is not int or self.exit_code != 0:
            raise ValueError("D1 enforced isolation requires observed exit_code 0")
        if not self.image_digest.startswith("sha256:"):
            raise ValueError("D1 isolation image_digest must be immutable")
        _sha256(self.image_digest.removeprefix("sha256:"), label="D1 isolation image_digest")
        if self.platform != "linux/amd64":
            raise ValueError("D1 isolation platform must be linux/amd64")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "analysis_sha256": self.analysis_sha256,
            "command_sha256": self.command_sha256,
            "exit_code": self.exit_code,
            "image_digest": self.image_digest,
            "platform": self.platform,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "stderr_sha256": self.stderr_sha256,
            "stdout_sha256": self.stdout_sha256,
        }

    @property
    def evidence_digest(self) -> str:
        return canonical_digest(self.manifest)


def _validate_isolation_values(
    *,
    request_digest: str,
    executor_kind: D1ExecutorKind,
    network_disabled: bool,
    filesystem_read_only: bool,
    live_tracking_blocked: bool,
    mechanism: str,
    execution_evidence: D1IsolationExecutionEvidence | None,
) -> None:
    _sha256(request_digest, label="D1 isolation request_digest")
    if type(executor_kind) is not D1ExecutorKind:
        raise TypeError("D1 isolation executor_kind must be nominal")
    if not (
        network_disabled is True and filesystem_read_only is True and live_tracking_blocked is True
    ):
        raise ValueError("D1 isolation attestation does not satisfy the requested boundary")
    if type(mechanism) is not str or not mechanism.strip():
        raise ValueError("D1 isolation attestation mechanism must be nonempty")
    if executor_kind is D1ExecutorKind.EXTERNAL_ENFORCED:
        if type(execution_evidence) is not D1IsolationExecutionEvidence:
            raise TypeError("D1 enforced isolation requires process execution evidence")
    elif execution_evidence is not None:
        raise ValueError("D1 test isolation cannot carry external execution evidence")


@dataclass(frozen=True, slots=True, weakref_slot=True)
class D1ExternalIsolationAttestation:
    request_digest: str
    executor_kind: D1ExecutorKind
    network_disabled: bool
    filesystem_read_only: bool
    live_tracking_blocked: bool
    mechanism: str
    execution_evidence: D1IsolationExecutionEvidence | None
    _attestation: _ExternalIsolationAttestation | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
        init=False,
    )

    def __post_init__(self) -> None:
        _validate_isolation_values(
            request_digest=self.request_digest,
            executor_kind=self.executor_kind,
            network_disabled=self.network_disabled,
            filesystem_read_only=self.filesystem_read_only,
            live_tracking_blocked=self.live_tracking_blocked,
            mechanism=self.mechanism,
            execution_evidence=self.execution_evidence,
        )
        if self.executor_kind is D1ExecutorKind.EXTERNAL_ENFORCED:
            raise TypeError(
                "D1 enforced isolation is factory-only; use the verified external adapter"
            )

    def __copy__(self) -> NoReturn:
        raise TypeError("D1 external isolation attestation cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        raise TypeError("D1 external isolation attestation cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("D1 external isolation attestation cannot be copied")

    def require_factory_attested(self) -> None:
        if not _is_minted_external_isolation(self):
            raise TypeError("factory-attested external isolation was not adapter-minted")
        attestation = getattr(self, "_attestation", None)
        if type(attestation) is not _ExternalIsolationAttestation:
            raise TypeError("factory-attested external isolation marker is absent")
        if attestation.owner_ref() is not self:
            raise TypeError("external isolation marker belongs to another instance")
        _validate_isolation_values(
            request_digest=self.request_digest,
            executor_kind=self.executor_kind,
            network_disabled=self.network_disabled,
            filesystem_read_only=self.filesystem_read_only,
            live_tracking_blocked=self.live_tracking_blocked,
            mechanism=self.mechanism,
            execution_evidence=self.execution_evidence,
        )
        if canonical_digest(self.manifest) != attestation.manifest_digest:
            raise TypeError("external isolation changed after minting")

    @classmethod
    def test_fake(cls, request: D1IsolationRequest) -> D1ExternalIsolationAttestation:
        return cls(
            request.request_digest,
            D1ExecutorKind.TEST_FAKE,
            True,
            True,
            True,
            "in-process-contract-fake",
            None,
        )

    @property
    def filing_eligible(self) -> bool:
        if self.executor_kind is not D1ExecutorKind.EXTERNAL_ENFORCED:
            return False
        try:
            self.require_factory_attested()
        except (TypeError, ValueError):
            return False
        return type(self.execution_evidence) is D1IsolationExecutionEvidence

    @property
    def execution_evidence_digest(self) -> str | None:
        return None if self.execution_evidence is None else self.execution_evidence.evidence_digest

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "executor_kind": self.executor_kind.value,
            "filesystem_read_only": self.filesystem_read_only,
            "live_tracking_blocked": self.live_tracking_blocked,
            "mechanism": self.mechanism,
            "network_disabled": self.network_disabled,
            "request_digest": self.request_digest,
            "execution_evidence": (
                None if self.execution_evidence is None else self.execution_evidence.manifest
            ),
        }


class _ExternalIsolationAttestation:
    __slots__ = ("manifest_digest", "owner_ref")

    manifest_digest: str
    owner_ref: weakref.ReferenceType[D1ExternalIsolationAttestation]

    def __init__(self, owner: D1ExternalIsolationAttestation, manifest_digest: str) -> None:
        self.owner_ref = weakref.ref(owner)
        self.manifest_digest = manifest_digest


_EXTERNAL_ISOLATION_LOCK = threading.Lock()
_MINTED_EXTERNAL_ISOLATIONS: dict[
    int,
    weakref.ReferenceType[D1ExternalIsolationAttestation],
] = {}


def _record_minted_external_isolation(value: D1ExternalIsolationAttestation) -> None:
    identifier = id(value)

    def discard(reference: weakref.ReferenceType[D1ExternalIsolationAttestation]) -> None:
        with _EXTERNAL_ISOLATION_LOCK:
            if _MINTED_EXTERNAL_ISOLATIONS.get(identifier) is reference:
                _MINTED_EXTERNAL_ISOLATIONS.pop(identifier, None)

    reference = weakref.ref(value, discard)
    with _EXTERNAL_ISOLATION_LOCK:
        _MINTED_EXTERNAL_ISOLATIONS[identifier] = reference


def _is_minted_external_isolation(value: D1ExternalIsolationAttestation) -> bool:
    with _EXTERNAL_ISOLATION_LOCK:
        reference = _MINTED_EXTERNAL_ISOLATIONS.get(id(value))
        return reference is not None and reference() is value


def _mint_external_isolation(  # pyright: ignore[reportUnusedFunction]
    *,
    request: D1IsolationRequest,
    mechanism: str,
    execution_evidence: D1IsolationExecutionEvidence,
) -> D1ExternalIsolationAttestation:
    if type(request) is not D1IsolationRequest:
        raise TypeError("D1 external isolation requires D1IsolationRequest")
    _validate_isolation_values(
        request_digest=request.request_digest,
        executor_kind=D1ExecutorKind.EXTERNAL_ENFORCED,
        network_disabled=True,
        filesystem_read_only=True,
        live_tracking_blocked=True,
        mechanism=mechanism,
        execution_evidence=execution_evidence,
    )
    value = object.__new__(D1ExternalIsolationAttestation)
    object.__setattr__(value, "request_digest", request.request_digest)
    object.__setattr__(value, "executor_kind", D1ExecutorKind.EXTERNAL_ENFORCED)
    object.__setattr__(value, "network_disabled", True)
    object.__setattr__(value, "filesystem_read_only", True)
    object.__setattr__(value, "live_tracking_blocked", True)
    object.__setattr__(value, "mechanism", mechanism)
    object.__setattr__(value, "execution_evidence", execution_evidence)
    object.__setattr__(
        value,
        "_attestation",
        _ExternalIsolationAttestation(value, canonical_digest(value.manifest)),
    )
    _record_minted_external_isolation(value)
    return value


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


def _issue_d1_analysis_receipt(
    *,
    analysis: D1AliasedMachineAnalysis,
    admitted: D1AdmittedAnalysisCapture,
    aliases: D1ArmAliasMap,
    implementation: D1AnalysisImplementationIdentity,
    isolation: D1ExternalIsolationAttestation,
    filing_bindings: D1AnalysisFilingBindings | None,
) -> D1AnalysisExecutionReceipt:
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
    if isolation.filing_eligible is not (type(filing_bindings) is D1AnalysisFilingBindings):
        raise ValueError(
            "D1 analysis receipt filing eligibility requires exact filed-byte bindings"
        )
    if filing_bindings is not None and filing_bindings.output_sha256 != analysis.analysis_digest:
        raise ValueError("D1 analysis receipt filing output differs from analysis")
    manifest: dict[str, object] = {
        "alias_commitment": aliases.sealed_commitment,
        "analysis_digest": analysis.analysis_digest,
        "bindings": expected,
        "filing_eligible": isolation.filing_eligible,
        "filing_bindings": (None if filing_bindings is None else filing_bindings.manifest),
        "implementation": implementation.manifest,
        "inputs": _inputs(admitted),
        "isolation": isolation.manifest,
        "schema_version": D1_ANALYSIS_EXECUTION_RECEIPT_SCHEMA,
    }
    canonical = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return D1AnalysisExecutionReceipt(canonical)


def issue_d1_analysis_receipt(
    *,
    analysis: D1AliasedMachineAnalysis,
    admitted: D1AdmittedAnalysisCapture,
    aliases: D1ArmAliasMap,
    implementation: D1AnalysisImplementationIdentity,
    isolation: D1ExternalIsolationAttestation,
) -> D1AnalysisExecutionReceipt:
    """Issue a non-filing receipt for an in-process analysis contract fake."""
    if isolation.filing_eligible:
        raise ValueError("filing-eligible analysis requires issue_d1_verified_analysis_receipt")
    return _issue_d1_analysis_receipt(
        analysis=analysis,
        admitted=admitted,
        aliases=aliases,
        implementation=implementation,
        isolation=isolation,
        filing_bindings=None,
    )


def issue_d1_verified_analysis_receipt(
    *,
    analysis: D1VerifiedMachineAnalysis,
    request: D1ExternalAnalysisRequestEnvelopeV2,
    worker_response: D1ExternalAnalysisWorkerResponseV2,
    implementation: D1AnalysisImplementationIdentity,
    isolation: D1ExternalIsolationAttestation,
) -> D1AnalysisExecutionReceipt:
    """Bind the complete v2 request/execution/analysis chain before filing."""
    if (
        type(analysis) is not D1VerifiedMachineAnalysis
        or type(request) is not D1ExternalAnalysisRequestEnvelopeV2
        or type(worker_response) is not D1ExternalAnalysisWorkerResponseV2
        or type(implementation) is not D1AnalysisImplementationIdentity
        or type(isolation) is not D1ExternalIsolationAttestation
    ):
        raise TypeError("D1 v2 analysis receipt requires nominal verified values")
    if analysis.expected is not request.input_control:
        raise ValueError("D1 v2 receipt request input differs from verified analysis")
    reader_source = request.input_control.reader_source
    reader_digests = {entry.identity: entry.sha256 for entry in reader_source.modules}
    complete_reader_module = reader_source.complete_reader.rpartition(".")[0]
    defect_reader_module = reader_source.terminal_defect_reader.rpartition(".")[0]
    if implementation.analysis_code_digest != reader_source.analysis_code_digest:
        raise ValueError("D1 v2 receipt analysis code digest differs from reader source")
    if implementation.complete_reader_digest != reader_digests.get(complete_reader_module):
        raise ValueError("D1 v2 receipt complete reader digest differs from reader source")
    if implementation.defect_reader_digest != reader_digests.get(defect_reader_module):
        raise ValueError("D1 v2 receipt terminal defect reader digest differs from reader source")
    if analysis.aggregate.manifest_digest != request.input_control.aggregate.manifest_digest:
        raise ValueError("D1 v2 receipt aggregate differs from request input")
    if worker_response.analysis_request_digest != request.analysis_request_digest:
        raise ValueError("D1 v2 receipt worker response differs from request")
    if worker_response.isolation_request_digest != request.isolation_request_digest:
        raise ValueError("D1 v2 receipt worker response differs from isolation")
    if worker_response.input_manifest_sha256 != request.input_manifest_sha256:
        raise ValueError("D1 v2 receipt worker response differs from input manifest")
    if worker_response.analysis_sha256 != analysis.analysis_digest:
        raise ValueError("D1 v2 receipt worker response differs from analysis")
    if isolation.request_digest != request.isolation_request_digest:
        raise ValueError("D1 v2 receipt isolation differs from request")
    isolation.require_factory_attested()
    evidence = isolation.execution_evidence
    if type(evidence) is not D1IsolationExecutionEvidence:
        raise TypeError("D1 v2 receipt isolation lacks execution evidence")
    if (
        evidence.request_sha256
        != hashlib.sha256(encode_d1_external_analysis_request_v2(request)).hexdigest()
    ):
        raise ValueError("D1 v2 receipt execution request hash differs")
    if (
        evidence.response_sha256
        != hashlib.sha256(
            encode_d1_external_analysis_worker_response_v2(worker_response)
        ).hexdigest()
    ):
        raise ValueError("D1 v2 receipt execution response hash differs")
    if evidence.analysis_sha256 != analysis.analysis_digest:
        raise ValueError("D1 v2 receipt execution analysis hash differs")
    admitted = D1AdmittedAnalysisCapture(
        design_digest=request.input_control.design.design_digest,
        decision_contract_digest=request.input_control.design.decision_contract_digest,
        capture_digest=analysis.aggregate.manifest_digest,
        cells=(),
        exclusions=(),
        runs=(),
        input_artifacts=analysis.aggregate.dvc_handoff.artifacts,
    )
    filing_bindings = D1AnalysisFilingBindings(
        capture_manifest_sha256=hashlib.sha256(
            request.input_control.aggregate.canonical_json.encode("utf-8")
        ).hexdigest(),
        preregistration_sha256=request.input_control.preregistration.sha256,
        reader_sha256=hashlib.sha256(
            request.input_control.reader_source.canonical_bytes
        ).hexdigest(),
        isolation_attestation_sha256=hashlib.sha256(
            canonical_d1_analysis_receipt_bytes(isolation.manifest)
        ).hexdigest(),
        output_sha256=analysis.analysis_digest,
    )
    return _issue_d1_analysis_receipt(
        analysis=analysis.analysis,
        admitted=admitted,
        aliases=request.input_control.alias_application.aliases,
        implementation=implementation,
        isolation=isolation,
        filing_bindings=filing_bindings,
    )
