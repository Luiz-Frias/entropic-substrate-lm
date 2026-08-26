"""Stateful D1 trainer adapter over raw process contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from entropic_substrate.d1_artifact_snapshot import normalize_d1_relative_artifact_path
from entropic_substrate.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    D1StagedTrainerRequestReceipt,
    D1TrainerStartReceipt,
    attest_d1_staged_trainer_request,
    attest_d1_trainer_start,
    d1_runtime_observation_sha256,
)
from entropic_substrate.d1_operator_values import D1RuntimeObservation
from entropic_substrate.d1_trainer_output_publication import load_d1_published_trainer_output
from entropic_substrate.d1_trainer_process_contracts import (
    D1TrainerAdapterPhase,
    D1TrainerArtifactReadback,
    D1TrainerProcessTransport,
    d1_trainer_request_path,
    parse_d1_runtime_report,
    parse_d1_start_report,
)
from entropic_substrate.d1_trainer_request_codec import (
    D1TrainerRequest,
    d1_trainer_command_sha256,
    d1_trainer_request_sha256,
    parse_d1_trainer_request,
    serialize_d1_trainer_request,
)
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.experiments.d1_config import D1LaunchProfile
from entropic_substrate.experiments.d1_primitives import require_nonempty_string, require_sha256

__all__ = ["D1TrainerEvidenceAdapter"]


@dataclass(slots=True)
class _Session:
    pod_id: str
    attempt_sha256: str
    phase: D1TrainerAdapterPhase
    runtime: D1RuntimeObservation | None = None
    request: D1TrainerRequest | None = None
    staged: D1StagedTrainerRequestReceipt | None = None


def _pod(value: str) -> str:
    require_nonempty_string(value, label="trainer pod_id")
    if value.strip() != value:
        raise ValueError("trainer pod_id must be trimmed")
    return value


def _attempt(value: str) -> str:
    require_sha256(value, label="trainer attempt")
    return value


class D1TrainerEvidenceAdapter:
    def __init__(
        self, *, transport: D1TrainerProcessTransport, catalog: Iterable[D1LaunchProfile]
    ) -> None:
        methods = (
            "bootstrap",
            "read_runtime_report",
            "write_request",
            "read_request",
            "start_trainer",
            "read_start_report",
            "await_export",
        )
        if not all(callable(getattr(transport, method, None)) for method in methods):
            raise TypeError("D1 trainer adapter requires the raw process transport")
        self._transport, self._catalog = transport, tuple(catalog)
        if not self._catalog or any(type(item) is not D1LaunchProfile for item in self._catalog):
            raise TypeError("D1 trainer adapter requires an exact profile catalog")
        identities = tuple((item.name, item.scale) for item in self._catalog)
        if len(set(identities)) != len(identities):
            raise ValueError("D1 trainer adapter profile catalog repeats an identity")
        self._sessions: dict[str, _Session] = {}

    def phase(self, pod_id: str) -> D1TrainerAdapterPhase:
        return self._session(pod_id).phase

    def _session(self, pod_id: str, expected: D1TrainerAdapterPhase | None = None) -> _Session:
        session = self._sessions.get(_pod(pod_id))
        if session is None:
            raise ValueError("D1 trainer pod has no bootstrap session")
        if expected is not None and session.phase is not expected:
            raise ValueError(f"D1 trainer phase is {session.phase}, expected {expected}")
        return session

    def bootstrap_runtime(self, pod_id: str, attempt_sha256: str) -> D1RuntimeObservation:
        pod, attempt = _pod(pod_id), _attempt(attempt_sha256)
        if pod in self._sessions:
            raise ValueError("D1 trainer bootstrap was already attempted for this pod")
        session = _Session(pod, attempt, D1TrainerAdapterPhase.BOOTSTRAP_ATTEMPTED)
        self._sessions[pod] = session
        self._transport.bootstrap(pod, attempt)
        session.phase = D1TrainerAdapterPhase.RUNTIME_READBACK_ATTEMPTED
        raw = self._transport.read_runtime_report(pod, attempt)
        session.runtime = parse_d1_runtime_report(raw, pod_id=pod, attempt_sha256=attempt)
        session.phase = D1TrainerAdapterPhase.RUNTIME_ATTESTED
        return session.runtime

    def stage_inputs(
        self, pod_id: str, *, request: D1TrainerRequest
    ) -> D1StagedTrainerRequestReceipt:
        session = self._session(pod_id, D1TrainerAdapterPhase.RUNTIME_ATTESTED)
        runtime = session.runtime
        if runtime is None:
            raise RuntimeError("D1 trainer session lost its runtime")
        payload, request_sha = (
            serialize_d1_trainer_request(request),
            d1_trainer_request_sha256(request),
        )
        observed = (
            runtime.trainer_started_at,
            runtime.cuda_version,
            runtime.driver_version,
            runtime.pytorch_version,
            runtime.compiler_enabled,
            runtime.compiler_version,
            runtime.trainer_monotonic_origin_ns,
        )
        expected = (
            request.start.trainer_started_at,
            request.start.cuda_version,
            request.start.driver_version,
            request.start.pytorch_version,
            request.start.compiler_enabled,
            request.start.compiler_version,
            request.origin.monotonic_ns,
        )
        if observed != expected:
            raise ValueError("D1 trainer request differs from attested runtime")
        target = d1_trainer_request_path(session.attempt_sha256)
        session.phase = D1TrainerAdapterPhase.STAGE_WRITE_ATTEMPTED
        self._transport.write_request(session.pod_id, session.attempt_sha256, target, payload)
        session.phase = D1TrainerAdapterPhase.STAGE_READBACK_ATTEMPTED
        readback = self._transport.read_request(session.pod_id, session.attempt_sha256, target)
        if type(readback) is not D1TrainerArtifactReadback:
            raise TypeError("D1 trainer transport must return raw request readback")
        path = normalize_d1_relative_artifact_path(
            readback.path,
            label="D1 staged trainer request",
        )
        if path != target:
            raise ValueError("D1 trainer request readback path differs from its write target")
        if readback.payload != payload:
            raise ValueError("D1 trainer request readback bytes differ")
        parsed = parse_d1_trainer_request(readback.payload, catalog=self._catalog)
        if type(parsed) is not type(request) or parsed != request:
            raise ValueError("D1 trainer request readback differs nominally")
        artifact = ArtifactIdentity(path, ContentHash("sha256", request_sha))
        session.staged = attest_d1_staged_trainer_request(request=request, artifact=artifact)
        session.request, session.phase = request, D1TrainerAdapterPhase.REQUEST_STAGED
        return session.staged

    def start_trainer(self, pod_id: str) -> D1TrainerStartReceipt:
        session = self._session(pod_id, D1TrainerAdapterPhase.REQUEST_STAGED)
        request, runtime = session.request, session.runtime
        if request is None or runtime is None:
            raise RuntimeError("D1 trainer session lost staged authority")
        request_sha = d1_trainer_request_sha256(request)
        session.phase = D1TrainerAdapterPhase.START_ATTEMPTED
        self._transport.start_trainer(session.pod_id, session.attempt_sha256, request_sha)
        session.phase = D1TrainerAdapterPhase.START_READBACK_ATTEMPTED
        raw = self._transport.read_start_report(session.pod_id, session.attempt_sha256, request_sha)
        report = parse_d1_start_report(
            raw,
            pod_id=session.pod_id,
            attempt_sha256=session.attempt_sha256,
            request_sha256=request_sha,
            runtime_sha256=d1_runtime_observation_sha256(runtime),
            trainer_started_at=request.start.trainer_started_at,
            command_sha256=d1_trainer_command_sha256(request),
        )
        receipt = attest_d1_trainer_start(
            pod_id=report.pod_id,
            attempt_sha256=report.attempt_sha256,
            request_sha256=report.request_sha256,
            runtime_sha256=report.runtime_sha256,
            trainer_started_at=report.trainer_started_at,
            process_id=report.process_id,
            command_sha256=report.command_sha256,
        )
        session.phase = D1TrainerAdapterPhase.TRAINER_STARTED
        return receipt

    def await_exported_handoff(self, pod_id: str) -> D1ExportedTrainerHandoff:
        session = self._session(pod_id, D1TrainerAdapterPhase.TRAINER_STARTED)
        if session.request is None:
            raise RuntimeError("D1 trainer session lost its request")
        session.phase = D1TrainerAdapterPhase.EXPORT_ATTEMPTED
        prefix = session.request.locations.checkpoint_path
        root = self._transport.await_export(session.pod_id, session.attempt_sha256, prefix)
        candidate = load_d1_published_trainer_output(artifact_root=root, publication_prefix=prefix)
        session.phase = D1TrainerAdapterPhase.EXPORT_ADMITTED_UNBOUND
        return candidate
