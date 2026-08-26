"""Pod-side D1 process worker over immutable request and report artifacts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from entropic_substrate.d1_artifact_snapshot import (
    normalize_d1_artifact_root,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.d1_observability import D1JournalObserver
from entropic_substrate.d1_operator_evidence import d1_runtime_observation_sha256
from entropic_substrate.d1_operator_values import D1RuntimeObservation
from entropic_substrate.d1_resource_activation import D1ResourceActivationFactory
from entropic_substrate.d1_trainer_cli_services import (
    D1FreshTrainerCliService,
    D1ResumeTrainerCliService,
)
from entropic_substrate.d1_trainer_process_contracts import (
    D1TrainerStartReport,
    d1_trainer_request_path,
    d1_trainer_runtime_report_path,
    d1_trainer_start_report_path,
    parse_d1_runtime_report,
    parse_d1_start_report,
    serialize_d1_runtime_report,
    serialize_d1_start_report,
)
from entropic_substrate.d1_trainer_request_codec import (
    D1FreshTrainerRequest,
    D1ResumeTrainerRequest,
    D1TrainerRequest,
    d1_trainer_command_sha256,
    d1_trainer_request_sha256,
    parse_d1_trainer_request,
    verify_d1_trainer_request_runtime,
)
from entropic_substrate.d1_trainer_runtime import D1TrainerClock
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.experiments.d1_config import D1LaunchProfile
from entropic_substrate.experiments.d1_primitives import require_nonempty_string

__all__ = ["D1TrainerProcessWorker"]


def _pod_id(value: str) -> str:
    require_nonempty_string(value, label="D1 process worker pod_id")
    if value.strip() != value:
        raise ValueError("D1 process worker pod_id must be trimmed")
    return value


class D1TrainerProcessWorker:
    """Execute one report-bound fresh or resume request attempt inside a pod."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        catalog: Iterable[D1LaunchProfile],
        runtime_probe: Callable[[str], D1RuntimeObservation],
        process_identity: Callable[[], str],
        observer: D1JournalObserver,
        clock: D1TrainerClock | None = None,
        resource_activation_factory: D1ResourceActivationFactory | None = None,
    ) -> None:
        if not callable(runtime_probe) or not callable(process_identity):
            raise TypeError("D1 process worker requires runtime and process identity probes")
        self._root = normalize_d1_artifact_root(artifact_root)
        self._catalog = tuple(catalog)
        self._runtime_probe = runtime_probe
        self._process_identity = process_identity
        self._fresh = D1FreshTrainerCliService(
            catalog=self._catalog,
            artifact_root=self._root,
            clock=clock,
            resource_activation_factory=resource_activation_factory,
        )
        self._resume = D1ResumeTrainerCliService(
            catalog=self._catalog,
            artifact_root=self._root,
            observer=observer,
            clock=clock,
            resource_activation_factory=resource_activation_factory,
        )

    def _publish_report(self, relative: str, payload: bytes, *, label: str) -> ArtifactIdentity:
        identity = publish_d1_atomic_bytes(self._root, relative, payload, label=label)
        path, observed, digest = read_d1_artifact_bytes(self._root, relative, label=label)
        if (
            observed != payload
            or identity.path != str(path)
            or identity.content_hash != ContentHash("sha256", digest)
        ):
            raise ValueError(f"{label} changed after atomic publication")
        return identity

    def bootstrap(self, pod_id: str, attempt_sha256: str) -> ArtifactIdentity:
        """Measure and durably publish one runtime report for this attempt."""
        pod = _pod_id(pod_id)
        relative = d1_trainer_runtime_report_path(attempt_sha256)
        runtime = self._runtime_probe(pod)
        if type(runtime) is not D1RuntimeObservation or runtime.pod_id != pod:
            raise ValueError("D1 runtime probe differs from the requested pod")
        payload = serialize_d1_runtime_report(runtime, attempt_sha256=attempt_sha256)
        return self._publish_report(relative, payload, label="D1 trainer runtime report")

    def _runtime(self, pod_id: str, attempt_sha256: str) -> D1RuntimeObservation:
        _, payload, _ = read_d1_artifact_bytes(
            self._root,
            d1_trainer_runtime_report_path(attempt_sha256),
            label="D1 trainer runtime report",
        )
        return parse_d1_runtime_report(
            payload,
            pod_id=pod_id,
            attempt_sha256=attempt_sha256,
        )

    def _request(self, attempt_sha256: str, expected_sha256: str) -> tuple[D1TrainerRequest, Path]:
        path, payload, observed_sha256 = read_d1_artifact_bytes(
            self._root,
            d1_trainer_request_path(attempt_sha256),
            label="D1 staged trainer request",
        )
        if observed_sha256 != expected_sha256:
            raise ValueError("D1 staged trainer request digest differs from start authority")
        request = parse_d1_trainer_request(payload, catalog=self._catalog)
        if d1_trainer_request_sha256(request) != expected_sha256:
            raise ValueError("D1 parsed trainer request digest differs from start authority")
        return request, path

    def start_trainer(
        self,
        pod_id: str,
        attempt_sha256: str,
        request_sha256: str,
    ) -> Mapping[str, object]:
        """Publish start evidence, then invoke the matching real trainer service."""
        pod = _pod_id(pod_id)
        relative = d1_trainer_start_report_path(attempt_sha256, request_sha256)
        runtime = self._runtime(pod, attempt_sha256)
        request, request_path = self._request(attempt_sha256, request_sha256)
        verify_d1_trainer_request_runtime(request, runtime)
        process_id = self._process_identity()
        report = D1TrainerStartReport(
            pod_id=pod,
            attempt_sha256=attempt_sha256,
            request_sha256=request_sha256,
            runtime_sha256=d1_runtime_observation_sha256(runtime),
            trainer_started_at=request.start.trainer_started_at,
            process_id=process_id,
            command_sha256=d1_trainer_command_sha256(request),
        )
        payload = serialize_d1_start_report(report)
        self._publish_report(relative, payload, label="D1 trainer start report")
        admitted = parse_d1_start_report(
            read_d1_artifact_bytes(
                self._root,
                relative,
                label="D1 trainer start report",
            )[1],
            pod_id=pod,
            attempt_sha256=attempt_sha256,
            request_sha256=request_sha256,
            runtime_sha256=report.runtime_sha256,
            trainer_started_at=report.trainer_started_at,
            command_sha256=report.command_sha256,
        )
        if admitted != report:
            raise ValueError("D1 trainer start report changed after readback")
        if type(request) is D1FreshTrainerRequest:
            return self._fresh.execute_request(request, manifest_path=request_path)
        if type(request) is D1ResumeTrainerRequest:
            return self._resume.execute_request(request, manifest_path=request_path)
        raise AssertionError(
            "validated D1 trainer request lost its nominal type"
        )  # pragma: no cover
