"""Verified Docker adapter for fresh, externally isolated D1 analysis."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import Final, NoReturn, Protocol, cast

import entropic_substrate.analysis.d1_execution_receipt as execution_receipt_module
from entropic_substrate.analysis.d1_execution_receipt import (
    D1AnalysisRunnerPort,
    D1AnalysisRunnerRequest,
    D1AnalysisRunnerResult,
    D1ExternalIsolationAttestation,
    D1IsolationExecutionEvidence,
    D1IsolationRequest,
)
from entropic_substrate.analysis.d1_external_analysis_codec import (
    D1ExternalAnalysisRequestEnvelope,
    D1ExternalAnalysisRequestEnvelopeV2,
    D1ExternalAnalysisWorkerResponseV2,
    decode_d1_external_analysis_response,
    decode_d1_external_analysis_worker_response_v2,
    encode_d1_external_analysis_request,
    encode_d1_external_analysis_request_v2,
)
from entropic_substrate.analysis.d1_machine_analysis import D1AliasedMachineAnalysis
from entropic_substrate.analysis.d1_machine_analysis_binding import (
    bind_d1_machine_analysis_contract,
)
from entropic_substrate.analysis.d1_machine_analysis_verification import (
    D1VerifiedMachineAnalysis,
    verify_d1_machine_analysis,
)
from entropic_substrate.evidence.d1_capture_aggregate import D1HeterogeneousCaptureAggregate
from entropic_substrate.experiments.d1_primitives import canonical_digest

_PLATFORM: Final = "linux/amd64"
_ANALYSIS_NAME: Final = "blinded-analysis.json"
_RESPONSE_NAME: Final = "execution-receipt.json"
_WORKER_RESPONSE_NAME: Final = "worker-response.json"
_MECHANISM: Final = "docker-fixed-argv-read-only-network-none-v1"
_MECHANISM_V2: Final = "docker-fixed-argv-read-only-network-none-v2"


def _fail(detail: str, cause: Exception | None = None) -> NoReturn:
    error = RuntimeError(f"D1 external analysis adapter: {detail}")
    if cause is None:
        raise error
    raise error from cause


def _path(value: object, *, label: str, kind: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be a Path")
    lexical = Path(os.path.abspath(value))
    if lexical.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    try:
        metadata = lexical.stat(follow_symlinks=False)
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} must exist: {error}") from error
    if kind == "file" and not S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if kind == "directory" and not S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory")
    if kind == "either" and not (S_ISREG(metadata.st_mode) or S_ISDIR(metadata.st_mode)):
        raise ValueError(f"{label} must be a regular file or directory")
    if any(character in str(resolved) for character in (",", "\n", "\r")):
        raise ValueError(f"{label} contains an unsafe mount character")
    return resolved


def _image(value: object) -> tuple[str, str]:
    if (
        type(value) is not str
        or value.count("@") != 1
        or any(character.isspace() for character in value)
    ):
        raise ValueError("D1 analysis image must be an immutable repository@sha256 digest")
    repository, digest = value.split("@", maxsplit=1)
    if not repository or ":" in repository.rsplit("/", maxsplit=1)[-1]:
        raise ValueError("D1 analysis image repository cannot carry a mutable tag")
    if (
        not digest.startswith("sha256:")
        or len(digest) != 71
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise ValueError("D1 analysis image requires a lowercase sha256 digest")
    return value, digest


@dataclass(frozen=True, slots=True)
class D1ExternalAnalysisLaunch:
    """Host-side mount identities; container destinations and controls stay fixed."""

    image_reference: str
    capture_path: Path
    preregistration_path: Path
    reader_path: Path
    output_path: Path

    def __post_init__(self) -> None:
        image, _ = _image(self.image_reference)
        capture = _path(self.capture_path, label="capture_path", kind="directory")
        preregistration = _path(
            self.preregistration_path, label="preregistration_path", kind="file"
        )
        reader = _path(self.reader_path, label="reader_path", kind="either")
        output = _path(self.output_path, label="output_path", kind="directory")
        if any(output.iterdir()):
            raise ValueError("D1 analysis output_path must start empty")
        for source in (capture, preregistration, reader):
            if source == output or source in output.parents or output in source.parents:
                raise ValueError("D1 analysis writable output cannot overlap a read-only input")
        request = output.parent / f".{output.name}-request.json"
        if request.exists() or request.is_symlink():
            raise ValueError("D1 analysis request_path must not preexist")
        object.__setattr__(self, "image_reference", image)
        object.__setattr__(self, "capture_path", capture)
        object.__setattr__(self, "preregistration_path", preregistration)
        object.__setattr__(self, "reader_path", reader)
        object.__setattr__(self, "output_path", output)

    @property
    def image_digest(self) -> str:
        return _image(self.image_reference)[1]

    @property
    def request_path(self) -> Path:
        return self.output_path.parent / f".{self.output_path.name}-request.json"

    @property
    def analysis_path(self) -> Path:
        return self.output_path / _ANALYSIS_NAME

    @property
    def response_path(self) -> Path:
        return self.output_path / _RESPONSE_NAME

    @property
    def worker_response_path(self) -> Path:
        return self.output_path / _WORKER_RESPONSE_NAME


@dataclass(frozen=True, slots=True)
class D1ExternalAnalysisCommandResult:
    """Observed command identity and process result, not merely requested argv."""

    invoked_argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        if (
            type(self.invoked_argv) is not tuple
            or not self.invoked_argv
            or any(type(item) is not str or not item for item in self.invoked_argv)
        ):
            raise TypeError("D1 external command observation requires nonempty string argv")
        if type(self.returncode) is not int:
            raise TypeError("D1 external command returncode must be an integer")
        if type(self.stdout) is not str or type(self.stderr) is not str:
            raise TypeError("D1 external command streams must be strings")


class _AttestedCommandRunner(Protocol):
    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        stdin: str | None,
    ) -> D1ExternalAnalysisCommandResult: ...


class D1OutputAccessPort(Protocol):
    def grant(self, path: Path, *, uid: int, gid: int) -> None: ...

    def revoke(self, path: Path, *, uid: int, gid: int) -> None: ...


def _mount(source: Path, destination: str, *, read_only: bool) -> str:
    suffix = ",readonly" if read_only else ""
    return f"type=bind,src={source},dst={destination}{suffix}"


def _argv(launch: D1ExternalAnalysisLaunch) -> tuple[str, ...]:
    return (
        "docker",
        "run",
        "--rm",
        "--platform",
        _PLATFORM,
        "--network",
        "none",
        "--read-only",
        "--user",
        "65532:65532",
        "--cpus",
        "2",
        "--memory",
        "4g",
        "--pids-limit",
        "128",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",  # nosec B108 -- isolated container tmpfs
        "--mount",
        _mount(launch.request_path, "/d1/input/request.json", read_only=True),
        "--mount",
        _mount(launch.capture_path, "/d1/input/capture", read_only=True),
        "--mount",
        _mount(launch.preregistration_path, "/d1/input/preregistration", read_only=True),
        "--mount",
        _mount(launch.reader_path, "/d1/input/reader", read_only=True),
        "--mount",
        _mount(launch.output_path, "/d1/output", read_only=False),
        launch.image_reference,
        "python",
        "-m",
        "entropic_substrate.analysis.d1_external_worker",
        "--request",
        "/d1/input/request.json",
        "--analysis",
        "/d1/output/blinded-analysis.json",
        "--response",
        "/d1/output/execution-receipt.json",
    )


def _argv_v2(launch: D1ExternalAnalysisLaunch) -> tuple[str, ...]:
    legacy = _argv(launch)
    image_index = legacy.index(launch.image_reference)
    return (
        *legacy[: image_index + 1],
        "python",
        "-m",
        "entropic_substrate.analysis.d1_external_worker",
        "--request",
        "/d1/input/request.json",
        "--capture-root",
        "/d1/input/capture",
        "--preregistration",
        "/d1/input/preregistration",
        "--reader-root",
        "/d1/input/reader",
        "--analysis",
        "/d1/output/blinded-analysis.json",
        "--response",
        "/d1/output/worker-response.json",
    )


def _write_request(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    path.chmod(0o444)


def _read_once(path: Path, *, label: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"D1 external analysis {label} cannot be a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"D1 external analysis {label} is unreadable: {error}") from error
    try:
        if not S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"D1 external analysis {label} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _read_stable(path: Path, *, label: str) -> bytes:
    first = _read_once(path, label=label)
    second = _read_once(path, label=label)
    if first != second:
        raise ValueError(f"D1 external analysis {label} drifted while being read")
    return first


def _analysis(payload: bytes) -> D1AliasedMachineAnalysis:
    try:
        value: object = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"D1 external analysis output is invalid UTF-8 JSON: {error}") from error
    if type(value) is not dict:
        raise ValueError("D1 external analysis output must be a JSON object")
    canonical = json.dumps(
        cast(dict[str, object], value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    if canonical != payload:
        raise ValueError("D1 external analysis output must use canonical JSON bytes")
    return D1AliasedMachineAnalysis(payload.decode("utf-8"))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mint_isolation(
    *,
    request: D1IsolationRequest,
    mechanism: str,
    execution_evidence: D1IsolationExecutionEvidence,
) -> D1ExternalIsolationAttestation:
    """Conventionally adapter-only; the returned value proves runtime eligibility."""
    return execution_receipt_module._mint_external_isolation(  # pyright: ignore[reportPrivateUsage]
        request=request,
        mechanism=mechanism,
        execution_evidence=execution_evidence,
    )


class DockerD1ExternalAnalysisAdapter(D1AnalysisRunnerPort):
    """Run one immutable analysis image and verify all observable controls."""

    def __init__(self, *, launch: D1ExternalAnalysisLaunch, runner: _AttestedCommandRunner) -> None:
        if type(launch) is not D1ExternalAnalysisLaunch:
            raise TypeError("D1 external analysis adapter requires a launch value")
        if not hasattr(runner, "run"):
            raise TypeError("D1 external analysis adapter requires an attested command runner")
        self._launch = launch
        self._runner = runner

    def run(self, request: D1AnalysisRunnerRequest) -> D1AnalysisRunnerResult:
        if type(request) is not D1AnalysisRunnerRequest:
            raise TypeError("D1 external analysis adapter requires D1AnalysisRunnerRequest")
        if any(self._launch.output_path.iterdir()):
            raise ValueError("D1 external analysis output_path must be empty before execution")
        envelope = D1ExternalAnalysisRequestEnvelope(
            request.analysis_request_digest,
            request.isolation.request_digest,
        )
        request_bytes = encode_d1_external_analysis_request(envelope)
        argv = _argv(self._launch)
        try:
            _write_request(self._launch.request_path, request_bytes)
            try:
                observed = self._runner.run(argv, cwd=None, stdin=None)
            except Exception as error:
                _fail(str(error) or repr(error), error)
            if type(observed) is not D1ExternalAnalysisCommandResult:
                _fail("runner returned an unsupported command observation")
            if observed.invoked_argv != argv:
                _fail("observed argv differs from the fixed isolation command")
            if observed.returncode != 0:
                _fail(f"worker returned nonzero exit code {observed.returncode}")
            expected_entries = {_ANALYSIS_NAME, _RESPONSE_NAME}
            if {entry.name for entry in self._launch.output_path.iterdir()} != expected_entries:
                _fail("writable output mount contains unexpected or missing files")
            analysis_bytes = _read_stable(self._launch.analysis_path, label="analysis output")
            response_bytes = _read_stable(self._launch.response_path, label="execution receipt")
            response = decode_d1_external_analysis_response(response_bytes)
            if response.analysis_request_digest != request.analysis_request_digest:
                raise ValueError("D1 external analysis response request digest differs")
            if response.isolation_request_digest != request.isolation.request_digest:
                raise ValueError("D1 external analysis response isolation digest differs")
            analysis_digest = _sha256(analysis_bytes)
            if response.analysis_sha256 != analysis_digest:
                raise ValueError("D1 external analysis response analysis digest differs")
            machine = _analysis(analysis_bytes)
            if _read_stable(self._launch.analysis_path, label="analysis output") != analysis_bytes:
                raise ValueError("D1 external analysis output drifted after validation")
            if (
                _read_stable(self._launch.response_path, label="execution receipt")
                != response_bytes
            ):
                raise ValueError("D1 external analysis receipt drifted after validation")
            evidence = D1IsolationExecutionEvidence(
                command_sha256=canonical_digest({"argv": list(observed.invoked_argv)}),
                exit_code=observed.returncode,
                request_sha256=_sha256(request_bytes),
                analysis_sha256=analysis_digest,
                response_sha256=_sha256(response_bytes),
                stdout_sha256=_sha256(observed.stdout.encode()),
                stderr_sha256=_sha256(observed.stderr.encode()),
                image_digest=self._launch.image_digest,
                platform=_PLATFORM,
            )
            # Friend boundary: ordinary callers cannot mint enforced posture;
            # this adapter reaches the protected issuer only after every
            # observable process and filesystem control above has passed.
            isolation = _mint_isolation(
                request=request.isolation,
                mechanism=_MECHANISM,
                execution_evidence=evidence,
            )
            return D1AnalysisRunnerResult(machine, isolation)
        finally:
            self._launch.request_path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class D1ExternalAnalysisRunnerRequestV2:
    envelope: D1ExternalAnalysisRequestEnvelopeV2
    aggregate: D1HeterogeneousCaptureAggregate
    isolation: D1IsolationRequest

    def __post_init__(self) -> None:
        if type(self.envelope) is not D1ExternalAnalysisRequestEnvelopeV2:
            raise TypeError("D1 v2 runner request requires the v2 transport envelope")
        if type(self.aggregate) is not D1HeterogeneousCaptureAggregate:
            raise TypeError("D1 v2 runner request requires authoritative aggregate")
        if type(self.isolation) is not D1IsolationRequest:
            raise TypeError("D1 v2 runner request requires isolation policy")
        if self.envelope.isolation_request_digest != self.isolation.request_digest:
            raise ValueError("D1 v2 runner request isolation differs from transport")
        if self.envelope.input_control.aggregate.manifest_digest != self.aggregate.manifest_digest:
            raise ValueError("D1 v2 runner aggregate differs from verified input")


@dataclass(frozen=True, slots=True)
class D1ExternalAnalysisRunnerResultV2:
    analysis: D1VerifiedMachineAnalysis
    isolation: D1ExternalIsolationAttestation
    worker_response: D1ExternalAnalysisWorkerResponseV2
    request: D1ExternalAnalysisRequestEnvelopeV2


class DockerD1ExternalAnalysisAdapterV2:
    """Run the v2 worker and require final machine-analysis verification."""

    def __init__(
        self,
        *,
        launch: D1ExternalAnalysisLaunch,
        runner: _AttestedCommandRunner,
        output_access: D1OutputAccessPort,
    ) -> None:
        if type(launch) is not D1ExternalAnalysisLaunch:
            raise TypeError("D1 v2 external adapter requires a launch value")
        if not hasattr(runner, "run"):
            raise TypeError("D1 v2 external adapter requires an attested command runner")
        if not hasattr(output_access, "grant") or not hasattr(output_access, "revoke"):
            raise TypeError("D1 v2 external adapter requires output access capability")
        self._launch = launch
        self._runner = runner
        self._output_access = output_access

    def run(self, request: D1ExternalAnalysisRunnerRequestV2) -> D1ExternalAnalysisRunnerResultV2:
        if type(request) is not D1ExternalAnalysisRunnerRequestV2:
            raise TypeError("D1 v2 adapter requires its nominal request")
        if any(self._launch.output_path.iterdir()):
            raise ValueError("D1 v2 output_path must be empty before execution")
        if not self._launch.reader_path.is_dir():
            raise ValueError("D1 v2 reader_path must be a directory")
        request_bytes = encode_d1_external_analysis_request_v2(request.envelope)
        argv = _argv_v2(self._launch)
        access_granted = False
        try:
            self._output_access.grant(self._launch.output_path, uid=65532, gid=65532)
            access_granted = True
            _write_request(self._launch.request_path, request_bytes)
            try:
                observed = self._runner.run(argv, cwd=None, stdin=None)
            except Exception as error:
                _fail(str(error) or repr(error), error)
            if type(observed) is not D1ExternalAnalysisCommandResult:
                _fail("v2 runner returned an unsupported command observation")
            if observed.invoked_argv != argv:
                _fail("v2 observed argv differs from the fixed isolation command")
            if observed.returncode != 0:
                _fail(f"v2 worker returned nonzero exit code {observed.returncode}")
            expected_entries = {_ANALYSIS_NAME, _WORKER_RESPONSE_NAME}
            if {entry.name for entry in self._launch.output_path.iterdir()} != expected_entries:
                _fail("v2 writable output contains unexpected or missing files")
            analysis_bytes = _read_stable(self._launch.analysis_path, label="analysis output")
            response_bytes = _read_stable(
                self._launch.worker_response_path,
                label="worker response",
            )
            response = decode_d1_external_analysis_worker_response_v2(response_bytes)
            if response.analysis_request_digest != request.envelope.analysis_request_digest:
                raise ValueError("D1 v2 worker response request digest differs")
            if response.isolation_request_digest != request.isolation.request_digest:
                raise ValueError("D1 v2 worker response isolation digest differs")
            if response.input_manifest_sha256 != request.envelope.input_manifest_sha256:
                raise ValueError("D1 v2 worker response input digest differs")
            analysis_digest = _sha256(analysis_bytes)
            if response.analysis_sha256 != analysis_digest:
                raise ValueError("D1 v2 worker response analysis digest differs")
            binding = bind_d1_machine_analysis_contract(
                analysis_bytes,
                expected=request.envelope.input_control,
            )
            verified = verify_d1_machine_analysis(
                binding=binding,
                aggregate=request.aggregate,
            )
            if _read_stable(self._launch.analysis_path, label="analysis output") != analysis_bytes:
                raise ValueError("D1 v2 analysis output drifted after validation")
            if (
                _read_stable(self._launch.worker_response_path, label="worker response")
                != response_bytes
            ):
                raise ValueError("D1 v2 worker response drifted after validation")
            evidence = D1IsolationExecutionEvidence(
                command_sha256=canonical_digest({"argv": list(observed.invoked_argv)}),
                exit_code=observed.returncode,
                request_sha256=_sha256(request_bytes),
                analysis_sha256=analysis_digest,
                response_sha256=_sha256(response_bytes),
                stdout_sha256=_sha256(observed.stdout.encode()),
                stderr_sha256=_sha256(observed.stderr.encode()),
                image_digest=self._launch.image_digest,
                platform=_PLATFORM,
            )
            isolation = _mint_isolation(
                request=request.isolation,
                mechanism=_MECHANISM_V2,
                execution_evidence=evidence,
            )
            return D1ExternalAnalysisRunnerResultV2(
                verified,
                isolation,
                response,
                request.envelope,
            )
        finally:
            self._launch.request_path.unlink(missing_ok=True)
            if access_granted:
                self._output_access.revoke(
                    self._launch.output_path,
                    uid=65532,
                    gid=65532,
                )
