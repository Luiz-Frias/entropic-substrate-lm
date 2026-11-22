"""Pod-side D1 worker binds process reports to the real trainer services."""

from __future__ import annotations

import hashlib
import io
import json
import os
import traceback
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import d1_launch_admission_fixtures as launch
from d1_resume_fixtures import captured_step_one_case, origin, resume_start
from d1_trainer_fixtures import DeterministicTrainerClock, micro_plan, run_start
from entropic_substrate.composition.entrypoints.d1_trainer_process_worker import (
    D1TrainerProcessWorker,
)
from entropic_substrate.composition.entrypoints.d1_trainer_worker_command import (
    run_d1_trainer_worker_command,
)
from entropic_substrate.kernel.artifacts import (
    D1ArtifactSnapshotError,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.run.campaign.d1_operator_evidence import (
    d1_runtime_observation_sha256,
)
from entropic_substrate.ladder.run.campaign.d1_operator_values import D1RuntimeObservation
from entropic_substrate.ladder.run.provider.d1_system_runtime_probe import (
    D1RuntimeProbeError,
    SystemD1RuntimeProbe,
)
from entropic_substrate.ladder.run.transport.d1_trainer_process_contracts import (
    d1_trainer_request_path,
    d1_trainer_runtime_report_path,
    d1_trainer_start_report_path,
    parse_d1_runtime_report,
    parse_d1_start_report,
)
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    D1ResumeTrainerRequest,
    D1TrainerLocationSpec,
    D1TrainerRequest,
    build_d1_trainer_request,
    d1_trainer_command_sha256,
    d1_trainer_request_sha256,
    serialize_d1_trainer_request,
    verify_d1_trainer_request_runtime,
)

pytestmark = pytest.mark.unit

_POD = "pod-worker-123"
_ATTEMPT = "a" * 64
_PROCESS = "trainer-process-123"


class _Observer:
    def observe(self, _event: object) -> None:
        pass


def _runtime(request: D1TrainerRequest) -> D1RuntimeObservation:
    start = request.start
    return D1RuntimeObservation(
        _POD,
        start.trainer_started_at,
        start.cuda_version,
        start.driver_version,
        start.pytorch_version,
        start.compiler_enabled,
        start.compiler_version,
        request.origin.monotonic_ns,
    )


def _fresh_request(root: Path) -> D1TrainerRequest:
    plan = micro_plan(D1Arm.BOTH)
    request = build_d1_trainer_request(
        plan,
        run_start(plan),
        origin=origin(),
    )
    (root / Path(request.locations.journal_path).parent).mkdir(parents=True)
    return request


def _stage(root: Path, request: D1TrainerRequest, *, attempt: str = _ATTEMPT) -> None:
    relative = d1_trainer_request_path(attempt)
    (root / Path(relative).parent).mkdir(parents=True, exist_ok=True)
    publish_d1_atomic_bytes(
        root,
        relative,
        serialize_d1_trainer_request(request),
        label="staged trainer request",
    )


def _worker(
    root: Path,
    request: D1TrainerRequest,
    *,
    runtime: D1RuntimeObservation | None = None,
    process_id: str = _PROCESS,
    checkpoint_sync_factory=None,  # type: ignore[no-untyped-def]
    start_report_publisher=None,  # type: ignore[no-untyped-def]
) -> D1TrainerProcessWorker:
    measured = _runtime(request) if runtime is None else runtime
    return D1TrainerProcessWorker(
        artifact_root=root,
        catalog=(request.plan.profile,),
        runtime_probe=lambda _pod_id: measured,
        process_identity=lambda: process_id,
        observer=_Observer(),
        clock=DeterministicTrainerClock(),
        checkpoint_sync_factory=checkpoint_sync_factory,
        start_report_publisher=start_report_publisher,
    )


class _Sync:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def sync(self, **kwargs: object) -> None:
        self.steps.append(kwargs["successful_step"])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    "trainer_started_at cuda_version driver_version pytorch_version compiler monotonic".split(),
)
def test_shared_runtime_binding_refuses_every_request_side_drift(
    tmp_path: Path, field: str
) -> None:
    request = _fresh_request(tmp_path)
    runtime = _runtime(request)
    changes: dict[str, object] = {
        "trainer_started_at": runtime.trainer_started_at + timedelta(seconds=1),
        "cuda_version": "foreign-cuda",
        "driver_version": "foreign-driver",
        "pytorch_version": "foreign-pytorch",
        "trainer_monotonic_origin_ns": runtime.trainer_monotonic_origin_ns + 1,
    }
    key = "trainer_monotonic_origin_ns" if field == "monotonic" else field
    mutation = (
        {"compiler_enabled": True, "compiler_version": "foreign-compiler"}
        if field == "compiler"
        else {key: changes[key]}
    )
    with pytest.raises(ValueError, match="differs from attested runtime"):
        verify_d1_trainer_request_runtime(request, replace(runtime, **mutation))


def test_process_report_paths_are_deterministic_and_attempt_bound() -> None:
    request_sha = "b" * 64
    assert d1_trainer_runtime_report_path(_ATTEMPT) == f"trainer-runtime-{_ATTEMPT}.json"
    assert d1_trainer_start_report_path(_ATTEMPT, request_sha) == (
        f"trainer-start-{_ATTEMPT}-{request_sha}.json"
    )
    with pytest.raises(ValueError):
        d1_trainer_runtime_report_path("invalid")
    with pytest.raises(ValueError):
        d1_trainer_start_report_path(_ATTEMPT, "invalid")


def test_bootstrap_publishes_one_canonical_runtime_report_without_overwrite(
    tmp_path: Path,
) -> None:
    request = _fresh_request(tmp_path)
    worker = _worker(tmp_path, request)

    identity = worker.bootstrap(_POD, _ATTEMPT)
    path = d1_trainer_runtime_report_path(_ATTEMPT)
    observed_path, payload, digest = read_d1_artifact_bytes(
        tmp_path,
        path,
        label="runtime report",
    )

    assert identity.path == str(observed_path)
    assert identity.content_hash.digest == digest
    assert parse_d1_runtime_report(payload, pod_id=_POD, attempt_sha256=_ATTEMPT) == _runtime(
        request
    )
    with pytest.raises(D1ArtifactSnapshotError, match="occupied"):
        worker.bootstrap(_POD, _ATTEMPT)
    assert read_d1_artifact_bytes(tmp_path, path, label="runtime report")[1] == payload


def test_start_runs_real_fresh_service_after_durable_start_report_and_refuses_repeat(
    tmp_path: Path,
) -> None:
    request = _fresh_request(tmp_path)
    worker = _worker(tmp_path, request)
    runtime = _runtime(request)
    request_sha = d1_trainer_request_sha256(request)
    worker.bootstrap(_POD, _ATTEMPT)
    _stage(tmp_path, request)

    result = worker.start_trainer(_POD, _ATTEMPT, request_sha)

    assert result["handoff_kind"] == "fresh_completion"
    assert result["successful_step"] == 2
    assert (tmp_path / request.locations.journal_path).is_file()
    assert (tmp_path / request.locations.checkpoint_path).is_file()
    start_path = d1_trainer_start_report_path(_ATTEMPT, request_sha)
    _, start_payload, _ = read_d1_artifact_bytes(tmp_path, start_path, label="start report")
    assert (
        parse_d1_start_report(
            start_payload,
            pod_id=_POD,
            attempt_sha256=_ATTEMPT,
            request_sha256=request_sha,
            runtime_sha256=d1_runtime_observation_sha256(runtime),
            trainer_started_at=request.start.trainer_started_at,
            command_sha256=d1_trainer_command_sha256(request),
        ).process_id
        == _PROCESS
    )
    marker = Path(result["marker"]["path"])  # type: ignore[index]
    marker_bytes = marker.read_bytes()
    with pytest.raises(D1ArtifactSnapshotError, match="occupied"):
        worker.start_trainer(_POD, _ATTEMPT, request_sha)
    assert marker.read_bytes() == marker_bytes


def test_smoke_worker_routes_every_checkpoint_through_bound_sync(tmp_path: Path) -> None:
    request = _fresh_request(tmp_path)
    sync = _Sync()
    bindings: list[tuple[str, str, str]] = []

    def factory(pod_id: str, attempt_sha256: str, logical_run_id: str) -> _Sync:
        bindings.append((pod_id, attempt_sha256, logical_run_id))
        return sync

    worker = _worker(tmp_path, request, checkpoint_sync_factory=factory)
    worker.bootstrap(_POD, _ATTEMPT)
    _stage(tmp_path, request)
    worker.start_trainer(_POD, _ATTEMPT, d1_trainer_request_sha256(request))

    assert bindings == [(_POD, _ATTEMPT, request.plan.logical_run_id)]
    assert sync.steps == [1, 2]


def test_confirmatory_worker_refuses_missing_remote_sync_before_start_report(
    tmp_path: Path,
) -> None:
    profile = launch.profiles()[0]
    plan = launch.plan(profile, 101)
    start = launch.start(plan)
    request = build_d1_trainer_request(
        plan,
        start,
        replace(origin(), started_at=start.trainer_started_at),
    )
    root = tmp_path / Path(request.locations.journal_path).parent
    root.mkdir(parents=True)
    worker = _worker(tmp_path, request)
    worker.bootstrap(_POD, _ATTEMPT)
    _stage(tmp_path, request)
    request_sha = d1_trainer_request_sha256(request)

    with pytest.raises(ValueError, match="remote checkpoint sync"):
        worker.start_trainer(_POD, _ATTEMPT, request_sha)

    assert not (tmp_path / d1_trainer_start_report_path(_ATTEMPT, request_sha)).exists()


@pytest.mark.parametrize("mismatch", ["request_sha", "runtime"])
def test_start_refuses_request_digest_or_runtime_mismatch_before_report_and_training(
    tmp_path: Path,
    mismatch: str,
) -> None:
    request = _fresh_request(tmp_path)
    runtime = _runtime(request)
    if mismatch == "runtime":
        runtime = replace(runtime, pytorch_version="foreign-runtime")
    worker = _worker(tmp_path, request, runtime=runtime)
    worker.bootstrap(_POD, _ATTEMPT)
    _stage(tmp_path, request)
    expected_sha = d1_trainer_request_sha256(request)
    supplied_sha = "f" * 64 if mismatch == "request_sha" else expected_sha

    with pytest.raises(ValueError, match=r"request (digest|differs from attested runtime)"):
        worker.start_trainer(_POD, _ATTEMPT, supplied_sha)

    assert not (tmp_path / d1_trainer_start_report_path(_ATTEMPT, supplied_sha)).exists()
    assert not (tmp_path / request.locations.journal_path).exists()
    assert not (tmp_path / request.locations.checkpoint_path).exists()


def test_start_report_precedes_real_service_occupied_output_refusal(tmp_path: Path) -> None:
    request = _fresh_request(tmp_path)
    checkpoint = tmp_path / request.locations.checkpoint_path
    checkpoint.write_bytes(b"occupied-before-worker")
    reports: list[tuple[str, bytes]] = []
    worker = _worker(
        tmp_path,
        request,
        start_report_publisher=lambda path, payload: reports.append((path, payload)),
    )
    worker.bootstrap(_POD, _ATTEMPT)
    _stage(tmp_path, request)
    request_sha = d1_trainer_request_sha256(request)

    with pytest.raises(FileExistsError, match="checkpoint target already exists"):
        worker.start_trainer(_POD, _ATTEMPT, request_sha)

    assert reports == [
        (
            d1_trainer_start_report_path(_ATTEMPT, request_sha),
            (tmp_path / d1_trainer_start_report_path(_ATTEMPT, request_sha)).read_bytes(),
        )
    ]
    assert (tmp_path / d1_trainer_start_report_path(_ATTEMPT, request_sha)).is_file(), (
        "start report must be durable before the real service is invoked"
    )
    assert checkpoint.read_bytes() == b"occupied-before-worker"
    assert not (tmp_path / request.locations.journal_path).exists()


def test_invalid_process_identity_stops_before_report_or_training(tmp_path: Path) -> None:
    request = _fresh_request(tmp_path)
    worker = _worker(tmp_path, request, process_id=" ")
    worker.bootstrap(_POD, _ATTEMPT)
    _stage(tmp_path, request)
    request_sha = d1_trainer_request_sha256(request)
    with pytest.raises(ValueError, match="process_id"):
        worker.start_trainer(_POD, _ATTEMPT, request_sha)
    assert not (tmp_path / d1_trainer_start_report_path(_ATTEMPT, request_sha)).exists()
    assert not (tmp_path / request.locations.journal_path).exists()


def test_start_runs_real_resume_service_and_preserves_export_marker_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    case = captured_step_one_case(interrupted, monkeypatch)
    start = resume_start(case.plan, case.checkpoint_sha256)
    request = D1ResumeTrainerRequest(
        case.plan,
        start,
        origin(),
        D1TrainerLocationSpec(
            case.journal_path.relative_to(tmp_path).as_posix(),
            case.checkpoint_path.relative_to(tmp_path).as_posix(),
            "interrupted/quarantined-tail.jsonl",
        ),
    )
    worker = _worker(tmp_path, request)
    request_sha = d1_trainer_request_sha256(request)
    worker.bootstrap(_POD, _ATTEMPT)
    _stage(tmp_path, request)

    result = worker.start_trainer(_POD, _ATTEMPT, request_sha)

    assert result["handoff_kind"] == "resume_completion"
    assert result["successful_step"] == 2
    prefix = tmp_path / request.locations.checkpoint_path
    assert Path(f"{prefix}.handoff.json").is_file()
    assert Path(f"{prefix}.complete.json").is_file()
    assert result["marker"]["path"] == str(Path(f"{prefix}.complete.json"))  # type: ignore[index]


def _torch_runtime(*, cuda_available: bool, cuda_version: str | None) -> object:
    return SimpleNamespace(
        __version__="2.13.0+test",
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        version=SimpleNamespace(cuda=cuda_version),
    )


class _Nvml:
    def __init__(self, driver: object = b"560.35.03") -> None:
        self.driver = driver
        self.calls: list[str] = []

    def nvmlInit(self) -> None:
        self.calls.append("init")

    def nvmlSystemGetDriverVersion(self) -> object:
        self.calls.append("driver")
        return self.driver

    def nvmlShutdown(self) -> None:
        self.calls.append("shutdown")


def test_system_runtime_probe_measures_cpu_boundary_without_loading_nvml() -> None:
    measured_at = datetime(2026, 8, 26, 18, 0, tzinfo=UTC)
    calls: list[str] = []

    def unavailable_nvml() -> object:
        raise AssertionError("CPU probe must not import NVML")

    probe = SystemD1RuntimeProbe(
        torch_runtime=_torch_runtime(cuda_available=False, cuda_version=None),
        nvml_loader=unavailable_nvml,
        utc_now=lambda: calls.append("utc") or measured_at,
        monotonic_ns=lambda: calls.append("monotonic") or 123_456,
    )

    runtime = probe(_POD)

    assert calls == ["utc", "monotonic"]
    assert runtime.trainer_started_at == measured_at
    assert runtime.trainer_monotonic_origin_ns == 123_456
    assert runtime.pytorch_version == "2.13.0+test"
    assert runtime.cuda_version == "not-available"
    assert runtime.driver_version == "not-available"
    assert runtime.compiler_enabled is False
    assert runtime.compiler_version == "disabled"


def test_system_runtime_probe_normalizes_torch_version_string_subclass() -> None:
    version_type = type("TorchVersion", (str,), {})
    torch_runtime = _torch_runtime(cuda_available=False, cuda_version=None)
    torch_runtime.__version__ = version_type("2.13.0+cpu")  # type: ignore[attr-defined]
    probe = SystemD1RuntimeProbe(
        torch_runtime=torch_runtime,
        utc_now=lambda: datetime(2026, 8, 26, 18, 0, tzinfo=UTC),
        monotonic_ns=lambda: 1,
    )

    runtime = probe(_POD)

    assert runtime.pytorch_version == "2.13.0+cpu"
    assert type(runtime.pytorch_version) is str


def test_system_runtime_probe_measures_cuda_and_nvml_driver_with_shutdown() -> None:
    nvml = _Nvml()
    probe = SystemD1RuntimeProbe(
        torch_runtime=_torch_runtime(cuda_available=True, cuda_version="12.6"),
        nvml_loader=lambda: nvml,
        utc_now=lambda: datetime(2026, 8, 26, 18, 0, tzinfo=UTC),
        monotonic_ns=lambda: 789,
    )

    runtime = probe(_POD)

    assert runtime.cuda_version == "12.6"
    assert runtime.driver_version == "560.35.03"
    assert nvml.calls == ["init", "driver", "shutdown"]


@pytest.mark.parametrize("missing", ["cuda_version", "nvml"])
def test_system_runtime_probe_fails_loud_when_cuda_facts_are_unavailable(missing: str) -> None:
    nvml = _Nvml()

    def load_nvml() -> object:
        if missing == "nvml":
            raise ModuleNotFoundError("secret-nvml-import-detail")
        return nvml

    probe = SystemD1RuntimeProbe(
        torch_runtime=_torch_runtime(
            cuda_available=True,
            cuda_version=None if missing == "cuda_version" else "12.6",
        ),
        nvml_loader=load_nvml,
        utc_now=lambda: datetime(2026, 8, 26, 18, 0, tzinfo=UTC),
        monotonic_ns=lambda: 789,
    )

    with pytest.raises(D1RuntimeProbeError, match="runtime probe refused") as caught:
        probe(_POD)
    assert caught.value.code == f"{missing}_unavailable"
    assert caught.value.__cause__ is None
    assert "secret-nvml-import-detail" not in str(caught.value)
    assert "secret-nvml-import-detail" not in "".join(traceback.format_exception(caught.value))


def _canonical_output(stream: io.StringIO) -> dict[str, object]:
    raw = stream.getvalue()
    assert raw.endswith("\n") and raw.count("\n") == 1
    value = json.loads(raw)
    assert raw == json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    return value


def _run_command(
    argv: tuple[str, ...],
    *,
    request: D1TrainerRequest,
    stdout: io.StringIO,
    stderr: io.StringIO,
) -> int:
    return run_d1_trainer_worker_command(
        argv,
        catalog=(request.plan.profile,),
        runtime_probe=lambda _pod_id: _runtime(request),
        clock=DeterministicTrainerClock(),
        stdout=stdout,
        stderr=stderr,
    )


def test_internal_worker_command_bootstrap_and_foreground_start_real_tiny_trainer(
    tmp_path: Path,
) -> None:
    request = _fresh_request(tmp_path)
    request_sha = d1_trainer_request_sha256(request)
    stdout, stderr = io.StringIO(), io.StringIO()
    common = (
        "--artifact-root",
        str(tmp_path),
        "--pod-id",
        _POD,
        "--attempt-sha256",
        _ATTEMPT,
    )

    assert _run_command(("bootstrap", *common), request=request, stdout=stdout, stderr=stderr) == 0
    bootstrap = _canonical_output(stdout)
    assert bootstrap["status"] == "success"
    assert stderr.getvalue() == ""
    _stage(tmp_path, request)
    stdout = io.StringIO()

    assert (
        _run_command(
            ("start", *common, "--request-sha256", request_sha),
            request=request,
            stdout=stdout,
            stderr=stderr,
        )
        == 0
    )
    started = _canonical_output(stdout)
    assert started["status"] == "success"
    assert (tmp_path / request.locations.journal_path).is_file()
    assert (tmp_path / request.locations.checkpoint_path).is_file()
    assert Path(f"{tmp_path / request.locations.checkpoint_path}.handoff.json").is_file()
    assert Path(f"{tmp_path / request.locations.checkpoint_path}.complete.json").is_file()
    _, payload, _ = read_d1_artifact_bytes(
        tmp_path,
        d1_trainer_start_report_path(_ATTEMPT, request_sha),
        label="start report",
    )
    assert parse_d1_start_report(
        payload,
        pod_id=_POD,
        attempt_sha256=_ATTEMPT,
        request_sha256=request_sha,
        runtime_sha256=d1_runtime_observation_sha256(_runtime(request)),
        trainer_started_at=request.start.trainer_started_at,
        command_sha256=d1_trainer_command_sha256(request),
    ).process_id == str(os.getpid())


def test_internal_worker_command_emits_typed_canonical_error_without_raw_request(
    tmp_path: Path,
) -> None:
    request = _fresh_request(tmp_path)
    stdout, stderr = io.StringIO(), io.StringIO()
    common = (
        "--artifact-root",
        str(tmp_path),
        "--pod-id",
        _POD,
        "--attempt-sha256",
        _ATTEMPT,
    )
    assert _run_command(("bootstrap", *common), request=request, stdout=stdout, stderr=stderr) == 0
    relative = d1_trainer_request_path(_ATTEMPT)
    (tmp_path / Path(relative).parent).mkdir(parents=True)
    raw = b"secret-request-sentinel"
    publish_d1_atomic_bytes(tmp_path, relative, raw, label="malformed request")
    stdout, stderr = io.StringIO(), io.StringIO()

    exit_code = _run_command(
        ("start", *common, "--request-sha256", hashlib.sha256(raw).hexdigest()),
        request=request,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 21
    error = _canonical_output(stderr)
    assert error["status"] == "error"
    assert error["error"]["code"] == "worker_refused"  # type: ignore[index]
    assert stdout.getvalue() == ""
    assert "secret-request-sentinel" not in stderr.getvalue()


@pytest.mark.parametrize(
    ("kind", "expected_exit", "expected_code"),
    [
        ("argv", 21, "worker_refused"),
        ("runtime", 20, "runtime-sentinel"),
        ("defect", 70, "unexpected_defect"),
    ],
)
def test_internal_worker_command_preserves_failure_classes(
    tmp_path: Path, kind: str, expected_exit: int, expected_code: str
) -> None:
    stdout, stderr = io.StringIO(), io.StringIO()
    common = ("--artifact-root", str(tmp_path), "--pod-id", _POD, "--attempt-sha256", _ATTEMPT)
    argv = ("unknown",) if kind == "argv" else ("bootstrap", *common)
    error: Exception = (
        D1RuntimeProbeError("runtime-sentinel") if kind == "runtime" else ValueError("deep defect")
    )

    def probe(_pod_id: str) -> D1RuntimeObservation:
        raise error

    exit_code = run_d1_trainer_worker_command(
        argv,
        catalog=(micro_plan(D1Arm.NONE).profile,),
        runtime_probe=probe,
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == expected_exit
    assert _canonical_output(stderr)["error"]["code"] == expected_code  # type: ignore[index]
    assert stdout.getvalue() == ""


_CANONICAL_FIXTURES: tuple[dict[str, object], ...] = (
    {},
    {"b": 1, "a": [1, 2.5, None, True], "c": {"z": "x", "y": []}},
    {"unicode": "café ☃ \U0001f600", "nested": [[1], [2, [3]]]},
    {"n": -0.0, "big": 10**30, "small": 1e-9, "quote": 'a"b\\c\n'},
)


@pytest.mark.parametrize("payload", _CANONICAL_FIXTURES)
def test_worker_command_emit_equals_the_parent_canonical_line(payload: dict[str, object]) -> None:
    import entropic_substrate.composition.entrypoints.d1_trainer_worker_command as worker_command

    stream = io.StringIO()
    worker_command._emit(stream, payload)  # type: ignore[attr-defined]
    # Oracle: the parent `_canonical` expression, verbatim.
    parent = json.dumps(
        dict(payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert stream.getvalue() == parent + "\n"


def test_worker_command_uses_the_shared_canonical_json_not_a_local_dump() -> None:
    import entropic_substrate.composition.entrypoints.d1_trainer_worker_command as worker_command
    from entropic_substrate.kernel.identity import d1_primitives

    assert worker_command.canonical_json is d1_primitives.canonical_json  # type: ignore[attr-defined]
    assert "json.dumps" not in Path(worker_command.__file__).read_text(encoding="utf-8")


def test_worker_command_emit_refuses_nonfinite_values() -> None:
    import entropic_substrate.composition.entrypoints.d1_trainer_worker_command as worker_command

    stream = io.StringIO()
    with pytest.raises(ValueError, match="Out of range float"):
        worker_command._emit(stream, {"x": float("nan")})  # type: ignore[attr-defined]
    assert stream.getvalue() == ""
