"""Pod-side D1 worker binds process reports to the real trainer services."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from d1_resume_fixtures import captured_step_one_case, origin, resume_start
from d1_trainer_fixtures import DeterministicTrainerClock, micro_plan, run_start
from entropic_substrate.d1_artifact_snapshot import (
    D1ArtifactSnapshotError,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.d1_operator_evidence import d1_runtime_observation_sha256
from entropic_substrate.d1_operator_values import D1RuntimeObservation
from entropic_substrate.d1_trainer_process_contracts import (
    d1_trainer_request_path,
    d1_trainer_runtime_report_path,
    d1_trainer_start_report_path,
    parse_d1_runtime_report,
    parse_d1_start_report,
)
from entropic_substrate.d1_trainer_process_worker import D1TrainerProcessWorker
from entropic_substrate.d1_trainer_request_codec import (
    D1ResumeTrainerRequest,
    D1TrainerLocationSpec,
    D1TrainerRequest,
    build_d1_trainer_request,
    d1_trainer_command_sha256,
    d1_trainer_request_sha256,
    serialize_d1_trainer_request,
    verify_d1_trainer_request_runtime,
)
from entropic_substrate.experiments.d1_primitives import D1Arm

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
) -> D1TrainerProcessWorker:
    measured = _runtime(request) if runtime is None else runtime
    return D1TrainerProcessWorker(
        artifact_root=root,
        catalog=(request.plan.profile,),
        runtime_probe=lambda _pod_id: measured,
        process_identity=lambda: process_id,
        observer=_Observer(),
        clock=DeterministicTrainerClock(),
    )


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
    worker = _worker(tmp_path, request)
    worker.bootstrap(_POD, _ATTEMPT)
    _stage(tmp_path, request)
    request_sha = d1_trainer_request_sha256(request)

    with pytest.raises(FileExistsError, match="checkpoint target already exists"):
        worker.start_trainer(_POD, _ATTEMPT, request_sha)

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
