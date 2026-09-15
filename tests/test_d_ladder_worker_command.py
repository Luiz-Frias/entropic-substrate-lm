"""CLI composition crosses actual providers while tests substitute only IO boundaries."""

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.d1_operator_values import D1RuntimeObservation
from entropic_substrate.d1_resource_activation import UnavailableResourceActivationFactory
from entropic_substrate.d_ladder_remote import publish_remote_request
from entropic_substrate.d_ladder_remote_request import serialize_remote_request
from entropic_substrate.d_ladder_worker_command import run_worker_command
from entropic_substrate.d_ladder_worker_runtime import worker_resource_registration

from .test_d1_trainer_object_transport import _ObjectStore
from .test_d_ladder_remote_request import request_fixture

pytestmark = pytest.mark.unit


def test_missing_pod_identity_is_refused_without_exposing_environment(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = run_worker_command(
        [
            "serve",
            "--repo-root",
            str(tmp_path),
            "--workspace-root",
            str(tmp_path / "worker"),
            "--attempt-sha256",
            "c" * 64,
            "--maximum-wait-seconds",
            "1",
        ],
        environ={"SECRET": "never-emit-this"},  # pragma: allowlist secret
        objects=_ObjectStore(),
        stderr=stream,
    )
    assert code != 0 and "never-emit-this" not in stream.getvalue()


def test_serve_bootstraps_then_uses_measured_origin_and_actual_remote_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import entropic_substrate.d_ladder_worker_command as command

    source = tmp_path / "source"
    source.mkdir()
    original = request_fixture(source)
    start = replace(
        original.start,
        compiler_enabled=False,
        compiler_version="disabled",
        resource_activation=worker_resource_registration(),
    )
    request = replace(original, start=start)
    objects = _ObjectStore()
    identity = publish_remote_request(objects, request, source)
    runtime = D1RuntimeObservation(
        pod_id=request.pod_id,
        trainer_started_at=start.trainer_started_at,
        cuda_version=start.cuda_version,
        driver_version=start.driver_version,
        pytorch_version=start.pytorch_version,
        compiler_enabled=False,
        compiler_version="disabled",
        trainer_monotonic_origin_ns=1,
    )
    monkeypatch.setattr(command, "SystemD1RuntimeProbe", lambda: lambda _: runtime)
    monkeypatch.setattr(
        command, "measure_hardware", lambda: (start.accelerator_model, start.accelerator_count)
    )
    monkeypatch.setattr(command, "verify_worker_source", lambda *_: None)
    monkeypatch.setattr(
        command, "build_worker_factory", lambda _: UnavailableResourceActivationFactory()
    )
    calls = []

    def dispatch(store, selected, root, **kwargs):
        calls.append((selected, kwargs["observed_start"]))
        return identity

    monkeypatch.setattr(command, "run_remote_worker", dispatch)
    output = io.StringIO()
    assert (
        run_worker_command(
            [
                "serve",
                "--repo-root",
                str(source),
                "--workspace-root",
                str(tmp_path / "worker"),
                "--attempt-sha256",
                request.attempt_sha256,
                "--maximum-wait-seconds",
                "1",
            ],
            environ={
                "RUNPOD_POD_ID": request.pod_id,
                "D_LADDER_CODE_COMMIT": request.code_commit,
                "D_LADDER_IMAGE_DIGEST": start.image_digest,
            },
            objects=objects,
            stdout=output,
        )
        == 0
    )
    assert calls == [(identity, start)]
    bootstrap = json.loads(objects.values[f"{request.prefix}/bootstrap.json"])
    assert bootstrap["runtime"][
        "trainer_started_at"
    ] == start.trainer_started_at.isoformat().replace("+00:00", "Z")
    assert json.loads(output.getvalue())["status"] == "success"
    (source / "request.json").write_bytes(serialize_remote_request(request))
    publisher = io.StringIO()
    assert (
        run_worker_command(
            [
                "publish",
                "--repo-root",
                str(source),
                "--artifact-root",
                str(source),
                "--request-manifest",
                "request.json",
            ],
            environ={},
            objects=objects,
            stdout=publisher,
        )
        == 0
    )
    assert json.loads(publisher.getvalue())["sha256"] == request.digest


def test_bounded_wait_does_not_republish_or_synthesize_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import entropic_substrate.d_ladder_worker_command as command

    clock = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(command.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(command.time, "sleep", lambda _: None)
    from entropic_substrate.d1_trainer_mailbox_worker import D1TrainerMailboxWaitPolicy

    with pytest.raises(TimeoutError):
        command.wait_object(_ObjectStore(), "missing", D1TrainerMailboxWaitPolicy(1.0, 0.1))
