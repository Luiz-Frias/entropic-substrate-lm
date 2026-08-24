"""Hermetic guard teeth for the enforced external D1 analysis adapter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from entropic_substrate.analysis.d1_execution_receipt import (
    D1AnalysisRunnerRequest,
    D1ExecutorKind,
    D1IsolationRequest,
)
from entropic_substrate.analysis.d1_external_analysis_adapter import (
    D1ExternalAnalysisCommandResult,
    D1ExternalAnalysisLaunch,
    DockerD1ExternalAnalysisAdapter,
)
from entropic_substrate.analysis.d1_external_analysis_codec import (
    D1ExternalAnalysisResponseEnvelope,
    decode_d1_external_analysis_request,
    encode_d1_external_analysis_response,
)

from .test_d1_aliased_analysis import _packet

pytestmark = pytest.mark.unit

_IMAGE = "registry.example/entropic-substrate-lm@sha256:" + "a" * 64


@dataclass
class _Runner:
    launch: D1ExternalAnalysisLaunch
    returncode: int = 0
    argv_mutation: tuple[str, ...] = ()
    response_mutation: dict[str, str] = field(default_factory=dict)
    analysis_bytes: bytes | None = None
    output_drift: bool = False
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        stdin: str | None,
    ) -> D1ExternalAnalysisCommandResult:
        assert cwd is None
        assert stdin is None
        self.calls.append(argv)
        request = decode_d1_external_analysis_request(self.launch.request_path.read_bytes())
        analysis = self.analysis_bytes or _packet().canonical_json.encode()
        self.launch.analysis_path.write_bytes(analysis)
        response = D1ExternalAnalysisResponseEnvelope(
            analysis_request_digest=request.analysis_request_digest,
            isolation_request_digest=request.isolation_request_digest,
            analysis_sha256=hashlib.sha256(analysis).hexdigest(),
        )
        for key, value in self.response_mutation.items():
            response = replace(response, **{key: value})
        self.launch.response_path.write_bytes(encode_d1_external_analysis_response(response))
        if self.output_drift:
            self.launch.analysis_path.write_bytes(analysis + b"\n")
        return D1ExternalAnalysisCommandResult(
            invoked_argv=argv + self.argv_mutation,
            returncode=self.returncode,
            stdout="worker stdout",
            stderr="worker stderr",
        )


def _launch(tmp_path: Path) -> D1ExternalAnalysisLaunch:
    tmp_path.mkdir(parents=True, exist_ok=True)
    capture = tmp_path / "capture"
    capture.mkdir()
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text("{}", encoding="utf-8")
    reader = tmp_path / "reader"
    reader.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    return D1ExternalAnalysisLaunch(
        image_reference=_IMAGE,
        capture_path=capture,
        preregistration_path=preregistration,
        reader_path=reader,
        output_path=output,
    )


def _run(tmp_path: Path, **runner_options: object):
    launch = _launch(tmp_path)
    runner = _Runner(launch, **runner_options)
    result = DockerD1ExternalAnalysisAdapter(launch=launch, runner=runner).run(
        D1AnalysisRunnerRequest("1" * 64, D1IsolationRequest())
    )
    return launch, runner, result


def test_adapter_uses_one_exact_hardened_linux_amd64_container_invocation(
    tmp_path: Path,
) -> None:
    launch, runner, result = _run(tmp_path)

    assert result.analysis.canonical_json == _packet().canonical_json
    assert result.isolation.executor_kind is D1ExecutorKind.EXTERNAL_ENFORCED
    assert result.isolation.execution_evidence_digest is not None
    assert len(runner.calls) == 1
    argv = runner.calls[0]
    assert argv[:3] == ("docker", "run", "--rm")
    assert ("--platform", "linux/amd64") == argv[
        argv.index("--platform") : argv.index("--platform") + 2
    ]
    for flag, value in (
        ("--network", "none"),
        ("--user", "65532:65532"),
        ("--cpus", "2"),
        ("--memory", "4g"),
        ("--pids-limit", "128"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
    ):
        assert (flag, value) == argv[argv.index(flag) : argv.index(flag) + 2]
    assert "--read-only" in argv
    assert (
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",
    ) == argv[argv.index("--tmpfs") : argv.index("--tmpfs") + 2]
    mounts = [argv[index + 1] for index, item in enumerate(argv) if item == "--mount"]
    assert len(mounts) == 5
    assert sum("readonly" not in mount for mount in mounts) == 1
    assert mounts[-1].endswith("dst=/d1/output")
    assert "--env" not in argv and "-e" not in argv and "--env-file" not in argv
    assert argv[-9:] == (
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
    assert not launch.request_path.exists()


@pytest.mark.parametrize(
    ("image", "message"),
    [
        ("registry.example/model:latest", "immutable"),
        ("registry.example/model:tag@sha256:" + "a" * 64, "tag"),
        ("registry.example/model@sha256:not-a-digest", "digest"),
    ],
)
def test_launch_rejects_mutable_or_malformed_image_identity(
    tmp_path: Path, image: str, message: str
) -> None:
    launch = _launch(tmp_path)
    with pytest.raises(ValueError, match=message):
        replace(launch, image_reference=image)


@pytest.mark.parametrize(
    "mutation",
    [
        ("--network", "bridge"),
        ("--mount", "type=bind,src=/host,dst=/extra"),
        ("--env", "MLFLOW_TRACKING_URI=https://tracking.example"),
        ("--read-write",),
    ],
)
def test_adapter_rejects_observed_argv_drift_including_network_mount_env_or_write_access(
    tmp_path: Path, mutation: tuple[str, ...]
) -> None:
    with pytest.raises(RuntimeError, match="argv"):
        _run(tmp_path, argv_mutation=mutation)


def test_adapter_rejects_nonzero_exit_or_request_response_output_drift(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="nonzero"):
        _run(tmp_path / "exit", returncode=9)
    with pytest.raises(ValueError, match="request"):
        _run(
            tmp_path / "request",
            response_mutation={"analysis_request_digest": "9" * 64},
        )
    with pytest.raises(ValueError, match="isolation"):
        _run(
            tmp_path / "isolation",
            response_mutation={"isolation_request_digest": "8" * 64},
        )
    with pytest.raises(ValueError, match="analysis"):
        _run(tmp_path / "output", output_drift=True)


def test_adapter_rejects_symlinked_inputs_or_outputs(tmp_path: Path) -> None:
    launch = _launch(tmp_path)
    linked = tmp_path / "capture-link"
    linked.symlink_to(launch.capture_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        replace(launch, capture_path=linked)

    runner = _Runner(launch)
    original = runner.run

    def symlink_output(
        argv: tuple[str, ...], *, cwd: Path | None, stdin: str | None
    ) -> D1ExternalAnalysisCommandResult:
        result = original(argv, cwd=cwd, stdin=stdin)
        analysis = launch.analysis_path.read_bytes()
        launch.analysis_path.unlink()
        target = tmp_path / "elsewhere.json"
        target.write_bytes(analysis)
        launch.analysis_path.symlink_to(target)
        return result

    runner.run = symlink_output  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="symlink"):
        DockerD1ExternalAnalysisAdapter(launch=launch, runner=runner).run(
            D1AnalysisRunnerRequest("1" * 64, D1IsolationRequest())
        )
