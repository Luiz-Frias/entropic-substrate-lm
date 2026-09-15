"""Hermetic command-contract tests for the D1 Docker Build Cloud adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from entropic_substrate.d1_build_cloud_adapter import DockerBuildCloudD1Adapter
from entropic_substrate.d1_operator_ports import (
    D1CommandResult,
    D1OperatorTransportError,
)
from entropic_substrate.d1_operator_values import D1ImageBuildRequest

pytestmark = pytest.mark.unit

_COMMIT = "c" * 40
_DIGEST = "sha256:" + "e" * 64


@dataclass
class _Runner:
    metadata: object = field(
        default_factory=lambda: {
            "containerimage.digest": _DIGEST,
            "containerimage.descriptor": {"digest": _DIGEST},
        }
    )
    error: Exception | None = None
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    calls: list[tuple[tuple[str, ...], Path | None, str | None]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return False

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        stdin: str | None,
    ) -> D1CommandResult:
        self.calls.append((argv, cwd, stdin))
        if self.error is not None:
            raise self.error
        metadata_at = argv.index("--metadata-file") + 1
        Path(argv[metadata_at]).write_text(json.dumps(self.metadata), encoding="utf-8")
        return D1CommandResult(stdout=self.stdout, stderr=self.stderr, returncode=self.returncode)


def _request(tmp_path: Path) -> D1ImageBuildRequest:
    context = tmp_path / "repo"
    context.mkdir()
    dockerfile = context / "Containerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    return D1ImageBuildRequest(
        builder="cloud-builder",
        repository="registry.example/entropic-substrate-lm",
        source_commit=_COMMIT,
        context=context,
        dockerfile=dockerfile,
    )


def test_build_cloud_uses_fixed_argv_pushes_amd64_and_returns_digest_identity(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    runner = _Runner()

    image = DockerBuildCloudD1Adapter(runner=runner).build_and_push(request)

    assert image.reference == f"registry.example/entropic-substrate-lm@{_DIGEST}"
    assert image.source_commit == _COMMIT
    assert len(runner.calls) == 1
    argv, cwd, stdin = runner.calls[0]
    assert argv[:4] == ("docker", "buildx", "build", "--builder")
    assert argv[4] == "cloud-builder"
    assert ("--platform", "linux/amd64") == argv[
        argv.index("--platform") : argv.index("--platform") + 2
    ]
    assert ("--tag", request.build_tag) == argv[argv.index("--tag") : argv.index("--tag") + 2]
    assert "--push" in argv
    assert argv[-1] == "."
    assert cwd == request.context
    assert stdin is None
    assert all("secret" not in argument.lower() for argument in argv)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "containerimage.digest"),
        ({"containerimage.digest": "sha256:not-a-digest"}, "digest"),
        ([{"containerimage.digest": _DIGEST}], "JSON object"),
    ],
)
def test_build_cloud_refuses_missing_or_malformed_digest_metadata(
    tmp_path: Path, metadata: object, message: str
) -> None:
    with pytest.raises(D1OperatorTransportError, match=message) as captured:
        DockerBuildCloudD1Adapter(runner=_Runner(metadata=metadata)).build_and_push(
            _request(tmp_path)
        )

    assert captured.value.operation == "build_image"


def test_build_cloud_wraps_runner_failure_without_fallback(tmp_path: Path) -> None:
    runner = _Runner(error=RuntimeError("simulated build failure"))

    with pytest.raises(D1OperatorTransportError, match="simulated build failure") as captured:
        DockerBuildCloudD1Adapter(runner=runner).build_and_push(_request(tmp_path))

    assert captured.value.operation == "build_image"
    assert len(runner.calls) == 1


def test_build_cloud_refuses_nonzero_result_without_echoing_raw_output(tmp_path: Path) -> None:
    secret = "build-secret-sentinel"  # pragma: allowlist secret
    runner = _Runner(returncode=17, stdout=secret, stderr=secret)

    with pytest.raises(D1OperatorTransportError, match="exit 17") as caught:
        DockerBuildCloudD1Adapter(runner=runner).build_and_push(_request(tmp_path))

    assert secret not in str(caught.value)


def test_build_cloud_rejects_duplicate_or_nonfinite_json_metadata(tmp_path: Path) -> None:
    class _MalformedRunner(_Runner):
        def run(
            self,
            argv: tuple[str, ...],
            *,
            cwd: Path | None,
            stdin: str | None,
        ) -> D1CommandResult:
            self.calls.append((argv, cwd, stdin))
            metadata_at = argv.index("--metadata-file") + 1
            Path(argv[metadata_at]).write_text(
                f'{{"containerimage.digest":"{_DIGEST}","containerimage.digest":NaN}}',
                encoding="utf-8",
            )
            return D1CommandResult(stdout="", stderr="")

    with pytest.raises(D1OperatorTransportError, match="metadata"):
        DockerBuildCloudD1Adapter(runner=_MalformedRunner()).build_and_push(_request(tmp_path))
