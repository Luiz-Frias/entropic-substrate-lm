"""Docker Build Cloud adapter for immutable D1 linux/amd64 images."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import NoReturn, cast

from entropic_substrate.d1_operator_ports import (
    D1CommandRunner,
    D1OperatorTransportError,
)
from entropic_substrate.d1_operator_values import (
    D1ImageBuildRequest,
    D1ImmutableImage,
)


def _error(subject: str, detail: str, cause: Exception | None = None) -> NoReturn:
    error = D1OperatorTransportError(operation="build_image", subject=subject, detail=detail)
    if cause is None:
        raise error
    raise error from cause


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate build metadata key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    raise ValueError(f"non-finite build metadata constant {value!r}")


def _metadata(path: Path, *, subject: str) -> str:
    try:
        raw = path.read_text(encoding="utf-8")
        payload: object = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        _error(subject, f"build metadata is invalid: {error}", error)
    if type(payload) is not dict:
        _error(subject, "build metadata must be a JSON object")
    digest = cast(dict[str, object], payload).get("containerimage.digest")
    if type(digest) is not str:
        _error(subject, "build metadata requires containerimage.digest")
    return digest


class DockerBuildCloudD1Adapter:
    """Push one commit-tagged image and resolve its immutable OCI digest."""

    def __init__(self, *, runner: D1CommandRunner) -> None:
        if not hasattr(runner, "run"):
            raise TypeError("Docker Build Cloud adapter requires a command runner")
        self._runner = runner

    def build_and_push(self, request: D1ImageBuildRequest) -> D1ImmutableImage:
        if type(request) is not D1ImageBuildRequest:
            raise TypeError("build_and_push requires D1ImageBuildRequest")
        subject = request.build_tag
        dockerfile = request.dockerfile.relative_to(request.context).as_posix()
        with tempfile.TemporaryDirectory(prefix="d1-build-metadata-") as temporary:
            metadata = Path(temporary) / "metadata.json"
            argv = (
                "docker",
                "buildx",
                "build",
                "--builder",
                request.builder,
                "--platform",
                request.platform,
                "--file",
                dockerfile,
                "--tag",
                request.build_tag,
                "--label",
                f"org.opencontainers.image.revision={request.source_commit}",
                "--metadata-file",
                str(metadata),
                "--provenance=mode=max",
                "--push",
                ".",
            )
            try:
                self._runner.run(argv, cwd=request.context, stdin=None)
            except Exception as error:
                _error(subject, str(error) or repr(error), error)
            digest = _metadata(metadata, subject=subject)
        try:
            return D1ImmutableImage(
                repository=request.repository,
                digest=digest,
                source_commit=request.source_commit,
                platform=request.platform,
            )
        except ValueError as error:
            _error(subject, f"resolved image digest is invalid: {error}", error)
