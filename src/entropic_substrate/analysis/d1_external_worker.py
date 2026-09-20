"""Hermetic D1 external-worker core for v2 authority rehydration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISREG
from typing import NoReturn, cast

from entropic_substrate.analysis.d1_analysis_api import (
    D1AnalysisCompositionRequest,
    compose_d1_analysis,
)
from entropic_substrate.analysis.d1_execution_receipt import D1IsolationRequest
from entropic_substrate.analysis.d1_external_analysis_codec import (
    D1ExternalAnalysisWorkerResponseV2,
    decode_d1_external_analysis_request_v2,
    encode_d1_external_analysis_worker_response_v2,
)
from entropic_substrate.evidence.d1_capture_aggregate_codec import (
    hydrate_d1_capture_aggregate,
)
from entropic_substrate.experiments.d1_config import load_d1_profile_catalog
from entropic_substrate.experiments.d1_family import D1ExperimentFamily
from entropic_substrate.experiments.d1_registration_codec import (
    D1PreregistrationByteEvidence,
    hydrate_registered_design,
)

__all__ = [
    "D1ExternalWorkerArtifacts",
    "D1ExternalWorkerError",
    "main",
    "publish_d1_external_worker_artifacts",
    "run_d1_external_worker",
]

_FORBIDDEN_ENV_PREFIX = "MLFLOW_"
_ANALYSIS_NAME = "blinded-analysis.json"
_RESPONSE_NAME = "worker-response.json"
_REQUEST_PATH = Path("/d1/input/request.json")
_CAPTURE_ROOT = Path("/d1/input/capture")
_PREREGISTRATION_PATH = Path("/d1/input/preregistration")
_READER_ROOT = Path("/d1/input/reader")
_ANALYSIS_PATH = Path("/d1/output") / _ANALYSIS_NAME
_RESPONSE_PATH = Path("/d1/output") / _RESPONSE_NAME


class D1ExternalWorkerError(RuntimeError):
    """Sanitized worker refusal; raw mounted evidence never enters the message."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"D1 external worker refused [{code}]")


def _fail(code: str) -> NoReturn:
    raise D1ExternalWorkerError(code)


def _stage[T](code: str, operation: Callable[[], T]) -> T:
    try:
        return operation()
    except D1ExternalWorkerError:
        raise
    except Exception as error:
        raise D1ExternalWorkerError(code) from error


def _canonical_text(value: Mapping[str, object]) -> str:
    return json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _real_directory(value: object, *, code: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError("worker directory boundary requires Path")
    lexical = Path(os.path.abspath(value))
    if lexical.is_symlink() or not lexical.is_dir() or lexical.resolve() != lexical:
        _fail(code)
    return lexical


# RETRACTED (house rule 4, measured 2026-09-20): this module previously
# verified the mounted reader tree's exact file identity against the D1
# reader-source provenance closure (analysis/d1_reader_source_closure.py,
# retired this phase) via `_verify_reader_identity` and its
# `_regular_mounted_file` helper. Measured: `git grep -n "reader_root" tests/
# src/` after their removal shows the mount's only remaining consumer is
# `_real_directory` below, which checks that the mounted path exists and is
# a real, non-symlinked directory -- it verifies NOTHING about the files
# beneath it. The `/d1/input/reader` mount and its `--reader-root` argv are
# KEPT (COMPLETION ruling, 2026-09-20) rather than removed, because the v2
# request envelope's `reader_source` field they once cross-checked is
# retired in the same commit, and this repo's own request/response contract
# doors are frozen for this phase; nothing in analysis/d1_external_worker.py
# or analysis/d1_external_analysis_adapter.py verifies the mounted tree's
# contents any more.


def _family(expected_digest: str) -> D1ExperimentFamily:
    matches = [
        profile.family
        for profile in load_d1_profile_catalog()
        if profile.family.family_digest == expected_digest
    ]
    if not matches or any(family != matches[0] for family in matches[1:]):
        _fail("family")
    return matches[0]


def _guard_environment(environ: object) -> None:
    if not isinstance(environ, Mapping):
        raise TypeError("worker environment must be a string mapping")
    values = cast(Mapping[object, object], environ)
    if any(type(key) is not str or type(value) is not str for key, value in values.items()):
        raise TypeError("worker environment must be a string mapping")
    typed = cast(Mapping[str, str], values)
    if any(key.startswith(_FORBIDDEN_ENV_PREFIX) and value for key, value in typed.items()):
        _fail("live_tracking_config")


@dataclass(frozen=True, slots=True)
class D1ExternalWorkerArtifacts:
    analysis_bytes: bytes = field(repr=False)
    response_bytes: bytes = field(repr=False)
    analysis_sha256: str
    response: D1ExternalAnalysisWorkerResponseV2

    def __post_init__(self) -> None:
        if type(self.analysis_bytes) is not bytes or not self.analysis_bytes:
            raise TypeError("worker analysis artifact must be nonempty bytes")
        if type(self.response_bytes) is not bytes or not self.response_bytes:
            raise TypeError("worker response artifact must be nonempty bytes")
        digest = hashlib.sha256(self.analysis_bytes).hexdigest()
        if digest != self.analysis_sha256 or digest != self.response.analysis_sha256:
            raise ValueError("worker analysis artifact digest differs")
        if self.response_bytes != encode_d1_external_analysis_worker_response_v2(self.response):
            raise ValueError("worker response bytes differ from the typed response")


def run_d1_external_worker(
    *,
    request_payload: bytes,
    preregistration_payload: bytes,
    capture_root: Path,
    reader_root: Path,
    environ: Mapping[str, str],
) -> D1ExternalWorkerArtifacts:
    """Rehydrate every authority and compose analysis without live services."""
    if type(request_payload) is not bytes or type(preregistration_payload) is not bytes:
        raise TypeError("worker request and preregistration payloads must be bytes")
    if not isinstance(cast(object, capture_root), Path) or not isinstance(
        cast(object, reader_root), Path
    ):
        raise TypeError("worker capture and reader roots must be Path")
    _guard_environment(environ)
    capture = _real_directory(capture_root, code="capture")
    _real_directory(reader_root, code="reader_root")
    evidence = _stage(
        "preregistration",
        lambda: D1PreregistrationByteEvidence.from_bytes(preregistration_payload),
    )
    request = _stage(
        "request",
        lambda: decode_d1_external_analysis_request_v2(
            request_payload,
            preregistration_evidence=evidence,
        ),
    )
    isolation = D1IsolationRequest()
    if request.isolation_request_digest != isolation.request_digest:
        _fail("isolation")
    family = _stage("family", lambda: _family(request.input_control.expected_family_digest))
    design = _stage(
        "design",
        lambda: hydrate_registered_design(
            _canonical_text(request.input_control.design.manifest),
            expected_design_digest=request.input_control.design.design_digest,
            family=family,
            decision_contract=request.input_control.decision_contract,
            preregistration_evidence=evidence,
        ),
    )
    aggregate = _stage(
        "capture",
        lambda: hydrate_d1_capture_aggregate(
            request.input_control.aggregate.canonical_json,
            expected_manifest_digest=request.input_control.aggregate.manifest_digest,
            capture_root=capture,
        ),
    )
    composition = _stage(
        "composition",
        lambda: compose_d1_analysis(
            D1AnalysisCompositionRequest(
                root=capture,
                design=design,
                decision_contract=request.input_control.decision_contract,
                aggregate=aggregate,
                aliases=request.input_control.alias_application.aliases,
                isolation=isolation,
            )
        ),
    )
    analysis_bytes = composition.analysis.canonical_json.encode("utf-8")
    analysis_sha256 = hashlib.sha256(analysis_bytes).hexdigest()
    response = D1ExternalAnalysisWorkerResponseV2(
        analysis_request_digest=request.analysis_request_digest,
        isolation_request_digest=request.isolation_request_digest,
        input_manifest_sha256=request.input_manifest_sha256,
        analysis_sha256=analysis_sha256,
    )
    return D1ExternalWorkerArtifacts(
        analysis_bytes=analysis_bytes,
        response_bytes=encode_d1_external_analysis_worker_response_v2(response),
        analysis_sha256=analysis_sha256,
        response=response,
    )


def _read_once(path: Path, *, code: str) -> bytes:
    if path.is_symlink():
        _fail(code)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise D1ExternalWorkerError(code) from error
    try:
        if not S_ISREG(os.fstat(descriptor).st_mode):
            _fail(code)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _read_stable(path: Path, *, code: str) -> bytes:
    if not isinstance(cast(object, path), Path):
        raise TypeError("worker input path must be Path")
    first = _read_once(path, code=code)
    second = _read_once(path, code=code)
    if first != second:
        _fail(code)
    return first


def _publication_paths(analysis_path: Path, response_path: Path) -> tuple[Path, Path]:
    if not isinstance(cast(object, analysis_path), Path) or not isinstance(
        cast(object, response_path), Path
    ):
        raise TypeError("worker publication paths must be Path")
    analysis = Path(os.path.abspath(analysis_path))
    response = Path(os.path.abspath(response_path))
    if analysis.name != _ANALYSIS_NAME or response.name != _RESPONSE_NAME:
        _fail("publication")
    if analysis.parent != response.parent:
        _fail("publication")
    parent = analysis.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve() != parent:
        _fail("publication")
    if analysis.exists() or analysis.is_symlink() or response.exists() or response.is_symlink():
        _fail("publication")
    return analysis, response


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def publish_d1_external_worker_artifacts(
    *,
    artifacts: D1ExternalWorkerArtifacts,
    analysis_path: Path,
    response_path: Path,
) -> None:
    """Publish analysis first and the response commit marker second."""
    try:
        analysis, response = _publication_paths(analysis_path, response_path)
        if type(artifacts) is not D1ExternalWorkerArtifacts:
            raise TypeError("worker publication requires typed artifacts")
        _atomic_publish(analysis, artifacts.analysis_bytes)
        _atomic_publish(response, artifacts.response_bytes)
    except D1ExternalWorkerError:
        raise
    except Exception as error:
        raise D1ExternalWorkerError("publication") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="d1-external-worker")
    parser.add_argument("--request", type=Path, default=_REQUEST_PATH)
    parser.add_argument("--capture-root", type=Path, default=_CAPTURE_ROOT)
    parser.add_argument("--preregistration", type=Path, default=_PREREGISTRATION_PATH)
    parser.add_argument("--reader-root", type=Path, default=_READER_ROOT)
    parser.add_argument("--analysis", type=Path, default=_ANALYSIS_PATH)
    parser.add_argument("--response", type=Path, default=_RESPONSE_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifacts = run_d1_external_worker(
            request_payload=_read_stable(args.request, code="request"),
            preregistration_payload=_read_stable(
                args.preregistration,
                code="preregistration",
            ),
            capture_root=args.capture_root,
            reader_root=args.reader_root,
            environ=os.environ,
        )
        publish_d1_external_worker_artifacts(
            artifacts=artifacts,
            analysis_path=args.analysis,
            response_path=args.response,
        )
        return 0
    except D1ExternalWorkerError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError):
        print("D1 external worker refused [worker_boundary]", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
