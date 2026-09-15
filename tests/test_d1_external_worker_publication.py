"""Atomic publication and executable CLI for the isolated D1 worker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

import entropic_substrate.analysis.d1_external_worker as worker_module
from entropic_substrate.analysis.d1_external_analysis_codec import (
    encode_d1_external_analysis_request_v2,
)
from entropic_substrate.analysis.d1_external_worker import (
    D1ExternalWorkerArtifacts,
    D1ExternalWorkerError,
    main,
    publish_d1_external_worker_artifacts,
)

from .test_d1_external_worker import _install_reader_fake, _request_case, _run_case
from .test_d1_registered_design_hydration import _RAW_PREREGISTRATION

pytestmark = pytest.mark.unit

_ANALYSIS = "blinded-analysis.json"
_RESPONSE = "worker-response.json"


def test_publication_orders_analysis_before_response_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _request_case(tmp_path / "case")
    _install_reader_fake(monkeypatch, case.runs)
    artifacts = _run_case(case)
    output = tmp_path / "output"
    output.mkdir()
    analysis = output / _ANALYSIS
    response = output / _RESPONSE
    order: list[str] = []
    atomic_publish = worker_module.__dict__["_atomic_publish"]

    def record(path: Path, payload: bytes) -> None:
        order.append(path.name)
        atomic_publish(path, payload)

    monkeypatch.setattr(worker_module, "_atomic_publish", record)
    publish_d1_external_worker_artifacts(
        artifacts=artifacts,
        analysis_path=analysis,
        response_path=response,
    )

    assert order == [_ANALYSIS, _RESPONSE]
    assert analysis.read_bytes() == artifacts.analysis_bytes
    assert response.read_bytes() == artifacts.response_bytes


def test_response_failure_leaves_analysis_without_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _request_case(tmp_path / "case")
    _install_reader_fake(monkeypatch, case.runs)
    artifacts = _run_case(case)
    output = tmp_path / "output"
    output.mkdir()
    analysis = output / _ANALYSIS
    response = output / _RESPONSE
    atomic_publish = worker_module.__dict__["_atomic_publish"]

    def fail_marker(path: Path, payload: bytes) -> None:
        if path.name == _RESPONSE:
            raise OSError("synthetic marker failure")
        atomic_publish(path, payload)

    monkeypatch.setattr(worker_module, "_atomic_publish", fail_marker)
    with pytest.raises(D1ExternalWorkerError) as raised:
        publish_d1_external_worker_artifacts(
            artifacts=artifacts,
            analysis_path=analysis,
            response_path=response,
        )

    assert raised.value.code == "publication"
    assert analysis.read_bytes() == artifacts.analysis_bytes
    assert not response.exists()


def test_publication_refuses_existing_wrong_or_split_targets(tmp_path: Path) -> None:
    output = tmp_path / "output"
    other = tmp_path / "other"
    output.mkdir()
    other.mkdir()
    analysis = output / _ANALYSIS
    response = output / _RESPONSE
    analysis.write_bytes(b"existing")
    invalid_artifacts = cast(D1ExternalWorkerArtifacts, object())
    for analysis_path, response_path in (
        (analysis, response),
        (output / "wrong.json", response),
        (output / _ANALYSIS, other / _RESPONSE),
    ):
        with pytest.raises(D1ExternalWorkerError, match="publication"):
            publish_d1_external_worker_artifacts(
                artifacts=invalid_artifacts,
                analysis_path=analysis_path,
                response_path=response_path,
            )

    valid_output = tmp_path / "valid-output"
    valid_output.mkdir()
    with pytest.raises(D1ExternalWorkerError, match="publication"):
        publish_d1_external_worker_artifacts(
            artifacts=invalid_artifacts,
            analysis_path=valid_output / _ANALYSIS,
            response_path=valid_output / _RESPONSE,
        )
    assert not any(valid_output.iterdir())


def _clear_tracking_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("MLFLOW_"):
            monkeypatch.delenv(key, raising=False)


def test_cli_publishes_success_and_sanitizes_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case = _request_case(tmp_path / "case")
    _install_reader_fake(monkeypatch, case.runs)
    _clear_tracking_environment(monkeypatch)
    request_path = tmp_path / "request.json"
    request_path.write_bytes(encode_d1_external_analysis_request_v2(case.request))
    preregistration = tmp_path / "preregistration"
    preregistration.write_bytes(_RAW_PREREGISTRATION)
    output = tmp_path / "output"
    output.mkdir()
    args = [
        "--request",
        str(request_path),
        "--capture-root",
        str(case.capture_root),
        "--preregistration",
        str(preregistration),
        "--reader-root",
        str(case.reader_root),
        "--analysis",
        str(output / _ANALYSIS),
        "--response",
        str(output / _RESPONSE),
    ]

    assert main(args) == 0
    assert {item.name for item in output.iterdir()} == {_ANALYSIS, _RESPONSE}
    assert capsys.readouterr().err == ""

    failed_output = tmp_path / "failed-output"
    failed_output.mkdir()
    sensitive_text = "never-echo-this-value"
    preregistration.write_text(sensitive_text, encoding="utf-8")
    failed_args = [
        *args[:-4],
        "--analysis",
        str(failed_output / _ANALYSIS),
        "--response",
        str(failed_output / _RESPONSE),
    ]
    assert main(failed_args) == 2
    stderr = capsys.readouterr().err
    assert "[request]" in stderr
    assert sensitive_text not in stderr
    assert not any(failed_output.iterdir())
