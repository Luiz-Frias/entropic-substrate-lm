"""Real CLI service and lazy production composition for D1 inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from entropic_substrate.composition.entrypoints.cli import main
from entropic_substrate.composition.wiring.d1_cli import (
    D1_CLI_MANIFEST_SCHEMA,
    D1CliCommand,
    D1CliExitCode,
    D1CliInvalidEvidence,
    D1CliInvocation,
)
from entropic_substrate.composition.wiring.d1_inspect_service import D1InspectCliService
from entropic_substrate.ladder.analyze.d1_inspect_codec import d1_inspect_result_from_mapping
from entropic_substrate.ladder.analyze.d1_inspect_request import (
    D1_INSPECT_REQUEST_SCHEMA,
    D1InspectRequest,
)
from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1RunStatus

from .d1_telemetry_fixtures import LOGICAL_RUN_ID, valid_events, write_events

pytestmark = pytest.mark.unit


def _journal(path: Path, *, partial: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    events = valid_events()[:-1] if partial else valid_events()
    write_events(path, events)
    return path.stat().st_size


def _request(path: Path, *, max_bytes: int, allow_incomplete: bool = False) -> D1InspectRequest:
    return D1InspectRequest(
        path,
        max_bytes,
        allow_incomplete,
        2,
        LOGICAL_RUN_ID,
        None,
        None,
        None,
    )


def _invocation(manifest_path: Path, request: D1InspectRequest) -> D1CliInvocation:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    return D1CliInvocation(D1CliCommand.INSPECT, manifest_path, request.manifest)


def _outer_manifest(path: Path, request: D1InspectRequest) -> Path:
    path.write_text(
        json.dumps(
            {
                "command": D1CliCommand.INSPECT.value,
                "payload": request.manifest,
                "schema_version": D1_CLI_MANIFEST_SCHEMA,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_service_resolves_relative_path_from_manifest_directory_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "operator"
    journal = root / "artifacts" / "run.jsonl"
    size = _journal(journal)
    invocation = _invocation(
        root / "inspect.json",
        _request(Path("artifacts/run.jsonl"), max_bytes=size),
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = D1InspectCliService().execute(invocation)

    view = d1_inspect_result_from_mapping(dict(result))
    assert view.status is D1RunStatus.COMPLETE
    assert view.journal.path == str(journal)
    assert view.run.logical_run_id == LOGICAL_RUN_ID


def test_service_returns_partial_only_when_request_explicitly_allows_it(tmp_path: Path) -> None:
    journal = tmp_path / "partial.jsonl"
    size = _journal(journal, partial=True)
    service = D1InspectCliService()

    with pytest.raises(D1CliInvalidEvidence, match="partial"):
        service.execute(_invocation(tmp_path / "strict.json", _request(journal, max_bytes=size)))
    result = service.execute(
        _invocation(
            tmp_path / "allowed.json",
            _request(journal, max_bytes=size, allow_incomplete=True),
        )
    )
    assert d1_inspect_result_from_mapping(dict(result)).status is D1RunStatus.PARTIAL


@pytest.mark.parametrize("failure", ["wrong-command", "bad-request", "too-small", "identity"])
def test_service_normalizes_expected_boundary_refusals_to_invalid_evidence(
    tmp_path: Path, failure: str
) -> None:
    journal = tmp_path / "run.jsonl"
    size = _journal(journal)
    request = _request(journal, max_bytes=size)
    invocation = _invocation(tmp_path / "inspect.json", request)
    if failure == "wrong-command":
        invocation = D1CliInvocation(D1CliCommand.RUN_FRESH, invocation.manifest_path, {})
    elif failure == "bad-request":
        invocation = D1CliInvocation(
            D1CliCommand.INSPECT,
            invocation.manifest_path,
            {**request.manifest, "journal_path": "bad\x00path"},
        )
    elif failure == "too-small":
        invocation = _invocation(
            tmp_path / "small.json",
            _request(journal, max_bytes=size - 1),
        )
    else:
        invocation = _invocation(
            tmp_path / "identity.json",
            D1InspectRequest(journal, size, False, 2, "wrong-run", None, None, None),
        )

    with pytest.raises(D1CliInvalidEvidence) as caught:
        D1InspectCliService().execute(invocation)
    assert caught.value.__cause__ is not None


def test_default_main_inspect_is_real_and_never_loads_trainer_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    journal = tmp_path / "run.jsonl"
    size = _journal(journal)
    manifest = _outer_manifest(
        tmp_path / "inspect.json",
        _request(Path("run.jsonl"), max_bytes=size),
    )

    def forbidden_catalog(*, tiny: bool = False) -> tuple[()]:
        del tiny
        raise AssertionError("d1-inspect must not load trainer catalogs")

    monkeypatch.setattr(
        "entropic_substrate.composition.wiring.d1_cli_composition.load_d1_profile_catalog",
        forbidden_catalog,
    )

    code = main([D1CliCommand.INSPECT.value, "--manifest", str(manifest)])

    assert code is D1CliExitCode.SUCCESS
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "success"
    assert output["payload"]["schema_version"] == "d1-inspect-result-v1"


def test_default_main_maps_nul_target_to_exit_21_not_unexpected_defect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    value = _request(Path("run.jsonl"), max_bytes=1).manifest
    value["journal_path"] = "bad\x00path"
    manifest = tmp_path / "inspect.json"
    manifest.write_text(
        json.dumps(
            {
                "command": D1CliCommand.INSPECT.value,
                "payload": value,
                "schema_version": D1_CLI_MANIFEST_SCHEMA,
            }
        ),
        encoding="utf-8",
    )

    code = main([D1CliCommand.INSPECT.value, "--manifest", str(manifest)])

    assert code is D1CliExitCode.INVALID_EVIDENCE
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "invalid_evidence"
    assert output["error_type"] == D1CliInvalidEvidence.__name__
    assert "journal_path" in cast(str, output["detail"])


def test_request_schema_constant_remains_nested_beneath_cli_envelope() -> None:
    assert _request(Path("run.jsonl"), max_bytes=1).manifest["schema_version"] == (
        D1_INSPECT_REQUEST_SCHEMA
    )
