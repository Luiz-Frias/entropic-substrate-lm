"""Legacy CLI registration for the strict D1 application boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from entropic_substrate.cli import build_parser, main
from entropic_substrate.d1_cli import (
    D1_CLI_MANIFEST_SCHEMA,
    D1CliApplication,
    D1CliCommand,
    D1CliExitCode,
    D1CliInvocation,
)

pytestmark = pytest.mark.unit


@dataclass
class _Service:
    calls: list[D1CliInvocation] = field(default_factory=list)

    def execute(self, invocation: D1CliInvocation) -> dict[str, object]:
        self.calls.append(invocation)
        return {"receipt": invocation.payload["receipt"]}


def _application() -> tuple[D1CliApplication, dict[D1CliCommand, _Service]]:
    services = {command: _Service() for command in D1CliCommand}
    return D1CliApplication(services), services


def _manifest(tmp_path: Path, command: D1CliCommand) -> Path:
    path = tmp_path / f"{command.value}.json"
    path.write_text(
        json.dumps(
            {
                "command": command.value,
                "payload": {"receipt": f"{command.value}-receipt"},
                "schema_version": D1_CLI_MANIFEST_SCHEMA,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("command", list(D1CliCommand))
def test_each_d1_subcommand_accepts_only_a_canonical_manifest_path(
    tmp_path: Path, command: D1CliCommand
) -> None:
    parser = build_parser()
    manifest = _manifest(tmp_path, command)

    args = parser.parse_args([command.value, "--manifest", str(manifest)])

    assert args.command == command.value
    assert args.manifest == manifest
    with pytest.raises(SystemExit) as loose_flag:
        parser.parse_args([command.value, "--manifest", str(manifest), "--seed", "7"])
    assert loose_flag.value.code == 2


@pytest.mark.parametrize("command", list(D1CliCommand))
def test_main_dispatches_to_injected_application_and_prints_one_json_outcome(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: D1CliCommand,
) -> None:
    application, services = _application()
    manifest = _manifest(tmp_path, command)

    code = main(
        [command.value, "--manifest", str(manifest)],
        d1_application=application,
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "command": command.value,
        "detail": None,
        "error_type": None,
        "exit_code": 0,
        "payload": {"receipt": f"{command.value}-receipt"},
        "status": "success",
    }
    assert len(services[command].calls) == 1
    assert all(not service.calls for key, service in services.items() if key is not command)


def test_unconfigured_shell_command_returns_operator_gap_not_live_fallback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENTROPIC_ALLOW_UNGATED", "1")
    command = D1CliCommand.RUN_FRESH

    code = main([command.value, "--manifest", str(_manifest(tmp_path, command))])

    assert code is D1CliExitCode.OPERATOR_GAP
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "operator_gap"
    assert output["exit_code"] == 20
    assert "composition service is unavailable" in output["detail"]


def test_invalid_manifest_exit_survives_cli_rendering(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    application, services = _application()
    command = D1CliCommand.INSPECT
    path = tmp_path / "wrong-command.json"
    path.write_text(
        json.dumps(
            {
                "command": D1CliCommand.CAPTURE.value,
                "payload": {},
                "schema_version": D1_CLI_MANIFEST_SCHEMA,
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [command.value, "--manifest", str(path)],
        d1_application=application,
    )

    assert code is D1CliExitCode.INVALID_EVIDENCE
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "invalid_evidence"
    assert all(not service.calls for service in services.values())


def test_legacy_train_parser_contract_is_unchanged() -> None:
    args = build_parser().parse_args(["train", "--exp", "e1-fisher", "--seed", "7", "--quiet"])

    assert args.command == "train"
    assert args.exp == "e1-fisher"
    assert args.seed == 7
    assert args.quiet is True
