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
from test_d1_trainer_cli_services import _manifest as _trainer_manifest

pytestmark = pytest.mark.unit

_PUBLIC_D1_COMMANDS: tuple[str, ...] = (
    "d1-run-fresh",
    "d1-run-resume",
    "d1-campaign-run",
    "d1-capture-prepare",
    "d1-capture-finalize",
    "d1-inspect",
)


def test_public_d1_command_surface_is_the_hard_cutover_contract() -> None:
    assert tuple(command.value for command in D1CliCommand) == _PUBLIC_D1_COMMANDS


@pytest.mark.parametrize("legacy", ["d1-finalize", "d1-capture", "d1-file-gate"])
def test_removed_monolithic_commands_have_no_parser_alias(legacy: str) -> None:
    with pytest.raises(SystemExit) as captured:
        build_parser().parse_args([legacy, "--manifest", "request.json"])
    assert captured.value.code == 2


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


def test_default_shell_fresh_command_reaches_real_trainer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, request = _trainer_manifest(tmp_path)
    monkeypatch.setattr(
        "entropic_substrate.d1_cli_composition.load_d1_profile_catalog",
        lambda *, tiny=False: (request.plan.profile,) if tiny else (),
    )

    code = main([D1CliCommand.RUN_FRESH.value, "--manifest", str(manifest)])

    assert code is D1CliExitCode.SUCCESS
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "success"
    assert output["payload"]["handoff_kind"] == "fresh_completion"
    assert output["payload"]["successful_step"] == 2


def test_default_shell_resume_command_reaches_real_service_validation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, request = _trainer_manifest(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["command"] = D1CliCommand.RUN_RESUME.value
    value["payload"]["command"] = D1CliCommand.RUN_RESUME.value
    manifest.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(
        "entropic_substrate.d1_cli_composition.load_d1_profile_catalog",
        lambda *, tiny=False: (request.plan.profile,) if tiny else (),
    )

    code = main([D1CliCommand.RUN_RESUME.value, "--manifest", str(manifest)])

    assert code is D1CliExitCode.INVALID_EVIDENCE
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "invalid_evidence"
    assert "segment mode" in output["detail"]


def test_unconfigured_campaign_command_returns_operator_gap_not_live_fallback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENTROPIC_ALLOW_UNGATED", "1")
    command = D1CliCommand.CAMPAIGN_RUN

    code = main([command.value, "--manifest", str(_manifest(tmp_path, command))])

    assert code is D1CliExitCode.OPERATOR_GAP
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "operator_gap"
    assert output["exit_code"] == 20
    assert "composition service is unavailable" in output["detail"]


@pytest.mark.parametrize(
    "command",
    [D1CliCommand.CAPTURE_PREPARE, D1CliCommand.CAPTURE_FINALIZE],
)
def test_default_capture_commands_are_real_and_reject_bad_payload_as_invalid_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: D1CliCommand,
) -> None:
    code = main([command.value, "--manifest", str(_manifest(tmp_path, command))])

    assert code is D1CliExitCode.INVALID_EVIDENCE
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "invalid_evidence"
    assert "request" in output["detail"]
    assert "composition service is unavailable" not in output["detail"]


def test_invalid_manifest_exit_survives_cli_rendering(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    application, services = _application()
    command = D1CliCommand.INSPECT
    path = tmp_path / "wrong-command.json"
    path.write_text(
        json.dumps(
            {
                "command": D1CliCommand.CAPTURE_PREPARE.value,
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
