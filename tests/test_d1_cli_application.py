"""D1 command-envelope dispatch and failure-semantics contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from entropic_substrate.composition.wiring.d1_cli import (
    D1_CLI_MANIFEST_SCHEMA,
    D1CliApplication,
    D1CliCommand,
    D1CliExitCode,
    D1CliInvalidEvidence,
    D1CliInvocation,
    D1CliOperatorGap,
    D1CliServiceUnavailable,
)

pytestmark = pytest.mark.unit


@dataclass
class _RecordingService:
    name: str
    calls: list[D1CliInvocation] = field(default_factory=list)
    failure: Exception | None = None

    def execute(self, invocation: D1CliInvocation) -> dict[str, object]:
        self.calls.append(invocation)
        if self.failure is not None:
            raise self.failure
        return {"service": self.name, "subject": invocation.payload["subject"]}


def _manifest(tmp_path: Path, command: D1CliCommand) -> Path:
    path = tmp_path / f"{command.value}.json"
    path.write_text(
        json.dumps(
            {
                "command": command.value,
                "payload": {"subject": f"{command.value}-subject"},
                "schema_version": D1_CLI_MANIFEST_SCHEMA,
            }
        ),
        encoding="utf-8",
    )
    return path


def _application(
    *, replacement: tuple[D1CliCommand, _RecordingService] | None = None
) -> tuple[D1CliApplication, dict[D1CliCommand, _RecordingService]]:
    services = {command: _RecordingService(command.value) for command in D1CliCommand}
    if replacement is not None:
        services[replacement[0]] = replacement[1]
    return D1CliApplication(services), services


@pytest.mark.parametrize("command", list(D1CliCommand))
def test_each_command_dispatches_exactly_once_to_its_injected_service(
    tmp_path: Path, command: D1CliCommand
) -> None:
    application, services = _application()

    outcome = application.execute(command, _manifest(tmp_path, command))

    assert outcome.exit_code is D1CliExitCode.SUCCESS
    assert outcome.manifest == {
        "command": command.value,
        "detail": None,
        "error_type": None,
        "exit_code": 0,
        "payload": {"service": command.value, "subject": f"{command.value}-subject"},
        "status": "success",
    }
    assert len(services[command].calls) == 1
    assert services[command].calls[0].manifest_path.name == f"{command.value}.json"
    assert services[command].calls[0].payload == {"subject": f"{command.value}-subject"}
    assert all(not service.calls for key, service in services.items() if key is not command)


@pytest.mark.parametrize(
    ("failure", "code", "status"),
    [
        (D1CliOperatorGap("provider billing receipt is absent"), 20, "operator_gap"),
        (D1CliInvalidEvidence("filing graph did not verify"), 21, "invalid_evidence"),
        (RuntimeError("unhandled adapter defect"), 70, "unexpected_defect"),
    ],
)
def test_exit_codes_distinguish_gap_invalid_evidence_and_defect(
    tmp_path: Path, failure: Exception, code: int, status: str
) -> None:
    command = D1CliCommand.CAPTURE_FINALIZE
    service = _RecordingService(command.value, failure=failure)
    application, _ = _application(replacement=(command, service))

    outcome = application.execute(command, _manifest(tmp_path, command))

    assert outcome.exit_code == code
    assert outcome.status == status
    assert outcome.payload is None
    assert outcome.error_type == type(failure).__name__
    assert outcome.detail == str(failure)
    assert len(service.calls) == 1


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ("[]", "top level"),
        ('{"command":"d1-inspect"}', "keys"),
        (
            json.dumps(
                {
                    "command": "d1-run-fresh",
                    "payload": {},
                    "schema_version": D1_CLI_MANIFEST_SCHEMA,
                }
            ),
            "does not match",
        ),
        (
            '{"command":"d1-inspect","command":"d1-inspect",'
            f'"payload":{{}},"schema_version":"{D1_CLI_MANIFEST_SCHEMA}"}}',
            "duplicate",
        ),
        (
            '{"command":"d1-inspect","payload":{"metric":NaN},'
            f'"schema_version":"{D1_CLI_MANIFEST_SCHEMA}"}}',
            "non-finite",
        ),
    ],
)
def test_manifest_refusal_happens_before_any_service_call(
    tmp_path: Path, payload: str, detail: str
) -> None:
    path = tmp_path / "request.json"
    path.write_text(payload, encoding="utf-8")
    application, services = _application()

    outcome = application.execute(D1CliCommand.INSPECT, path)

    assert outcome.exit_code is D1CliExitCode.INVALID_EVIDENCE
    assert outcome.status == "invalid_evidence"
    assert detail in cast(str, outcome.detail)
    assert all(not service.calls for service in services.values())


def test_symlinked_or_missing_manifest_is_refused_before_dispatch(tmp_path: Path) -> None:
    target = _manifest(tmp_path, D1CliCommand.INSPECT)
    link = tmp_path / "request-link.json"
    link.symlink_to(target)
    application, services = _application()

    linked = application.execute(D1CliCommand.INSPECT, link)
    missing = application.execute(D1CliCommand.INSPECT, tmp_path / "missing.json")

    assert linked.exit_code is D1CliExitCode.INVALID_EVIDENCE
    assert missing.exit_code is D1CliExitCode.INVALID_EVIDENCE
    assert "regular non-symlink" in cast(str, linked.detail)
    assert "regular non-symlink" in cast(str, missing.detail)
    assert all(not service.calls for service in services.values())


def test_unavailable_service_remains_a_gap_even_with_legacy_ungated_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENTROPIC_ALLOW_UNGATED", "1")
    command = D1CliCommand.CAMPAIGN_RUN
    unavailable = D1CliServiceUnavailable("validated filing service is not installed")
    application, _ = _application(replacement=(command, unavailable))

    outcome = application.execute(command, _manifest(tmp_path, command))

    assert outcome.exit_code is D1CliExitCode.OPERATOR_GAP
    assert outcome.status == "operator_gap"
    assert "not installed" in cast(str, outcome.detail)


def test_application_requires_every_core_service_without_default_fallback() -> None:
    with pytest.raises(ValueError, match="missing D1 command services"):
        D1CliApplication({D1CliCommand.INSPECT: _RecordingService("inspect")})
