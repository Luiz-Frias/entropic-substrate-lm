"""Public D1 trainer services reach the real predictive model and trainer."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from d1_trainer_fixtures import ORIGIN_NS, micro_plan, run_start
from entropic_substrate.d1_cli import (
    D1_CLI_MANIFEST_SCHEMA,
    D1CliApplication,
    D1CliCommand,
    D1CliExitCode,
    D1CliServiceUnavailable,
)
from entropic_substrate.d1_trainer_cli_services import (
    D1FreshTrainerCliService,
    D1ResumeTrainerCliService,
)
from entropic_substrate.d1_trainer_request_codec import (
    D1FreshTrainerRequest,
    D1TrainerLocationSpec,
    trainer_request_to_mapping,
)
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.experiments.d1_primitives import D1Arm

pytestmark = pytest.mark.unit


class _Clock:
    def __init__(self, started_at) -> None:
        self._started_at = started_at
        self._utc = 0
        self._mono = ORIGIN_NS

    def utc_now(self):
        self._utc += 1
        return self._started_at + timedelta(seconds=self._utc)

    def monotonic_ns(self) -> int:
        self._mono += 1_000_000_000
        return self._mono


class _Observer:
    def observe(self, _event: object) -> None:
        pass


def _manifest(tmp_path: Path) -> tuple[Path, D1FreshTrainerRequest]:
    plan = micro_plan(D1Arm.BOTH)
    start = run_start(plan)
    request = D1FreshTrainerRequest(
        plan,
        start,
        D1TrainerWallOrigin(start.trainer_started_at, ORIGIN_NS),
        D1TrainerLocationSpec("run/journal.jsonl", "run/checkpoint.pt"),
    )
    (tmp_path / "run").mkdir()
    path = tmp_path / "fresh.json"
    path.write_text(
        json.dumps(
            {
                "command": D1CliCommand.RUN_FRESH.value,
                "payload": trainer_request_to_mapping(request),
                "schema_version": D1_CLI_MANIFEST_SCHEMA,
            }
        ),
        encoding="utf-8",
    )
    return path, request


def _application(fresh: D1FreshTrainerCliService) -> D1CliApplication:
    unavailable = {
        command: D1CliServiceUnavailable(f"{command.value} is not composed in this test")
        for command in D1CliCommand
    }
    unavailable[D1CliCommand.RUN_FRESH] = fresh
    return D1CliApplication(unavailable)


def test_d1_run_fresh_cli_reaches_real_optimizer_journal_and_checkpoint(tmp_path: Path) -> None:
    manifest, request = _manifest(tmp_path)
    service = D1FreshTrainerCliService(
        catalog=(request.plan.profile,),
        artifact_root=tmp_path,
        clock=_Clock(request.start.trainer_started_at),
    )

    outcome = _application(service).execute(D1CliCommand.RUN_FRESH, manifest)

    assert outcome.exit_code is D1CliExitCode.SUCCESS
    assert outcome.payload is not None
    assert outcome.payload["handoff_kind"] == "fresh_completion"
    assert outcome.payload["successful_step"] == 2
    assert (tmp_path / "run/journal.jsonl").is_file()
    assert (tmp_path / "run/checkpoint.pt").is_file()


def test_resume_service_rejects_a_fresh_request_before_trainer_execution(tmp_path: Path) -> None:
    manifest, request = _manifest(tmp_path)
    service = D1ResumeTrainerCliService(
        catalog=(request.plan.profile,),
        artifact_root=tmp_path,
        observer=_Observer(),
    )
    application = D1CliApplication(
        {
            command: (
                service
                if command is D1CliCommand.RUN_RESUME
                else D1CliServiceUnavailable("not under test")
            )
            for command in D1CliCommand
        }
    )

    outcome = application.execute(D1CliCommand.RUN_RESUME, manifest)

    assert outcome.exit_code is D1CliExitCode.INVALID_EVIDENCE
    assert "command does not match" in str(outcome.detail)
