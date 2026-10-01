"""Public D1 trainer services reach the real predictive model and trainer."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch

from d1_resume_assertions import assert_nested_equal
from d1_resume_fixtures import captured_step_one_case, origin, resume_start
from d1_trainer_fixtures import (
    ORIGIN_NS,
    DeterministicTrainerClock,
    micro_plan,
    run_start,
)
from entropic_substrate.composition.wiring.d1_cli import (
    D1_CLI_MANIFEST_SCHEMA,
    D1CliApplication,
    D1CliCommand,
    D1CliExitCode,
    D1CliServiceUnavailable,
)
from entropic_substrate.composition.wiring.d1_trainer_cli_services import (
    D1FreshTrainerCliService,
    D1ResumeTrainerCliService,
)
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.ladder.run.transport.d1_trainer_output_publication import (
    load_d1_published_trainer_output,
)
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    D1FreshTrainerRequest,
    D1ResumeTrainerRequest,
    D1TrainerLocationSpec,
    trainer_request_to_mapping,
)

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
    assert (tmp_path / "run/checkpoint.pt.handoff.json").is_file()
    assert (tmp_path / "run/checkpoint.pt.complete.json").is_file()
    assert outcome.payload["marker"]["path"] == str(tmp_path / "run/checkpoint.pt.complete.json")
    published = load_d1_published_trainer_output(
        artifact_root=tmp_path,
        publication_prefix="run/checkpoint.pt",
    )
    assert published.handoff.successful_step == 2


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


def test_fresh_service_never_reports_success_when_marker_publication_fails(
    tmp_path: Path,
) -> None:
    manifest, request = _manifest(tmp_path)
    handoff_path = tmp_path / "run/checkpoint.pt.handoff.json"
    marker_path = tmp_path / "run/checkpoint.pt.complete.json"
    handoff_path.write_bytes(b"occupied")
    service = D1FreshTrainerCliService(
        catalog=(request.plan.profile,),
        artifact_root=tmp_path,
        clock=_Clock(request.start.trainer_started_at),
    )

    outcome = _application(service).execute(D1CliCommand.RUN_FRESH, manifest)

    assert outcome.exit_code is D1CliExitCode.INVALID_EVIDENCE
    assert "publication failed" in str(outcome.detail)
    assert not marker_path.exists()


def test_d1_run_resume_cli_restores_and_completes_real_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    case = captured_step_one_case(interrupted, monkeypatch)
    start = resume_start(case.plan, case.checkpoint_sha256)
    request = D1ResumeTrainerRequest(
        case.plan,
        start,
        origin(),
        D1TrainerLocationSpec(
            str(case.journal_path.relative_to(tmp_path)),
            str(case.checkpoint_path.relative_to(tmp_path)),
            "interrupted/quarantined-tail.jsonl",
        ),
    )
    manifest = tmp_path / "resume.json"
    manifest.write_text(
        json.dumps(
            {
                "command": D1CliCommand.RUN_RESUME.value,
                "payload": trainer_request_to_mapping(request),
                "schema_version": D1_CLI_MANIFEST_SCHEMA,
            }
        ),
        encoding="utf-8",
    )
    resume = D1ResumeTrainerCliService(
        catalog=(case.plan.profile,),
        artifact_root=tmp_path,
        observer=_Observer(),
        clock=DeterministicTrainerClock(),
    )
    services = {
        command: (
            resume
            if command is D1CliCommand.RUN_RESUME
            else D1CliServiceUnavailable("not under test")
        )
        for command in D1CliCommand
    }

    outcome = D1CliApplication(services).execute(D1CliCommand.RUN_RESUME, manifest)

    assert outcome.exit_code is D1CliExitCode.SUCCESS
    assert outcome.payload is not None
    assert outcome.payload["handoff_kind"] == "resume_completion"
    assert outcome.payload["successful_step"] == 2
    assert case.journal_path.is_file()
    assert case.checkpoint_path.is_file()
    prefix = str(case.checkpoint_path.relative_to(tmp_path))
    assert (tmp_path / f"{prefix}.handoff.json").is_file()
    assert (tmp_path / f"{prefix}.complete.json").is_file()
    assert (
        load_d1_published_trainer_output(
            artifact_root=tmp_path,
            publication_prefix=prefix,
        ).handoff.successful_step
        == 2
    )
    resumed = torch.load(case.checkpoint_path, map_location="cpu", weights_only=True)
    assert case.uninterrupted_terminal is not None
    for field in ("model", "optimizer", "sampler", "rng"):
        assert_nested_equal(resumed[field], case.uninterrupted_terminal[field])


_CANONICAL_FIXTURES: tuple[dict[str, object], ...] = (
    {},
    {"b": 1, "a": [1, 2.5, None, True], "c": {"z": "x", "y": []}},
    {"unicode": "café ☃ \U0001f600", "tuple_free": [[1], [2, [3]]]},
    {"n": -0.0, "big": 10**30, "small": 1e-9, "quote": 'a"b\\c\n'},
)


def _parent_bytes(value: dict[str, object]) -> bytes:
    """The parent expression of ``_request_bytes``, verbatim (oracle for the repoint)."""
    return json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


@pytest.mark.parametrize("payload", _CANONICAL_FIXTURES)
def test_request_bytes_equal_the_parent_canonical_bytes(payload: dict[str, object]) -> None:
    from types import SimpleNamespace

    from entropic_substrate.composition.wiring import d1_trainer_cli_services as services

    invocation = SimpleNamespace(payload=payload)
    assert services._request_bytes(invocation) == _parent_bytes(payload)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_request_bytes_refuses_nonfinite_payload_as_invalid_evidence(bad: float) -> None:
    from types import SimpleNamespace

    from entropic_substrate.composition.wiring import d1_trainer_cli_services as services
    from entropic_substrate.composition.wiring.d1_cli import D1CliInvalidEvidence

    with pytest.raises(D1CliInvalidEvidence, match="not canonical"):
        services._request_bytes(SimpleNamespace(payload={"x": bad}))  # type: ignore[arg-type]


def test_request_bytes_is_the_shared_canonical_json_not_a_local_dump() -> None:
    from entropic_substrate.composition.wiring import d1_trainer_cli_services as services
    from entropic_substrate.kernel.identity import d1_primitives

    assert services.canonical_json is d1_primitives.canonical_json  # type: ignore[attr-defined]
    assert "json.dumps" not in Path(services.__file__).read_text(encoding="utf-8")
