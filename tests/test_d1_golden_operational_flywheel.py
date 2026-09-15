"""Production-shaped D1 operations across campaign, recovery, and capture rails."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import d1_launch_admission_fixtures as fixtures
from d1_resume_fixtures import captured_step_one_case, origin, resume_start
from entropic_substrate.cli import main
from entropic_substrate.d1_campaign_bootstrap import (
    adopt_d1_finalized_cell_after_crash,
)
from entropic_substrate.d1_campaign_cli_service import D1CampaignRunCliService
from entropic_substrate.d1_cli import (
    D1_CLI_MANIFEST_SCHEMA,
    D1CliApplication,
    D1CliCommand,
    D1CliExitCode,
    D1CliServiceUnavailable,
)
from entropic_substrate.d1_r2_trainer_mailbox_worker import D1R2TrainerMailboxWorker
from entropic_substrate.d1_trainer_mailbox_worker import D1TrainerMailboxWaitPolicy
from entropic_substrate.d1_trainer_request_codec import (
    D1ResumeTrainerRequest,
    D1TrainerLocationSpec,
)
from entropic_substrate.evidence.d1_lifecycle_values import D1ProcessState

from .test_d1_campaign_bootstrap_adoption import _bootstrapped
from .test_d1_campaign_cli_service import _case as _campaign_case
from .test_d1_campaign_cli_service import _Factory, _Loader
from .test_d1_capture_finalize_coordinator import _prepared
from .test_d1_confirmatory_matrix import FakeGit
from .test_d1_r2_trainer_mailbox_worker import _Recovery, _seed
from .test_d1_trainer_object_transport import _ObjectStore
from .test_d1_trainer_process_worker import _POD, _worker

pytestmark = pytest.mark.unit


def _application(service: object) -> D1CliApplication:
    services = {
        command: D1CliServiceUnavailable(f"unused golden command {command.value}")
        for command in D1CliCommand
    }
    services[D1CliCommand.CAMPAIGN_RUN] = service  # type: ignore[assignment]
    return D1CliApplication(services)


def _campaign_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            {
                "command": D1CliCommand.CAMPAIGN_RUN.value,
                "payload": payload,
                "schema_version": D1_CLI_MANIFEST_SCHEMA,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_golden_operational_flywheel_runs_all_registered_recovery_boundaries(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, queued_controller, campaign_manifest = _campaign_case(
        tmp_path / "campaign-command",
        0,
    )
    campaign_service = D1CampaignRunCliService(
        catalog=fixtures.profiles(),
        authority_loader=_Loader(loaded),
        git=FakeGit(),
        controller_factory=_Factory(queued_controller),
    )
    _campaign_manifest(campaign_manifest, loaded.request.manifest)

    campaign_code = main(
        [D1CliCommand.CAMPAIGN_RUN.value, "--manifest", str(campaign_manifest)],
        d1_application=_application(campaign_service),
    )
    campaign_output = json.loads(capsys.readouterr().out)

    assert campaign_code is D1CliExitCode.SUCCESS
    assert campaign_output["payload"]["status"] == "campaign_complete"
    assert campaign_output["payload"]["completed_cell_count"] == 4
    assert queued_controller.closed is True

    crash_root = tmp_path / "crash-adoption"
    crash_root.mkdir()
    crash = _bootstrapped(crash_root)
    finalized = crash.controller.execute_fresh(crash.case.intent)
    assert finalized.finalized_cell is not None
    provider_events = tuple(crash.events)

    adopted, adopted_generation = adopt_d1_finalized_cell_after_crash(
        design=crash.case.design,
        intent=crash.case.intent,
        checkpoint=crash.checkpoint,
        process_manager=finalized.process_manager,
        candidates=((finalized.finalized_cell, finalized.finalized_cell_artifact),),
        repository=crash.repository,
        generation=crash.receipt,
    )

    assert tuple(crash.events) == provider_events
    assert adopted.next_index == 1
    assert adopted_generation.sha256 != crash.receipt.sha256

    capture_campaign, capture_coordinator, prepared, preparation_root, repo_root, events = (
        _prepared(tmp_path / "two-phase-capture")
    )
    capture_finalized = capture_coordinator.finalize(
        campaign=capture_campaign,
        preparation_artifact=prepared.artifact,
        repo_root=repo_root,
        preparation_custody_root=preparation_root,
    )

    assert prepared.record.status.value == "commit_required"
    assert capture_finalized is not None
    assert capture_finalized.captured.process_manager.state is D1ProcessState.CAPTURED
    assert events == ["export", "dvc_add", "dvc_push", "cloud", "git"]

    source = tmp_path / "resume-source"
    source.mkdir()
    resume_case = captured_step_one_case(source, monkeypatch)
    resume_root = tmp_path / "resume-worker"
    resume_root.mkdir()
    start = resume_start(resume_case.plan, resume_case.checkpoint_sha256)
    locations = D1TrainerLocationSpec(
        "results/resume/journal.jsonl",
        "results/resume/checkpoint.pt",
        "results/resume/quarantined-tail.jsonl",
    )
    request = D1ResumeTrainerRequest(resume_case.plan, start, origin(), locations)
    objects = _ObjectStore()
    _seed(objects, request)
    recovery = _Recovery(
        resume_case.checkpoint_path,
        resume_case.journal_path,
        resume_root / locations.checkpoint_path,
        resume_root / locations.journal_path,
    )
    worker = D1R2TrainerMailboxWorker(
        repo_root=resume_root,
        pod_id=_POD,
        objects=objects,
        catalog=(request.plan.profile,),
        worker_factory=lambda publisher: _worker(
            resume_root,
            request,
            start_report_publisher=publisher,
        ),
        recovery=recovery,
        wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
    )

    resumed = worker.run()

    assert resumed["handoff_kind"] == "resume_completion"
    assert resumed["successful_step"] == 2
    assert recovery.calls == [(request.plan.logical_run_id, resume_case.checkpoint_sha256)]
