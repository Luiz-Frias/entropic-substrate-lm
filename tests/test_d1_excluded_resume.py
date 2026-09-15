"""Real interrupted pilot recovery preserves one scientific attempt and all segments."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from entropic_substrate.d1_artifact_snapshot import publish_d1_atomic_bytes
from entropic_substrate.d1_excluded_capture import D1ExcludedCaptureService
from entropic_substrate.d1_excluded_operator import D1ExcludedOperator
from entropic_substrate.d1_observability import AuthoritativeJournalOnlyObserver
from entropic_substrate.d1_operator_controller import D1ResumeOperatorHistory
from entropic_substrate.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    attest_d1_staged_trainer_request,
)
from entropic_substrate.d1_resume_trainer import run_d1_resume
from entropic_substrate.d1_trainer_request_codec import (
    D1ResumeTrainerRequest,
    parse_d1_trainer_request,
    serialize_d1_trainer_request,
)
from entropic_substrate.evidence.d1_journal import AuthoritativeRunJournal, read_journal
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.evidence.resource_provenance import SegmentStartMode, execution_segment_id
from entropic_substrate.experiments.d1_launch_attempts import D1LaunchAttemptStatus

from .test_d1_excluded_capture import CapturePorts
from .test_d1_excluded_operator import LivePilotTrainer, pilot_case
from .test_d1_operator_controller import _Clock, _Provider, _TrainerClock

pytestmark = pytest.mark.unit


class ResumePilotTrainer(LivePilotTrainer):
    def stage_inputs(self, pod_id, *, request):
        self.staged_request = request
        self.events.append("stage_inputs")
        artifact = publish_d1_atomic_bytes(
            self.root,
            "resume-request.json",
            serialize_d1_trainer_request(request),
            label="pilot resume",
        )
        return attest_d1_staged_trainer_request(request=request, artifact=artifact)

    def await_exported_handoff(self, pod_id: str) -> D1ExportedTrainerHandoff:
        self.events.append("await_handoff")
        assert self.staged_request is not None
        request = parse_d1_trainer_request(
            serialize_d1_trainer_request(self.staged_request),
            catalog=(self.staged_request.plan.profile,),
        )
        assert type(request) is D1ResumeTrainerRequest
        locations = request.locations.resume(self.root)
        completion = run_d1_resume(
            plan=request.plan,
            start=request.start,
            origin=request.origin,
            locations=locations,
            observer=AuthoritativeJournalOnlyObserver(),
            clock=_TrainerClock(request.start.trainer_started_at),
        )
        return D1ExportedTrainerHandoff(completion, locations.journal_path)


def _interrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first, intent, provider, trainer, events = pilot_case(tmp_path)
    real_append = AuthoritativeRunJournal.append_checkpoint

    def power_loss(journal, checkpoint_sha256):
        state = real_append(journal, checkpoint_sha256)
        if state.successful_step == 1:
            raise SystemExit("simulated host loss after durable checkpoint")
        return state

    with monkeypatch.context() as patch:
        patch.setattr(AuthoritativeRunJournal, "append_checkpoint", power_loss)
        with pytest.raises(SystemExit, match="host loss"):
            first.execute_fresh(intent)
    request = trainer.staged_request
    assert request is not None
    checkpoint = tmp_path / request.locations.checkpoint_path
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    prior_start = request.start
    prior_end = prior_start.trainer_started_at + timedelta(seconds=100)
    termination = provider.terminate("pod-123", start=prior_start, trainer_ended_at=prior_end)
    history = D1ResumeOperatorHistory(
        (prior_start,), (termination,), (provider.billing("pod-123", start=prior_start),)
    )
    next_start = replace(
        prior_start,
        execution_segment_index=1,
        execution_segment_id=execution_segment_id(prior_start.logical_run_id, 1),
        segment_start_mode=SegmentStartMode.RESUME,
        resume_checkpoint_sha256=digest,
        pod_created_at=prior_end + timedelta(seconds=2),
        trainer_started_at=prior_end + timedelta(seconds=3),
    )
    next_intent = replace(intent, execution_segment_index=1, resume_checkpoint_sha256=digest)
    next_trainer = ResumePilotTrainer(None, events, next_start)
    next_trainer.root = tmp_path
    next_provider = _Provider(next_start, events)
    resumed = D1ExcludedOperator(
        attempt_ledger=first.attempt_ledger,
        runpod=next_provider,
        trainer=next_trainer,
        custody=first._custody,
        clock=_Clock(next_start.pod_created_at),
    )
    return resumed, next_intent, history, next_provider, events


def test_real_pilot_resume_keeps_checkpoint_plan_and_one_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator, intent, history, provider, _events = _interrupted(tmp_path, monkeypatch)
    original = operator.attempt_ledger.events
    result = operator.execute_resume(intent, history=history)
    assert result.plan.plan_digest == intent.plan.plan_digest
    assert result.attempt_ledger.events[:-1] == original
    assert result.attempt_ledger.events[-1].status is D1LaunchAttemptStatus.COMPLETED
    assert provider.delete_count == 1
    events = read_journal(Path(result.journal.path))
    assert sum(event.event_kind is JournalEventKind.RUN_RESUME for event in events) == 1
    assert events[-1].event_kind is JournalEventKind.RUN_END
    assert result.export.handoff.resumed_from_sha256 == intent.resume_checkpoint_sha256
    ports = CapturePorts(tmp_path)
    capture = D1ExcludedCaptureService(
        repo_root=tmp_path, source_root=tmp_path, dvc=ports, cloud=ports, git=ports
    )
    receipt = capture.finalize(capture.prepare(result))
    assert receipt.read_journal_bytes() == Path(result.journal.path).read_bytes()


@pytest.mark.parametrize("mutation", ["segment", "logical", "billing", "terminal", "seed"])
def test_invalid_resume_history_is_rejected_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    operator, intent, history, _provider, events = _interrupted(tmp_path, monkeypatch)
    if mutation == "segment":
        intent = replace(intent, execution_segment_index=2)
    elif mutation == "logical":
        intent = replace(intent, plan=replace(intent.plan, logical_run_id="another-pilot"))
    elif mutation == "billing":
        from d1_billing_finalizer_fixtures import partial_provider_bill

        history = replace(history, billings=(partial_provider_bill(history.starts[0]),))
    elif mutation == "terminal":
        identity = operator.attempt_ledger.events[-1].identity
        operator.attempt_ledger = operator.attempt_ledger.transition(
            identity,
            status=D1LaunchAttemptStatus.COMPLETED,
            observed_at=history.terminations[-1].pod_terminated_at,
        )
    else:
        run = intent.plan.run
        changed = replace(run, execution=replace(run.execution, seed=run.execution.seed + 1))
        intent = replace(intent, plan=replace(intent.plan, run=changed))
    before = list(events)
    with pytest.raises(ValueError):
        operator.execute_resume(intent, history=history)
    assert events == before
