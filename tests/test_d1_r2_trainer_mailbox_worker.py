"""The pod-side R2 worker executes and exports one real D1 trainer attempt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from d1_resume_fixtures import captured_step_one_case, origin, resume_start
from entropic_substrate.d1_r2_trainer_mailbox_worker import (
    D1R2TrainerMailboxWorker,
    D1R2TrainerMailboxWorkerError,
)
from entropic_substrate.d1_trainer_evidence_adapter import D1TrainerEvidenceAdapter
from entropic_substrate.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxCommand,
    d1_trainer_mailbox_attempt_prefix,
    d1_trainer_mailbox_command_path,
    d1_trainer_mailbox_error_path,
    parse_d1_trainer_mailbox_error,
    serialize_d1_trainer_mailbox_command,
)
from entropic_substrate.d1_trainer_mailbox_worker import D1TrainerMailboxWaitPolicy
from entropic_substrate.d1_trainer_object_transport import D1ObjectMailboxTrainerTransport
from entropic_substrate.d1_trainer_request_codec import (
    D1ResumeTrainerRequest,
    D1TrainerLocationSpec,
    d1_trainer_request_sha256,
    serialize_d1_trainer_request,
)

from .test_d1_trainer_object_transport import _ObjectStore
from .test_d1_trainer_process_worker import _ATTEMPT, _POD, _fresh_request, _worker

pytestmark = pytest.mark.unit


@dataclass
class _Recovery:
    source_checkpoint: Path | None = None
    source_journal: Path | None = None
    target_checkpoint: Path | None = None
    target_journal: Path | None = None
    fail: bool = False

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def restore(
        self,
        *,
        logical_run_id: str,
        expected_checkpoint_sha256: str,
    ) -> None:
        self.calls.append((logical_run_id, expected_checkpoint_sha256))
        if self.fail:
            raise RuntimeError("provider-private-detail")
        values = (
            self.source_checkpoint,
            self.source_journal,
            self.target_checkpoint,
            self.target_journal,
        )
        if all(path is not None for path in values):
            source_checkpoint, source_journal, target_checkpoint, target_journal = values
            assert target_checkpoint is not None and target_journal is not None
            assert source_checkpoint is not None and source_journal is not None
            target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            target_checkpoint.write_bytes(source_checkpoint.read_bytes())
            target_journal.write_bytes(source_journal.read_bytes())


def _seed(store: _ObjectStore, request) -> str:  # type: ignore[no-untyped-def]
    prefix = d1_trainer_mailbox_attempt_prefix(_POD, _ATTEMPT)
    bootstrap = D1TrainerMailboxCommand.bootstrap(pod_id=_POD, attempt_sha256=_ATTEMPT)
    request_payload = serialize_d1_trainer_request(request)
    request_sha = d1_trainer_request_sha256(request)
    start = D1TrainerMailboxCommand.start(
        pod_id=_POD,
        attempt_sha256=_ATTEMPT,
        request_sha256=request_sha,
    )
    store.seed(
        d1_trainer_mailbox_command_path(bootstrap), serialize_d1_trainer_mailbox_command(bootstrap)
    )
    store.seed(f"{prefix}/requests/{_ATTEMPT}.json", request_payload)
    store.seed(d1_trainer_mailbox_command_path(start), serialize_d1_trainer_mailbox_command(start))
    return request_sha


def test_r2_worker_and_operator_adapter_complete_real_tiny_round_trip(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    request = _fresh_request(repo)
    store = _ObjectStore()
    request_sha = _seed(store, request)
    recovery = _Recovery()
    worker = D1R2TrainerMailboxWorker(
        repo_root=repo,
        pod_id=_POD,
        objects=store,
        catalog=(request.plan.profile,),
        worker_factory=lambda publisher: _worker(
            repo,
            request,
            start_report_publisher=publisher,
        ),
        recovery=recovery,
        wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
    )

    result = worker.run()

    assert result["handoff_kind"] == "fresh_completion"
    prefix = d1_trainer_mailbox_attempt_prefix(_POD, _ATTEMPT)
    assert f"{prefix}/trainer-runtime-{_ATTEMPT}.json" in store.values
    assert any(key.startswith(f"{prefix}/trainer-start-") for key in store.values)
    terminal_marker = f"{prefix}/{request.locations.checkpoint_path}.complete.json"
    assert terminal_marker in store.values
    assert store.publishes[-1] == terminal_marker

    materialized = tmp_path / "materialized"
    materialized.mkdir()
    transport = D1ObjectMailboxTrainerTransport(
        objects=store,
        materialization_root=materialized,
        wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
    )
    adapter = D1TrainerEvidenceAdapter(transport=transport, catalog=(request.plan.profile,))
    adapter.bootstrap_runtime(_POD, _ATTEMPT)
    adapter.stage_inputs(_POD, request=request)
    started = adapter.start_trainer(_POD)
    candidate = adapter.await_exported_handoff(_POD)
    assert started.request_sha256 == request_sha
    assert candidate.handoff.successful_step == 2
    assert candidate.journal_path.is_relative_to(materialized)
    assert recovery.calls == []


def test_r2_worker_restores_real_resume_before_trainer_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    case = captured_step_one_case(source, monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    start = resume_start(case.plan, case.checkpoint_sha256)
    locations = D1TrainerLocationSpec(
        "results/resume/journal.jsonl",
        "results/resume/checkpoint.pt",
        "results/resume/quarantined-tail.jsonl",
    )
    request = D1ResumeTrainerRequest(case.plan, start, origin(), locations)
    store = _ObjectStore()
    _seed(store, request)
    recovery = _Recovery(
        case.checkpoint_path,
        case.journal_path,
        repo / locations.checkpoint_path,
        repo / locations.journal_path,
    )
    worker = D1R2TrainerMailboxWorker(
        repo_root=repo,
        pod_id=_POD,
        objects=store,
        catalog=(request.plan.profile,),
        worker_factory=lambda publisher: _worker(
            repo,
            request,
            start_report_publisher=publisher,
        ),
        recovery=recovery,
        wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
    )

    result = worker.run()

    assert result["handoff_kind"] == "resume_completion"
    assert result["successful_step"] == 2
    assert recovery.calls == [(request.plan.logical_run_id, case.checkpoint_sha256)]


def test_r2_worker_publishes_recovery_failure_before_any_start_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    case = captured_step_one_case(source, monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    start = resume_start(case.plan, case.checkpoint_sha256)
    request = D1ResumeTrainerRequest(
        case.plan,
        start,
        origin(),
        D1TrainerLocationSpec(
            "results/resume/journal.jsonl",
            "results/resume/checkpoint.pt",
            "results/resume/quarantined-tail.jsonl",
        ),
    )
    store = _ObjectStore()
    _seed(store, request)
    recovery = _Recovery(fail=True)
    worker = D1R2TrainerMailboxWorker(
        repo_root=repo,
        pod_id=_POD,
        objects=store,
        catalog=(request.plan.profile,),
        worker_factory=lambda publisher: _worker(
            repo,
            request,
            start_report_publisher=publisher,
        ),
        recovery=recovery,
        wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
    )

    with pytest.raises(D1R2TrainerMailboxWorkerError, match="recovery"):
        worker.run()

    report = parse_d1_trainer_mailbox_error(
        store.values[d1_trainer_mailbox_error_path(_POD, _ATTEMPT)]
    )
    assert report.stage == "recovery" and report.error_type == "RuntimeError"
    assert not any("trainer-start-" in key for key in store.values)


def test_r2_worker_refuses_multiple_attempts_before_factory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    request = _fresh_request(repo)
    store = _ObjectStore()
    _seed(store, request)
    foreign_attempt = "b" * 64
    foreign = D1TrainerMailboxCommand.bootstrap(
        pod_id=_POD,
        attempt_sha256=foreign_attempt,
    )
    store.seed(
        d1_trainer_mailbox_command_path(foreign), serialize_d1_trainer_mailbox_command(foreign)
    )
    calls: list[object] = []

    with pytest.raises(D1R2TrainerMailboxWorkerError, match="multiple_attempts"):
        D1R2TrainerMailboxWorker(
            repo_root=repo,
            pod_id=_POD,
            objects=store,
            catalog=(request.plan.profile,),
            worker_factory=lambda publisher: calls.append(publisher),
            recovery=_Recovery(),
            wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
        ).run()
    assert calls == []


def test_r2_worker_publishes_sanitized_error_after_attempt_discovery(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    request = _fresh_request(repo)
    store = _ObjectStore()
    _seed(store, request)
    private_detail = "provider-private-detail"

    with pytest.raises(D1R2TrainerMailboxWorkerError) as raised:
        D1R2TrainerMailboxWorker(
            repo_root=repo,
            pod_id=_POD,
            objects=store,
            catalog=(request.plan.profile,),
            worker_factory=lambda _publisher: (_ for _ in ()).throw(RuntimeError(private_detail)),
            recovery=_Recovery(),
            wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
        ).run()

    assert private_detail not in str(raised.value)
    error_key = d1_trainer_mailbox_error_path(_POD, _ATTEMPT)
    report = parse_d1_trainer_mailbox_error(store.values[error_key])
    assert report.stage == "bootstrap" and report.error_type == "RuntimeError"


@dataclass
class _FakeTime:
    value: float = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def test_r2_worker_attempt_wait_is_bounded_without_factory_effects(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    request = _fresh_request(repo)
    store = _ObjectStore()
    timer = _FakeTime()
    calls: list[object] = []
    worker = D1R2TrainerMailboxWorker(
        repo_root=repo,
        pod_id=_POD,
        objects=store,
        catalog=(request.plan.profile,),
        worker_factory=lambda publisher: calls.append(publisher),
        recovery=_Recovery(),
        wait_policy=D1TrainerMailboxWaitPolicy(0.5, 0.1),
        monotonic=timer.monotonic,
        sleep=timer.sleep,
    )

    with pytest.raises(D1R2TrainerMailboxWorkerError, match="timed_out"):
        worker.run()
    assert calls == [] and timer.value >= 0.5
