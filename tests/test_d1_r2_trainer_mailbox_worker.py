"""The pod-side R2 worker executes and exports one real D1 trainer attempt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

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
    d1_trainer_request_sha256,
    serialize_d1_trainer_request,
)

from .test_d1_trainer_object_transport import _ObjectStore
from .test_d1_trainer_process_worker import _ATTEMPT, _POD, _fresh_request, _worker

pytestmark = pytest.mark.unit


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
        wait_policy=D1TrainerMailboxWaitPolicy(0.5, 0.1),
        monotonic=timer.monotonic,
        sleep=timer.sleep,
    )

    with pytest.raises(D1R2TrainerMailboxWorkerError, match="timed_out"):
        worker.run()
    assert calls == [] and timer.value >= 0.5
