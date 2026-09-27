"""The pod-side D1 mailbox worker executes one marker-authorized attempt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from entropic_substrate.kernel.artifacts import publish_d1_atomic_bytes
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxCommand,
    d1_trainer_mailbox_attempt_prefix,
    d1_trainer_mailbox_command_path,
    parse_d1_trainer_mailbox_error,
    serialize_d1_trainer_mailbox_command,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_worker import (
    D1TrainerMailboxWaitPolicy,
    D1TrainerMailboxWorker,
    D1TrainerMailboxWorkerError,
)
from entropic_substrate.ladder.run.transport.d1_trainer_process_contracts import (
    d1_trainer_runtime_report_path,
    d1_trainer_start_report_path,
)
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    d1_trainer_request_sha256,
)

from .test_d1_trainer_process_worker import _fresh_request, _stage, _worker

pytestmark = pytest.mark.unit

_POD = "pod-worker-123"
_ATTEMPT = "a" * 64


def _publish_command(volume: Path, command: D1TrainerMailboxCommand) -> None:
    relative = d1_trainer_mailbox_command_path(command)
    (volume / Path(relative).parent).mkdir(parents=True, exist_ok=True)
    payload = serialize_d1_trainer_mailbox_command(command)
    publish_d1_atomic_bytes(volume, relative, payload, label="test mailbox command")


def _attempt_root(volume: Path, attempt: str = _ATTEMPT) -> Path:
    return volume / d1_trainer_mailbox_attempt_prefix(_POD, attempt)


def test_mailbox_worker_drives_real_tiny_trainer_from_committed_markers(
    tmp_path: Path,
) -> None:
    volume = tmp_path / "volume"
    volume.mkdir()
    root = _attempt_root(volume)
    root.mkdir(parents=True)
    request = _fresh_request(root)
    _stage(root, request)
    request_sha = d1_trainer_request_sha256(request)
    bootstrap = D1TrainerMailboxCommand.bootstrap(
        pod_id=_POD,
        attempt_sha256=_ATTEMPT,
    )
    start = D1TrainerMailboxCommand.start(
        pod_id=_POD,
        attempt_sha256=_ATTEMPT,
        request_sha256=request_sha,
    )
    _publish_command(volume, bootstrap)
    _publish_command(volume, start)
    mailbox = D1TrainerMailboxWorker(
        volume_root=volume,
        pod_id=_POD,
        worker_factory=lambda artifact_root: _worker(artifact_root, request),
        wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
    )

    result = mailbox.run()

    assert result["handoff_kind"] == "fresh_completion"
    assert result["successful_step"] == 2
    assert (root / d1_trainer_runtime_report_path(_ATTEMPT)).is_file()
    assert (root / d1_trainer_start_report_path(_ATTEMPT, request_sha)).is_file()
    assert (root / f"{request.locations.checkpoint_path}.complete.json").is_file()
    with pytest.raises(D1TrainerMailboxWorkerError, match="already ran"):
        mailbox.run()


def test_mailbox_worker_rejects_foreign_bootstrap_and_publishes_safe_error(
    tmp_path: Path,
) -> None:
    volume = tmp_path / "volume"
    volume.mkdir()
    root = _attempt_root(volume)
    root.mkdir(parents=True)
    foreign = D1TrainerMailboxCommand.bootstrap(
        pod_id="foreign-pod",
        attempt_sha256=_ATTEMPT,
    )
    path = d1_trainer_mailbox_command_path(
        D1TrainerMailboxCommand.bootstrap(pod_id=_POD, attempt_sha256=_ATTEMPT)
    )
    (volume / Path(path).parent).mkdir(parents=True)
    publish_d1_atomic_bytes(
        volume,
        path,
        serialize_d1_trainer_mailbox_command(foreign),
        label="foreign mailbox command",
    )
    factory_calls: list[Path] = []

    with pytest.raises(D1TrainerMailboxWorkerError, match="bootstrap"):
        D1TrainerMailboxWorker(
            volume_root=volume,
            pod_id=_POD,
            worker_factory=lambda artifact_root: factory_calls.append(artifact_root),
            wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
        ).run()

    assert factory_calls == []
    error = parse_d1_trainer_mailbox_error((root / "trainer-error.json").read_bytes())
    assert error.pod_id == _POD
    assert error.attempt_sha256 == _ATTEMPT
    assert error.stage == "bootstrap"
    assert error.error_type == "ValueError"


@dataclass
class _FakeTime:
    value: float = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def test_mailbox_worker_times_out_without_provider_or_worker_effects(tmp_path: Path) -> None:
    volume = tmp_path / "volume"
    volume.mkdir()
    observed: list[Path] = []
    timer = _FakeTime()
    worker = D1TrainerMailboxWorker(
        volume_root=volume,
        pod_id=_POD,
        worker_factory=lambda root: observed.append(root),
        wait_policy=D1TrainerMailboxWaitPolicy(0.5, 0.1),
        monotonic=timer.monotonic,
        sleep=timer.sleep,
    )

    with pytest.raises(D1TrainerMailboxWorkerError, match=r"attempt.*timed out"):
        worker.run()

    assert observed == []
    assert timer.value >= 0.5


def test_mailbox_worker_refuses_ambiguous_attempt_namespace_before_factory(
    tmp_path: Path,
) -> None:
    volume = tmp_path / "volume"
    volume.mkdir()
    for digest in (_ATTEMPT, "b" * 64):
        _attempt_root(volume, digest).mkdir(parents=True)
    observed: list[Path] = []

    with pytest.raises(D1TrainerMailboxWorkerError, match="multiple attempts"):
        D1TrainerMailboxWorker(
            volume_root=volume,
            pod_id=_POD,
            worker_factory=lambda root: observed.append(root),
            wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
        ).run()

    assert observed == []


@pytest.mark.parametrize(
    ("maximum", "interval"),
    [(0.0, 0.1), (1.0, 0.0), (1.0, 2.0), (float("inf"), 0.1)],
)
def test_mailbox_wait_policy_requires_finite_positive_bounds(
    maximum: float,
    interval: float,
) -> None:
    with pytest.raises(ValueError):
        D1TrainerMailboxWaitPolicy(maximum, interval)
