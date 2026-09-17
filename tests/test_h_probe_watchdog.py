"""Watchdog policy protects task-owned resources without creating or guessing ownership."""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from tests.test_h_probe_control import root
from tests.test_h_probe_launch import launch

from entropic_substrate.h_probe_control import (
    _ProbeControlWriter,
    own_probe_role,
    read_probe_control,
)
from entropic_substrate.h_probe_launch import H0ProbeLaunch
from entropic_substrate.h_probe_watchdog import ProbeWatchdogFailure, run_probe_watchdog
from entropic_substrate.h_runpod_transport import ProbeAccount, ProbePod, ProbeProviderError

pytestmark = pytest.mark.unit


@dataclass
class Provider:
    launch: H0ProbeLaunch
    present: bool = False
    account_id: str = "account-fixture"
    delete_lag: int = 0
    deletions: list[str] = field(default_factory=list)

    def account(self) -> ProbeAccount:
        return ProbeAccount(self.account_id, 20, 0)

    def check_template(self, launch: H0ProbeLaunch) -> None:
        assert launch == self.launch

    def list_owned(self, launch: H0ProbeLaunch) -> tuple[ProbePod, ...]:
        assert launch == self.launch
        if not self.present:
            return ()
        return (
            ProbePod(
                "owned-pod",
                launch.pod_name,
                str(launch.request.image),
                1,
                "RUNNING",
                "running",
                1,
                "a" * 64,
            ),
        )

    def get_owned(self, launch: H0ProbeLaunch, pod_id: str) -> ProbePod | None:
        assert pod_id == "owned-pod"
        found = self.list_owned(launch)
        if self.deletions and self.delete_lag:
            self.delete_lag -= 1
            if not self.delete_lag:
                self.present = False
        return found[0] if found else None

    def delete_owned(self, launch: H0ProbeLaunch, pod_id: str) -> str:
        assert launch == self.launch and pod_id == "owned-pod"
        self.deletions.append(pod_id)
        if not self.delete_lag:
            self.present = False
        return "deletion_acknowledged"


def declare(
    controller: _ProbeControlWriter,
    *,
    outcome: str = "created",
    stop: bool = False,
    pid: int | None = None,
) -> None:
    guard_pid = os.getpid() if pid is None else pid
    controller.write(
        "create-intent",
        {"schema": "h-probe-create-intent-v1", "guard_pid": guard_pid},
        max_bytes=10_000,
    )
    controller.write(
        "allocation",
        {
            "schema": "h-probe-allocation-v1",
            "guard_pid": guard_pid,
            "outcome": outcome,
            "pod_id": "owned-pod" if outcome == "created" else None,
            "response_sha256": "a" * 64 if outcome == "created" else None,
        },
        max_bytes=10_000,
    )
    if stop:
        controller.write(
            "stop",
            {
                "schema": "h-probe-stop-v1",
                "guard_pid": guard_pid,
                "reason": "capture_verified",
                "pod_id": "owned-pod",
            },
            max_bytes=10_000,
        )


def test_controller_loss_triggers_deadline_cleanup_and_separate_absence_check(
    tmp_path: Path,
) -> None:
    spec = replace(launch(), controller_timeout_seconds=3, watchdog_timeout_seconds=8)
    directory, provider = root(tmp_path), Provider(spec, delete_lag=2)
    clock = [0.0]

    def wait(seconds: float) -> None:
        if clock[0] == 0:
            with own_probe_role(directory, spec, "controller") as controller:
                declare(controller)
            provider.present = True
        clock[0] += seconds

    result = run_probe_watchdog(
        directory,
        spec,
        provider,
        poll_seconds=1,
        max_control_bytes=10_000,
        monotonic=lambda: clock[0],
        wait=wait,
    )
    assert result["termination_verified"] is True and result["reason"] == "controller_deadline"
    assert result["scientific_authority"] == "none"
    assert clock[0] >= 4 and provider.deletions == ["owned-pod"]
    assert read_probe_control(directory, spec, "watchdog", "result", max_bytes=10_000) == result


def test_verified_capture_stop_does_not_wait_for_deadline(tmp_path: Path) -> None:
    spec = replace(launch(), controller_timeout_seconds=3, watchdog_timeout_seconds=6)
    directory, provider = root(tmp_path), Provider(spec)
    clock = [0.0]

    def wait(seconds: float) -> None:
        if clock[0] == 0:
            with own_probe_role(directory, spec, "controller") as controller:
                declare(controller, stop=True)
            provider.present = True
        clock[0] += seconds

    result = run_probe_watchdog(
        directory,
        spec,
        provider,
        poll_seconds=1,
        max_control_bytes=10_000,
        monotonic=lambda: clock[0],
        wait=wait,
    )
    assert result["reason"] == "capture_verified" and result["termination_verified"] is True
    assert clock[0] < 3


def test_uncertain_create_never_adopts_or_deletes_name_only_candidates(tmp_path: Path) -> None:
    spec = replace(launch(), controller_timeout_seconds=2, watchdog_timeout_seconds=4)
    directory, provider = root(tmp_path), Provider(spec)
    clock = [0.0]

    def wait(seconds: float) -> None:
        if clock[0] == 0:
            with own_probe_role(directory, spec, "controller") as controller:
                declare(controller, outcome="uncertain")
            provider.present = True
        clock[0] += seconds

    with pytest.raises(ProbeWatchdogFailure) as error:
        run_probe_watchdog(
            directory,
            spec,
            provider,
            poll_seconds=1,
            max_control_bytes=10_000,
            monotonic=lambda: clock[0],
            wait=wait,
        )
    assert error.value.record["status"] == "cleanup_unresolved"
    assert error.value.record["termination_verified"] is False
    assert error.value.record["unattributed_pod_ids"] == ["owned-pod"]
    assert provider.deletions == []


def test_preexisting_resource_refuses_arming_without_deletion(tmp_path: Path) -> None:
    spec, directory = launch(), root(tmp_path)
    provider = Provider(spec, present=True)
    with pytest.raises(ValueError, match="preexisting"):
        run_probe_watchdog(directory, spec, provider, poll_seconds=1, max_control_bytes=10_000)
    assert provider.deletions == []
    assert read_probe_control(directory, spec, "watchdog", "ready", max_bytes=10_000) is None


@pytest.mark.parametrize("failure", ["account", "foreign_guard", "provider"])
def test_faults_never_certify_termination(tmp_path: Path, failure: str) -> None:
    spec = replace(launch(), controller_timeout_seconds=2, watchdog_timeout_seconds=4)
    directory, provider = root(tmp_path), Provider(spec)
    clock = [0.0]

    def wait(seconds: float) -> None:
        if clock[0] == 0:
            with own_probe_role(directory, spec, "controller") as controller:
                declare(controller, pid=os.getpid() + 1 if failure == "foreign_guard" else None)
            provider.present = True
            if failure == "account":
                provider.account_id = "different-account"
            if failure == "provider":

                def unavailable(_launch: H0ProbeLaunch) -> tuple[ProbePod, ...]:
                    raise ProbeProviderError("network_error")

                provider.list_owned = unavailable
        clock[0] += seconds

    with pytest.raises(ProbeWatchdogFailure) as error:
        run_probe_watchdog(
            directory,
            spec,
            provider,
            poll_seconds=1,
            max_control_bytes=10_000,
            monotonic=lambda: clock[0],
            wait=wait,
        )
    assert error.value.record["termination_verified"] is False
    assert provider.deletions == []


def test_inventory_gap_does_not_block_attributed_pod_cleanup(tmp_path: Path) -> None:
    spec = replace(launch(), controller_timeout_seconds=2, watchdog_timeout_seconds=4)
    directory, provider = root(tmp_path), Provider(spec)
    clock = [0.0]

    def wait(seconds: float) -> None:
        if clock[0] == 0:
            with own_probe_role(directory, spec, "controller") as controller:
                declare(controller)
            provider.present = True

            def unavailable(_launch: H0ProbeLaunch) -> tuple[ProbePod, ...]:
                raise ProbeProviderError("network_error")

            def get_owned(value: H0ProbeLaunch, pod_id: str) -> ProbePod | None:
                assert pod_id == "owned-pod"
                found = Provider.list_owned(provider, value)
                return found[0] if found else None

            provider.list_owned, provider.get_owned = unavailable, get_owned
        clock[0] += seconds

    with pytest.raises(ProbeWatchdogFailure) as error:
        run_probe_watchdog(
            directory,
            spec,
            provider,
            poll_seconds=1,
            max_control_bytes=10_000,
            monotonic=lambda: clock[0],
            wait=wait,
        )
    assert provider.deletions == ["owned-pod"]
    assert error.value.record["attributed_pod_absent"] is True
    assert error.value.record["termination_verified"] is False


@pytest.mark.parametrize("poll", [0, -1, True, float("nan"), float("inf")])
def test_invalid_polling_cannot_arm_watchdog(tmp_path: Path, poll: float) -> None:
    directory, spec = root(tmp_path), launch()
    with pytest.raises(ValueError):
        run_probe_watchdog(
            directory, spec, Provider(spec), poll_seconds=poll, max_control_bytes=10_000
        )
    assert not (directory / "watchdog-ready.json").exists()


def test_expiry_without_intent_does_not_claim_a_terminated_run(tmp_path: Path) -> None:
    spec = replace(launch(), controller_timeout_seconds=2, watchdog_timeout_seconds=4)
    directory, provider, clock = root(tmp_path), Provider(spec), [0.0]

    def wait(seconds: float) -> None:
        clock[0] += seconds

    result = run_probe_watchdog(
        directory,
        spec,
        provider,
        poll_seconds=1,
        max_control_bytes=10_000,
        monotonic=lambda: clock[0],
        wait=wait,
    )
    assert result["status"] == "expired_without_intent" and result["termination_verified"] is False
    assert provider.deletions == []
