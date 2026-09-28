"""Watchdog policy protects task-owned resources without creating or guessing ownership."""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from tests.test_h_probe_control import root
from tests.test_h_probe_launch import launch

from entropic_substrate.ladder.run.h_probe_control import (
    ProbeControlPublisher,
    own_probe_role,
    read_probe_control,
)
from entropic_substrate.ladder.run.h_probe_launch import H0ProbeLaunch
from entropic_substrate.ladder.run.h_probe_watchdog import (
    UNCERTAIN_ABSENT_CONFIRMATIONS,
    ProbeWatchdogFailure,
    run_probe_watchdog,
)
from entropic_substrate.ladder.run.h_runpod_transport import (
    ProbeAccount,
    ProbePod,
    ProbeProviderError,
)

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
    controller: ProbeControlPublisher,
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


def test_uncertain_absence_confirmation_threshold_is_pinned_at_three() -> None:
    """K=3 is the spec's literal choice (FOR-1566); other tests derive from the
    constant symbolically, so they cannot themselves catch a change to its value."""
    assert UNCERTAIN_ABSENT_CONFIRMATIONS == 3


def test_cancelled_uncertain_absence_resolves_after_exactly_k_valid_empty_reads(
    tmp_path: Path,
) -> None:
    spec = replace(launch(), controller_timeout_seconds=50, watchdog_timeout_seconds=100)
    directory, provider = root(tmp_path), Provider(spec)
    with own_probe_role(directory, spec, "controller") as controller:
        declare(controller, outcome="uncertain")
        controller.write(
            "stop",
            {
                "schema": "h-probe-stop-v1",
                "guard_pid": os.getpid(),
                "reason": "cancelled",
                "pod_id": None,
            },
            max_bytes=10_000,
        )
    clock = [0.0]
    calls = [0]
    original_list_owned = provider.list_owned

    def counting_list_owned(spec_arg: H0ProbeLaunch) -> tuple[ProbePod, ...]:
        calls[0] += 1
        return original_list_owned(spec_arg)

    provider.list_owned = counting_list_owned

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
    assert result["status"] == "cancelled_uncertain_absent"
    assert result["reason"] == "cancelled" and result["pod_id"] is None
    assert result["termination_verified"] is True
    assert result["attributed_pod_absent"] is True
    assert result["inventory_verified"] is True
    assert result["unattributed_pod_ids"] == []
    # Resolves the instant the K-th empty read lands, before any (K)-th wait().
    assert result["elapsed_seconds"] == UNCERTAIN_ABSENT_CONFIRMATIONS - 1
    # +1 for the pre-loop arming check (provider.list_owned at entry, before the
    # preexisting-candidates guard), which this counting fixture also observes.
    assert calls[0] == UNCERTAIN_ABSENT_CONFIRMATIONS + 1


def test_k_minus_one_valid_empty_reads_do_not_resolve_the_uncertain_absence(
    tmp_path: Path,
) -> None:
    spec = replace(
        launch(),
        controller_timeout_seconds=1,
        watchdog_timeout_seconds=UNCERTAIN_ABSENT_CONFIRMATIONS - 1,
    )
    directory, provider = root(tmp_path), Provider(spec)
    with own_probe_role(directory, spec, "controller") as controller:
        declare(controller, outcome="uncertain")
        controller.write(
            "stop",
            {
                "schema": "h-probe-stop-v1",
                "guard_pid": os.getpid(),
                "reason": "cancelled",
                "pod_id": None,
            },
            max_bytes=10_000,
        )
    clock = [0.0]

    def wait(seconds: float) -> None:
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


def test_uncertain_absence_count_starts_after_the_cancelled_stop_not_after_intent(
    tmp_path: Path,
) -> None:
    spec = replace(launch(), controller_timeout_seconds=50, watchdog_timeout_seconds=100)
    directory, provider = root(tmp_path), Provider(spec)
    with own_probe_role(directory, spec, "controller") as controller:
        declare(controller, outcome="uncertain")  # intent + allocation only, no stop yet
    clock = [0.0]
    waits = [0]
    pre_stop_polls = 2

    def wait(seconds: float) -> None:
        waits[0] += 1
        if waits[0] == pre_stop_polls:
            with own_probe_role(directory, spec, "controller") as controller:
                controller.write(
                    "stop",
                    {
                        "schema": "h-probe-stop-v1",
                        "guard_pid": os.getpid(),
                        "reason": "cancelled",
                        "pod_id": None,
                    },
                    max_bytes=10_000,
                )
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
    assert result["status"] == "cancelled_uncertain_absent"
    # Empty reads before the stop never count: elapsed = pre-stop polls + (K - 1).
    assert result["elapsed_seconds"] == pre_stop_polls + UNCERTAIN_ABSENT_CONFIRMATIONS - 1


def test_candidate_at_any_poll_permanently_disables_the_uncertain_absence_exit(
    tmp_path: Path,
) -> None:
    spec = replace(
        launch(),
        controller_timeout_seconds=1,
        watchdog_timeout_seconds=2 * UNCERTAIN_ABSENT_CONFIRMATIONS + 4,
    )
    directory, provider = root(tmp_path), Provider(spec)
    with own_probe_role(directory, spec, "controller") as controller:
        declare(controller, outcome="uncertain")
        controller.write(
            "stop",
            {
                "schema": "h-probe-stop-v1",
                "guard_pid": os.getpid(),
                "reason": "cancelled",
                "pod_id": None,
            },
            max_bytes=10_000,
        )
    calls = [0]

    def list_owned(spec_arg: H0ProbeLaunch) -> tuple[ProbePod, ...]:
        calls[0] += 1
        if calls[0] == 2:
            return (
                ProbePod(
                    "owned-pod",
                    spec_arg.pod_name,
                    str(spec_arg.request.image),
                    1,
                    "RUNNING",
                    "running",
                    1,
                    "a" * 64,
                ),
            )
        return ()

    provider.list_owned = list_owned
    clock = [0.0]

    def wait(seconds: float) -> None:
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
    # Plenty of consecutive empty reads followed the transient candidate, and it
    # still never resolves: the exit is disabled for the rest of this run.
    assert error.value.record["status"] == "cleanup_unresolved"
    assert error.value.record["termination_verified"] is False
    assert calls[0] > UNCERTAIN_ABSENT_CONFIRMATIONS + 1


def test_retryable_provider_error_resets_the_uncertain_absence_count(tmp_path: Path) -> None:
    spec = replace(launch(), controller_timeout_seconds=50, watchdog_timeout_seconds=100)
    directory, provider = root(tmp_path), Provider(spec)
    with own_probe_role(directory, spec, "controller") as controller:
        declare(controller, outcome="uncertain")
        controller.write(
            "stop",
            {
                "schema": "h-probe-stop-v1",
                "guard_pid": os.getpid(),
                "reason": "cancelled",
                "pod_id": None,
            },
            max_bytes=10_000,
        )
    calls = [0]

    def list_owned(_spec_arg: H0ProbeLaunch) -> tuple[ProbePod, ...]:
        calls[0] += 1
        if calls[0] == 2:
            raise ProbeProviderError("network_error")
        return ()

    provider.list_owned = list_owned
    clock = [0.0]

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
    assert result["status"] == "cancelled_uncertain_absent"
    assert result["provider_error_count"] == 1 and result["last_error"] == "network_error"
    assert result["guard_fault"] is None
    # Read 2 errored and reset the count; K fresh reads afterward still resolve it,
    # landing one poll later than the error-free case (calls 1,[2 error],3,4,5).
    assert calls[0] == UNCERTAIN_ABSENT_CONFIRMATIONS + 2
    assert result["elapsed_seconds"] == UNCERTAIN_ABSENT_CONFIRMATIONS


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
