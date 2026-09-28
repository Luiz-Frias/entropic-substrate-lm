"""Controller composition tests use explicit fake provider/guardian observations."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from tests.test_h_probe_launch import launch

from entropic_substrate.ladder.run.h_probe_process import ProbeWatchdogProcess
from entropic_substrate.ladder.run.h_probe_watchdog import UNCERTAIN_ABSENT_CONFIRMATIONS
from entropic_substrate.ladder.run.h_runpod_transport import (
    ProbeAccount,
    ProbePod,
    ProbeProviderError,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker.h_probe_allocation import (
    ProbeAllocationHandoffError,
)
from entropic_substrate.ladder.run.worker.h_probe_controller import (
    ProbeControllerFailure,
    _termination,
    run_probe_controller,
)

pytestmark = pytest.mark.unit


class Control:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def write(self, name: str, payload: dict[str, object], *, max_bytes: int) -> None:
        assert max_bytes > 0 and name not in self.records
        self.records[name] = payload


class Guard:
    def __init__(self, root: Path) -> None:
        self.root, self.launch, self.pid = root, launch(), 123
        self.control = Control()
        self.config = SimpleNamespace(
            max_control_bytes=100_000,
            shutdown_timeout_seconds=1,
            provider_timeout_seconds=1,
            poll_seconds=1,
        )
        self.ready = {"account_id": "fixture", "controller_deadline": 3.0, "watchdog_deadline": 6.0}
        self.alive = True
        self.joined = False
        self.result: dict[str, object] = {
            "schema": "h-probe-watchdog-result-v1",
            "scientific_authority": "none",
            "status": "terminated",
            "pod_id": "owned-pod",
            "termination_verified": True,
            "attributed_pod_absent": True,
            "inventory_verified": True,
            "unattributed_pod_ids": [],
            "guard_fault": None,
        }

    def require_alive(self) -> None:
        if not self.alive:
            raise ValueError("fixture guardian lost")

    def wait_result(self, *, timeout_seconds: float) -> dict[str, object]:
        assert timeout_seconds > 0 and "stop" in self.control.records
        self.alive, self.joined = False, True
        return self.result


class Provider:
    def __init__(self, guard: Guard) -> None:
        self.guard = guard
        self.deleted: list[str] = []
        self.present = True
        self.pod = ProbePod(
            "owned-pod",
            guard.launch.pod_name,
            str(guard.launch.request.image),
            1,
            "RUNNING",
            None,
            0.49,
            "a" * 64,
        )

    def account(self) -> ProbeAccount:
        return ProbeAccount("fixture", 20, 0)

    def check_template(self, spec: object) -> None:
        assert spec == self.guard.launch

    def create_once(self, spec: object) -> ProbePod:
        assert spec == self.guard.launch
        return self.pod

    def get_owned(self, spec: object, pod_id: str) -> ProbePod | None:
        assert spec == self.guard.launch and pod_id == self.pod.pod_id
        return self.pod if self.present else None

    def delete_owned(self, spec: object, pod_id: str) -> str:
        assert spec == self.guard.launch and pod_id == self.pod.pod_id
        self.deleted.append(pod_id)
        self.present = False
        return "deletion_acknowledged"


def setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: object,
) -> tuple[Guard, Provider, list[str], list[float], Callable[[], dict[str, object]]]:
    guard = Guard(tmp_path)
    provider = Provider(guard)
    calls: list[str] = []
    clock = [0.0]

    def allocate(*_args: object) -> ProbePod:
        calls.append("create")
        guard.control.records["create-intent"] = {"guard_pid": guard.pid}
        guard.control.records["allocation"] = {"pod_id": provider.pod.pod_id}
        return provider.pod

    def collect(*_args: object, **kwargs: object) -> object:
        assert kwargs["expected_pod_id"] == provider.pod.pod_id
        calls.append("collect")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_controller.allocate_probe_once", allocate
    )
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_controller.collect_probe_terminal", collect
    )
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_controller.read_probe_control",
        lambda _root, _spec, _role, name, **_kw: guard.control.records.get(name),
    )
    destination = tmp_path / "capture"
    destination.mkdir()

    def run() -> dict[str, object]:
        return run_probe_controller(
            cast(ProbeWatchdogProcess, guard),
            provider,
            cast(D1TrainerMailboxObjectPort, object()),
            destination,
            poll_seconds=1,
            max_hourly_rate=1,
            monotonic=lambda: clock[0],
            wait=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        )

    return guard, provider, calls, clock, run


@pytest.mark.parametrize(
    "outcome,verified",
    [
        ("probe_workload_complete", True),
        ("paused", False),
        ("scientific_gap", False),
        ("execution_defect", False),
    ],
)
def test_terminal_capture_and_termination_are_separate_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    verified: bool,
) -> None:
    terminal = {
        "reported_outcome": outcome,
        "workload_verified": verified,
        "capture_status": "native_completion_receipt_verified"
        if verified
        else "failure_bytes_verified",
    }
    guard, provider, calls, _clock, run = setup(tmp_path, monkeypatch, terminal)
    result = run()
    assert calls == ["create", "collect"] and guard.joined
    assert result["scientific_authority"] == "none" and result["workload_verified"] is verified
    assert result["termination_verified"] is True and result["collection"] == terminal
    assert guard.control.records["stop"]["reason"] == (
        "capture_verified" if verified else "partial_capture"
    )
    assert guard.control.records["result"] == result and not provider.deleted


@pytest.mark.parametrize("failure", [None, ValueError("invalid worker evidence")])
def test_deadline_and_invalid_evidence_still_request_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception | None,
) -> None:
    guard, _provider, calls, clock, run = setup(tmp_path, monkeypatch, failure)
    with pytest.raises(ProbeControllerFailure) as caught:
        run()
    assert calls.count("create") == 1 and guard.joined
    assert caught.value.record["termination_verified"] is True
    assert caught.value.record["workload_verified"] is False
    assert guard.control.records["stop"]["reason"] == "cancelled"
    assert clock[0] == (3 if failure is None else 0)


@pytest.mark.parametrize("rate", [None, 2.0, True, -1.0, float("nan")])
def test_missing_or_excessive_measured_rate_cannot_continue_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rate: float | None,
) -> None:
    guard, provider, calls, _clock, run = setup(tmp_path, monkeypatch, {})
    provider.pod = replace(provider.pod, cost_per_hour=rate)
    with pytest.raises(ProbeControllerFailure):
        run()
    assert calls == ["create"] and guard.joined


@pytest.mark.parametrize(
    "field,value",
    [
        ("pod_id", "foreign-pod"),
        ("termination_verified", False),
        ("unattributed_pod_ids", ["unclaimed"]),
        ("guard_fault", "failure"),
    ],
)
def test_false_or_foreign_teardown_cannot_complete_the_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    guard, _provider, _calls, _clock, run = setup(
        tmp_path,
        monkeypatch,
        {"workload_verified": True, "capture_status": "native_completion_receipt_verified"},
    )
    guard.result[field] = value
    with pytest.raises(ProbeControllerFailure) as caught:
        run()
    assert caught.value.record["status"] == "controller_failed"
    assert caught.value.record["termination_verified"] is False


def test_guardian_death_attempts_owned_cleanup_but_never_claims_guard_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard, provider, _calls, _clock, run = setup(tmp_path, monkeypatch, None)

    def lost_guard(*_args: object, **_kwargs: object) -> None:
        guard.alive = False
        raise ValueError("guardian died after allocation")

    def failed_join(**_kwargs: object) -> dict[str, object]:
        raise ValueError("guardian exited without result")

    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_controller.collect_probe_terminal", lost_guard
    )
    monkeypatch.setattr(guard, "wait_result", failed_join)
    with pytest.raises(ProbeControllerFailure) as caught:
        run()
    assert provider.deleted == ["owned-pod"] and not provider.present
    assert caught.value.record["termination_verified"] is False
    assert caught.value.record["emergency_cleanup"]["owned_pod_absent"] is True


def test_changed_allocation_record_cannot_redirect_stop_or_emergency_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard, provider, _calls, _clock, run = setup(tmp_path, monkeypatch, None)

    def changed(*_args: object, **_kwargs: object) -> dict[str, object]:
        guard.control.records["allocation"] = {"pod_id": "foreign-pod"}
        return {"workload_verified": True, "capture_status": "native_completion_receipt_verified"}

    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_controller.collect_probe_terminal", changed
    )
    with pytest.raises(ProbeControllerFailure):
        run()
    assert "stop" not in guard.control.records and provider.deleted == ["owned-pod"]


def test_provider_error_code_reaches_the_controller_result_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard, _provider, calls, _clock, run = setup(tmp_path, monkeypatch, None)

    def failing_allocate(*_args: object) -> ProbePod:
        calls.append("create")
        raise ProbeProviderError("network_error", allocation_uncertain=True)

    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_controller.allocate_probe_once",
        failing_allocate,
    )
    with pytest.raises(ProbeControllerFailure) as caught:
        run()
    assert caught.value.record["provider_error_code"] == "network_error"
    assert "ProbeProviderError" in caught.value.record["error_types"]


def test_allocation_handoff_error_still_surfaces_its_wrapped_provider_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard, _provider, calls, _clock, run = setup(tmp_path, monkeypatch, None)
    wrapped = ProbeProviderError("timeout", allocation_uncertain=True)

    def failing_allocate(*_args: object) -> ProbePod:
        calls.append("create")
        raise ProbeAllocationHandoffError("owned-pod", wrapped)

    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_controller.allocate_probe_once",
        failing_allocate,
    )
    with pytest.raises(ProbeControllerFailure) as caught:
        run()
    assert caught.value.record["provider_error_code"] == "timeout"
    assert caught.value.record["pod_id"] == "owned-pod"


_BASE_TERMINATION_FIELDS: dict[str, object] = {
    "schema": "h-probe-watchdog-result-v1",
    "scientific_authority": "none",
    "termination_verified": True,
    "attributed_pod_absent": True,
    "inventory_verified": True,
    "unattributed_pod_ids": [],
    "guard_fault": None,
}


def test_termination_accepts_terminated_only_with_an_attributed_pod() -> None:
    record = {**_BASE_TERMINATION_FIELDS, "status": "terminated", "pod_id": "owned-pod"}
    assert _termination(record, "owned-pod") is True
    # Unchanged prior behavior: pod_id=None never accepts `terminated`, even if
    # the record happens to say pod_id=None too.
    assert _termination({**record, "pod_id": None}, None) is False
    assert _termination(record, "different-pod") is False


def test_termination_accepts_cancelled_uncertain_absent_only_without_a_pod() -> None:
    record = {
        **_BASE_TERMINATION_FIELDS,
        "status": "cancelled_uncertain_absent",
        "pod_id": None,
    }
    assert _termination(record, None) is True
    # A record claiming uncertain-absence while carrying an attributed pod_id,
    # or being asked about an attributed pod_id, is never accepted.
    assert _termination({**record, "pod_id": "owned-pod"}, None) is False
    assert _termination(record, "owned-pod") is False


@pytest.mark.parametrize(
    "field", ["termination_verified", "attributed_pod_absent", "inventory_verified"]
)
def test_termination_requires_every_verified_flag_for_both_statuses(field: str) -> None:
    attributed = {**_BASE_TERMINATION_FIELDS, "status": "terminated", "pod_id": "owned-pod"}
    absent = {**_BASE_TERMINATION_FIELDS, "status": "cancelled_uncertain_absent", "pod_id": None}
    attributed[field] = False
    absent[field] = False
    assert _termination(attributed, "owned-pod") is False
    assert _termination(absent, None) is False


def test_pod_id_none_bounds_the_watchdog_wait_to_the_resolution_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No pod was ever attributed: wait_result must use the watchdog's own K-poll
    uncertain-absent resolution window, never the full cleanup_deadline (which,
    at 11100s in production, is exactly the defect FOR-1566 measured)."""
    guard, _provider, calls, _clock, run = setup(tmp_path, monkeypatch, None)
    # Production sizing (measured 2026-09-28): controller_timeout_seconds=3600,
    # watchdog_timeout_seconds=11100 — the span the defect actually blocked for.
    guard.ready = {**guard.ready, "watchdog_deadline": 11_100.0}

    def failing_allocate(*_args: object) -> ProbePod:
        calls.append("create")
        guard.control.records["create-intent"] = {"guard_pid": guard.pid}
        guard.control.records["allocation"] = {"pod_id": None}
        raise ProbeProviderError("network_error", allocation_uncertain=True, possible_pod_id=None)

    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_controller.allocate_probe_once",
        failing_allocate,
    )
    seen_timeouts: list[float] = []
    original_wait_result = guard.wait_result

    def capturing_wait_result(*, timeout_seconds: float) -> dict[str, object]:
        seen_timeouts.append(timeout_seconds)
        guard.result = {
            "schema": "h-probe-watchdog-result-v1",
            "scientific_authority": "none",
            "status": "cancelled_uncertain_absent",
            "pod_id": None,
            "termination_verified": True,
            "attributed_pod_absent": True,
            "inventory_verified": True,
            "unattributed_pod_ids": [],
            "guard_fault": None,
        }
        return original_wait_result(timeout_seconds=timeout_seconds)

    monkeypatch.setattr(guard, "wait_result", capturing_wait_result)
    with pytest.raises(ProbeControllerFailure) as caught:
        run()
    expected_timeout = (UNCERTAIN_ABSENT_CONFIRMATIONS + 1) * (
        guard.config.poll_seconds + 2 * guard.config.provider_timeout_seconds
    ) + guard.config.shutdown_timeout_seconds
    assert seen_timeouts == [expected_timeout]
    # The reduced window must stay well under the full cleanup deadline it
    # replaces; that gap is the entire point of the fix.
    assert expected_timeout < guard.ready["watchdog_deadline"]
    # The provider error still fails the controller run, but the watchdog's own
    # verified-absent record is now accepted rather than contradicted.
    assert caught.value.record["termination_verified"] is True
    assert caught.value.record["status"] == "controller_failed"


def test_emergency_cleanup_refuses_a_changed_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard, provider, _calls, _clock, run = setup(tmp_path, monkeypatch, None)
    guard.result["termination_verified"] = False
    monkeypatch.setattr(provider, "account", lambda: ProbeAccount("foreign", 20, 0))
    with pytest.raises(ProbeControllerFailure) as caught:
        run()
    assert not provider.deleted and provider.present
    assert caught.value.record["emergency_cleanup"]["status"] == "emergency_failed"
