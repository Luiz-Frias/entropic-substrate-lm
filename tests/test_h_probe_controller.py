"""Controller composition tests use explicit fake provider/guardian observations."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from botocore.exceptions import (
    ClientError,
    ParamValidationError,
    ReadTimeoutError,
)
from tests.test_h_probe_launch import launch, pilot_launch

from entropic_substrate.kernel.runtime.logging import logger
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
from entropic_substrate.ladder.run.transport.d1_trainer_s3_object_adapter import (
    D1TrainerS3ObjectError,
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


@pytest.mark.parametrize(("rate", "admitted"), [(37.0, False), (36.9, True)])
def test_a_pilot_ceiling_past_its_spend_cap_never_reaches_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rate: float, admitted: bool
) -> None:
    """Rung B: 37 USD/h over the fixture's 3900 s watchdog is past the 40 USD cap."""
    guard, provider, calls, _clock, _run = setup(tmp_path, monkeypatch, {})
    guard.launch = pilot_launch()
    destination = tmp_path / "pilot-capture"
    destination.mkdir()

    def run() -> object:
        return run_probe_controller(
            cast(ProbeWatchdogProcess, guard),
            provider,
            cast(D1TrainerMailboxObjectPort, object()),
            destination,
            poll_seconds=1,
            max_hourly_rate=rate,
            monotonic=lambda: 10.0,  # past the controller deadline: no allocation either way
            wait=lambda _: None,
        )

    if admitted:
        with pytest.raises(BaseException) as raised:
            run()
        assert "spend cap" not in str(raised.value)
    else:
        with pytest.raises(ValueError, match="exceeds the spend cap"):
            run()
    assert "create" not in calls


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


@pytest.mark.parametrize("rate", [None, 2.0, True, -1.0, float("nan")])
def test_a_refused_measured_rate_deletes_the_pod_before_the_stop_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rate: float | None,
) -> None:
    """The rate is first readable in the create response, so a refused price is already
    billing: the controller deletes the owned pod at once, ahead of the stop handoff and
    the watchdog join, rather than leaving it to the watchdog's next poll."""
    guard, provider, calls, _clock, run = setup(tmp_path, monkeypatch, {})
    provider.pod = replace(provider.pod, cost_per_hour=rate)
    write, delete, wait = guard.control.write, provider.delete_owned, guard.wait_result

    def recorded_write(name: str, payload: dict[str, object], *, max_bytes: int) -> None:
        calls.append(f"control:{name}")
        write(name, payload, max_bytes=max_bytes)

    def recorded_delete(spec: object, pod_id: str) -> str:
        calls.append(f"delete:{pod_id}")
        return delete(spec, pod_id)

    def recorded_wait(*, timeout_seconds: float) -> dict[str, object]:
        calls.append("join")
        return wait(timeout_seconds=timeout_seconds)

    monkeypatch.setattr(guard.control, "write", recorded_write)
    monkeypatch.setattr(provider, "delete_owned", recorded_delete)
    monkeypatch.setattr(guard, "wait_result", recorded_wait)
    with pytest.raises(ProbeControllerFailure) as caught:
        run()
    assert calls == [
        "create",
        "control:pod",
        "delete:owned-pod",
        "control:stop",
        "join",
        "control:result",
    ]
    record = caught.value.record
    assert record["failure_stages"] == ["price_admission"]
    assert record["emergency_cleanup"] == {
        "status": "emergency_attempt",
        "pod_id": "owned-pod",
        "acknowledgement": "deletion_acknowledged",
        "owned_pod_absent": True,
    }


def test_a_refused_rate_with_a_failed_stop_handoff_deletes_the_pod_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The immediate price-refusal deletion is the emergency attempt; a later stop failure
    must not issue a second delete or overwrite the first attempt's record."""
    guard, provider, _calls, _clock, run = setup(tmp_path, monkeypatch, {})
    provider.pod = replace(provider.pod, cost_per_hour=2.0)
    write = guard.control.write

    def failing_stop(name: str, payload: dict[str, object], *, max_bytes: int) -> None:
        if name == "stop":
            raise OSError("stop custody unavailable")
        write(name, payload, max_bytes=max_bytes)

    monkeypatch.setattr(guard.control, "write", failing_stop)
    with pytest.raises(ProbeControllerFailure) as caught:
        run()
    assert provider.deleted == ["owned-pod"]
    record = caught.value.record
    assert record["failure_stages"][:2] == ["price_admission", "stop_handoff"]
    assert record["emergency_cleanup"]["acknowledgement"] == "deletion_acknowledged"


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


# --------------------------------------------- FOR-1665: transient terminal-probe read
_CONTROLLER = "entropic_substrate.ladder.run.worker.h_probe_controller"
_COMPLETE: dict[str, object] = {
    "reported_outcome": "probe_workload_complete",
    "workload_verified": True,
    "capture_status": "native_completion_receipt_verified",
}


@contextmanager
def _warnings() -> Iterator[list[dict[str, Any]]]:
    """Every substrate record emitted inside the block: level, message and bound extras.

    A copy of ``captured`` in tests/test_h_probe_log_events.py, which imports this module
    (``setup``), so importing it back is circular: the fold the FOR-1667 brief allowed
    "if convenient" is not."""
    records: list[dict[str, Any]] = []

    def sink(message: Any) -> None:
        record = message.record
        records.append(
            {"level": record["level"].name, "message": record["message"], **record["extra"]}
        )

    handler = logger.add(sink, level="DEBUG")
    try:
        yield records
    finally:
        logger.remove(handler)


def _terminal_probe_lines(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("event") == "host.terminal_probe.transient"]


def _script_collect(monkeypatch: pytest.MonkeyPatch, outcomes: list[object]) -> list[int]:
    """Scripted terminal capture: an Exception is raised, the last outcome repeats."""
    calls = [0]

    def collect(*_args: object, **_kwargs: object) -> object:
        outcome = outcomes[min(calls[0], len(outcomes) - 1)]
        calls[0] += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(f"{_CONTROLLER}.collect_probe_terminal", collect)
    return calls


def _object_error(code: str, cause: BaseException | None = None) -> D1TrainerS3ObjectError:
    error = D1TrainerS3ObjectError(operation="read", code=code)
    error.__cause__ = cause
    return error


def _client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "GetObject"
    )


def _timeout_error() -> D1TrainerS3ObjectError:
    """A RECOVERABLE read failure (FOR-1667): the FOR-1665 envelope arms mean this one."""
    return _object_error("provider_failure", ReadTimeoutError(endpoint_url="https://r2.invalid"))


# FOR-1667: the permanent shapes (id -> the error the adapter raises), each fatal on tick one.
_PERMANENT_READS = {
    "response_invalid": lambda: _object_error("response_invalid"),
    "metadata_invalid": lambda: _object_error("metadata_invalid"),
    "byte_limit": lambda: _object_error("byte_limit"),
    "object_bytes": lambda: _object_error("object_bytes"),
    "access_denied_403": lambda: _object_error(
        "provider_failure", _client_error("AccessDenied", 403)
    ),
    "param_validation": lambda: _object_error(
        "provider_failure", ParamValidationError(report="bad Key")
    ),
}
_TRANSIENT_CAUSES = {
    "read_timeout": lambda: _timeout_error(),
    "slow_down_503": lambda: _object_error("provider_failure", _client_error("SlowDown", 503)),
    "unrecognized_cause": lambda: _object_error("provider_failure", RuntimeError("unrecognized")),
    "no_cause": lambda: _object_error("provider_failure"),
}


def test_a_transient_terminal_probe_error_is_a_logged_tick_not_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1 (a): the first tick's marker read raises, the controller keeps polling, collects."""
    guard, provider, _calls, clock, run = setup(tmp_path, monkeypatch, None)
    error = _timeout_error()
    _script_collect(monkeypatch, [error, _COMPLETE])
    with _warnings() as records:
        result = run()
    assert result["status"] == "terminal_collected" and result["failure_stages"] == []
    assert result["collection"] == _COMPLETE and result["error_types"] == []
    assert result["terminal_probe_transient_count"] == 1
    assert result["last_terminal_probe_error"] == error.safe_diagnostic()
    assert clock[0] == 1, "one poll interval passed: no extra sleep, counter abort or timeout"
    assert guard.control.records["stop"]["reason"] == "capture_verified" and not provider.deleted
    lines = _terminal_probe_lines(records)
    assert len(lines) == 1 and lines[0]["level"] == "WARNING"
    assert lines[0]["pod_name"] == guard.launch.pod_name and lines[0]["attempt"] == 1
    diagnostic = error.safe_diagnostic()
    assert lines[0]["operation"] == diagnostic["operation"] == "read"
    assert lines[0]["code"] == diagnostic["code"] == "provider_failure"
    assert lines[0]["provider_code"] == diagnostic["provider_code"]
    assert lines[0]["status"] == diagnostic["http_status"]
    assert lines[0]["message"].startswith("host.terminal_probe.transient pod_name=")
    assert "attempt=1" in lines[0]["message"]


def test_a_run_without_a_transient_probe_error_records_zero_and_no_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _guard, _provider, _calls, _clock, run = setup(tmp_path, monkeypatch, _COMPLETE)
    with _warnings() as records:
        result = run()
    assert result["terminal_probe_transient_count"] == 0
    assert result["last_terminal_probe_error"] is None
    assert not _terminal_probe_lines(records)


def test_a_terminal_probe_that_fails_every_tick_ends_at_the_deadline_not_at_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1 (b), the TRANSIENT arm: a recoverable read failure never ends the run; the
    deadline does. RETRACTED by FOR-1667: this pin once read "the read error is never the
    exit" for EVERY D1 error, which the permanent-code tests below show was wrong (a
    permanent code fails on the first tick); the claim now holds for recoverable reads."""
    guard, provider, _calls, clock, run = setup(tmp_path, monkeypatch, None)
    error = _timeout_error()
    _script_collect(monkeypatch, [error])
    with _warnings() as records, pytest.raises(ProbeControllerFailure) as caught:
        run()
    record = caught.value.record
    assert clock[0] == 3, "polled to the controller deadline, one interval per tick"
    assert record["terminal_probe_transient_count"] == 3
    assert record["last_terminal_probe_error"] == error.safe_diagnostic()
    assert record["failure_stages"] == ["deadline"]
    assert "terminal_capture" not in cast(list[str], record["failure_stages"])
    assert record["error_types"] == ["TimeoutError"] and record["collection"] is None
    assert record["status"] == "controller_failed" and record["termination_verified"] is True
    assert guard.control.records["stop"]["reason"] == "cancelled" and guard.joined
    assert [line["attempt"] for line in _terminal_probe_lines(records)] == [1, 2, 3]
    assert not provider.deleted, "the watchdog (stop handoff), not the probe, removes the pod"


@pytest.mark.parametrize(
    "failure",
    [ValueError("conflicting terminal claims for one worker request"), RuntimeError("other")],
)
def test_only_a_d1_object_error_is_transient_every_other_error_still_fails_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """C1 (d): the conflicting-claims ValueError (and a bare RuntimeError, the D1 error's
    base class) fail the controller on the first tick."""
    guard, _provider, _calls, clock, run = setup(tmp_path, monkeypatch, None)
    _script_collect(monkeypatch, [failure, _COMPLETE])
    with _warnings() as records, pytest.raises(ProbeControllerFailure) as caught:
        run()
    record = caught.value.record
    assert clock[0] == 0 and record["failure_stages"] == ["terminal_capture"]
    assert record["error_types"] == [type(failure).__name__]
    assert record["terminal_probe_transient_count"] == 0
    assert record["last_terminal_probe_error"] is None
    assert not _terminal_probe_lines(records) and guard.control.records["stop"]["reason"] == (
        "cancelled"
    )


def test_a_d1_error_after_the_capture_began_is_not_a_marker_probe_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy that failed mid-capture retains partial files; retrying would meet the
    fresh-destination refusal, so the original error fails the controller loudly."""
    guard, _provider, _calls, clock, run = setup(tmp_path, monkeypatch, None)

    def partial(*args: object, **_kwargs: object) -> object:
        cast(Path, args[2]).joinpath("terminal.json").write_bytes(b"{}")
        raise _timeout_error()

    monkeypatch.setattr(f"{_CONTROLLER}.collect_probe_terminal", partial)
    with _warnings() as records, pytest.raises(ProbeControllerFailure) as caught:
        run()
    record = caught.value.record
    assert clock[0] == 0 and record["failure_stages"] == ["terminal_capture"]
    assert record["error_types"] == ["D1TrainerS3ObjectError"]
    assert record["terminal_probe_transient_count"] == 0
    assert not _terminal_probe_lines(records) and guard.control.records["stop"]["reason"] == (
        "cancelled"
    )


def test_an_absent_pod_with_a_transient_probe_read_defers_to_the_next_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FOR-1665 C7: a tick whose marker read raised is not a None, so provider absence is
    deferred; the next tick re-reads the markers and the published terminal is collected."""
    guard, provider, _calls, clock, run = setup(tmp_path, monkeypatch, None)
    provider.present = False
    _script_collect(monkeypatch, [_timeout_error(), _COMPLETE])
    with _warnings() as records:
        result = run()
    assert result["status"] == "terminal_collected" and result["failure_stages"] == []
    assert result["collection"] == _COMPLETE and result["error_types"] == []
    assert result["terminal_probe_transient_count"] == 1
    assert clock[0] == 1, "the absence was deferred one poll interval, not named on tick one"
    assert not [r for r in records if r.get("event") == "host.provider_absent.observed"]
    assert guard.control.records["stop"]["reason"] == "capture_verified"


# --------------------------------------------- FOR-1667: permanent codes fail fast
@pytest.mark.parametrize("make", _PERMANENT_READS.values(), ids=_PERMANENT_READS.keys())
def test_a_permanent_object_read_code_fails_on_the_first_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make: Callable[[], D1TrainerS3ObjectError]
) -> None:
    """C1 (b)(i): not a logged tick and not the deadline: the read error is the exit."""
    guard, provider, _calls, clock, run = setup(tmp_path, monkeypatch, None)
    error = make()
    _script_collect(monkeypatch, [error])
    with _warnings() as records, pytest.raises(ProbeControllerFailure) as caught:
        run()
    record = caught.value.record
    assert clock[0] == 0, "failed on tick one: no poll interval passed"
    assert record["failure_stages"] == ["terminal_capture"]
    assert record["error_types"] == ["D1TrainerS3ObjectError"]
    assert record["terminal_probe_transient_count"] == 0
    # Ruling (g): the permanent read is not absorbed (count 0, no transient line) but the
    # result names the code that ended the run, like _owned_pod_absent's provider_error_code.
    assert record["last_terminal_probe_error"] == error.safe_diagnostic()
    assert record["collection"] is None
    assert not _terminal_probe_lines(records)
    assert guard.control.records["stop"]["reason"] == "cancelled" and not provider.deleted


@pytest.mark.parametrize("make", _PERMANENT_READS.values(), ids=_PERMANENT_READS.keys())
def test_an_absent_pod_with_a_permanent_read_fails_at_the_read_not_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make: Callable[[], D1TrainerS3ObjectError]
) -> None:
    """C1 (b)(iii): a permanent read propagates before the absence deferral is consulted."""
    _guard, provider, _calls, clock, run = setup(tmp_path, monkeypatch, None)
    provider.present = False
    error = make()
    _script_collect(monkeypatch, [error])
    with _warnings() as records, pytest.raises(ProbeControllerFailure) as caught:
        run()
    record = caught.value.record
    assert clock[0] == 0 and record["failure_stages"] == ["terminal_capture"]
    assert record["error_types"] == ["D1TrainerS3ObjectError"]
    assert record["last_terminal_probe_error"] == error.safe_diagnostic()
    assert not _terminal_probe_lines(records) and record["terminal_probe_transient_count"] == 0
    assert not [r for r in records if r.get("event") == "host.provider_absent.observed"]


@pytest.mark.parametrize("make", _TRANSIENT_CAUSES.values(), ids=_TRANSIENT_CAUSES.keys())
def test_a_recoverable_read_every_tick_keeps_the_for_1665_deadline_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make: Callable[[], D1TrainerS3ObjectError]
) -> None:
    """C1 (b)(ii): connection, server and unrecognized causes stay logged ticks."""
    _guard, _provider, _calls, clock, run = setup(tmp_path, monkeypatch, None)
    error = make()
    _script_collect(monkeypatch, [error])
    with _warnings() as records, pytest.raises(ProbeControllerFailure) as caught:
        run()
    record = caught.value.record
    assert clock[0] == 3 and record["terminal_probe_transient_count"] == 3
    assert record["failure_stages"] == ["deadline"] and record["error_types"] == ["TimeoutError"]
    assert [line["attempt"] for line in _terminal_probe_lines(records)] == [1, 2, 3]


def test_an_absent_pod_with_a_recoverable_read_every_tick_defers_to_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1 (b)(iv): the absence deferral keys on a transient tick, as in FOR-1665."""
    _guard, provider, _calls, clock, run = setup(tmp_path, monkeypatch, None)
    provider.present = False
    _script_collect(monkeypatch, [_timeout_error()])
    with _warnings() as records, pytest.raises(ProbeControllerFailure) as caught:
        run()
    record = caught.value.record
    assert clock[0] == 3 and record["failure_stages"] == ["deadline"]
    assert record["terminal_probe_transient_count"] == 3
    assert not [r for r in records if r.get("event") == "host.provider_absent.observed"]
