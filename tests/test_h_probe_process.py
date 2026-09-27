"""A ready file is admitted only with the actual isolated watchdog child alive."""

import os
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.test_h_probe_control import root
from tests.test_h_probe_launch import launch

from entropic_substrate.ladder.run.h_probe_control import own_probe_role, read_probe_control
from entropic_substrate.ladder.run.h_probe_launch import H0ProbeLaunch
from entropic_substrate.ladder.run.h_probe_process import (
    WatchdogProcessConfig,
    supervise_probe_watchdog,
)
from entropic_substrate.ladder.run.provider.d1_command_runner import (
    BoundedSubprocessD1CommandRunner,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def importable_test_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))


def fake_policy(directory: Path, spec: H0ProbeLaunch, config: WatchdogProcessConfig) -> None:
    """Test-only body; the production child wrapper still handles process isolation."""
    if config.executable.name == "crash":
        raise RuntimeError("test-only preflight failure")
    with own_probe_role(directory, spec, "watchdog") as writer:
        origin = time.monotonic()
        if config.executable.name != "silent":
            writer.write(
                "ready",
                {
                    "schema": "h-probe-watchdog-ready-v1",
                    "pid": os.getpid() + (config.executable.name == "wrong-pid"),
                    "account_id": "test-account",
                    "monotonic_origin": origin,
                    "controller_deadline": origin + spec.controller_timeout_seconds,
                    "watchdog_deadline": origin + spec.watchdog_timeout_seconds,
                    "observed_at_utc": datetime.now(UTC).isoformat(),
                },
                max_bytes=config.max_control_bytes,
            )
        while time.monotonic() < origin + 5:
            if read_probe_control(directory, spec, "controller", "stop", max_bytes=10_000):
                writer.write(
                    "result",
                    {
                        "schema": "h-probe-watchdog-result-v1",
                        "scientific_authority": "none",
                        "status": "test_only_observation",
                    },
                    max_bytes=config.max_control_bytes,
                )
                return
            time.sleep(0.01)


def config(tmp_path: Path, mode: str = "normal") -> WatchdogProcessConfig:
    return WatchdogProcessConfig(
        executable=tmp_path / mode,
        provider_timeout_seconds=1,
        max_response_bytes=10_000,
        max_launch_bytes=10_000,
        max_control_bytes=10_000,
        poll_seconds=0.01,
        startup_timeout_seconds=10,
        shutdown_timeout_seconds=5,
    )


def test_real_child_ready_is_pid_bound_and_process_group_is_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("entropic_substrate.ladder.run.h_probe_process._run_policy", fake_policy)
    directory, spec = root(tmp_path), launch()
    with supervise_probe_watchdog(
        directory, spec, config(tmp_path), expected_account_id="test-account"
    ) as supervisor:
        assert supervisor.pid != os.getpid()
        assert os.getpgid(supervisor.pid) == supervisor.pid
        supervisor.require_alive()
        supervisor.control.write("stop", {"reason": "fixture_only"}, max_bytes=10_000)
        result = supervisor.wait_result(timeout_seconds=5)
        assert result["status"] == "test_only_observation"
    assert supervisor.alive is False
    assert (directory / "watchdog-process.log").is_file()


@pytest.mark.parametrize("mode", ["crash", "wrong-pid", "silent"])
def test_failed_start_never_yields_a_ready_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr("entropic_substrate.ladder.run.h_probe_process._run_policy", fake_policy)
    directory, spec = root(tmp_path), launch()
    settings = config(tmp_path, mode)
    if mode == "silent":
        settings = replace(settings, startup_timeout_seconds=0.1)
    expected = {
        "crash": "exited before ready",
        "wrong-pid": "ready differs",
        "silent": "ready deadline expired",
    }[mode]
    with pytest.raises((ValueError, TimeoutError), match=expected):
        with supervise_probe_watchdog(
            directory, spec, settings, expected_account_id="test-account"
        ):
            pytest.fail("unverified watchdog yielded")
    if mode == "crash":
        assert "RuntimeError" in (directory / "watchdog-process.log").read_text()
    with own_probe_role(directory, spec, "watchdog"):
        pass  # failed child released its OS lease
    assert (
        read_probe_control(directory, spec, "controller", "create-intent", max_bytes=10_000) is None
    )


def test_account_mismatch_is_refused_before_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("entropic_substrate.ladder.run.h_probe_process._run_policy", fake_policy)
    directory, spec = root(tmp_path), launch()
    with pytest.raises(ValueError, match="ready"):
        with supervise_probe_watchdog(
            directory, spec, config(tmp_path), expected_account_id="another-account"
        ):
            pytest.fail("wrong account admitted")


def test_stale_ready_file_cannot_start_a_new_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("entropic_substrate.ladder.run.h_probe_process._run_policy", fake_policy)
    directory, spec = root(tmp_path), launch()
    with own_probe_role(directory, spec, "watchdog") as writer:
        writer.write("ready", {"pid": os.getpid()}, max_bytes=10_000)
    with pytest.raises(ValueError, match="existing"):
        with supervise_probe_watchdog(
            directory, spec, config(tmp_path), expected_account_id="test-account"
        ):
            pytest.fail("stale guard adopted")


@pytest.mark.parametrize("dangling", [False, True])
def test_existing_process_log_refuses_spawn_before_child_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dangling: bool
) -> None:
    directory, spec = root(tmp_path), launch()
    log = directory / "watchdog-process.log"
    if dangling:
        log.symlink_to(directory / "absent-target")
    else:
        log.write_text("retained failure evidence")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a replacement child was started")

    monkeypatch.setattr(
        "entropic_substrate.ladder.run.h_probe_process.multiprocessing.get_context", forbidden
    )
    with pytest.raises(ValueError, match="existing"):
        with supervise_probe_watchdog(
            directory, spec, config(tmp_path), expected_account_id="test-account"
        ):
            pytest.fail("existing log admitted")
    if dangling:
        assert log.is_symlink() and not log.exists()
    else:
        assert log.read_text() == "retained failure evidence"


@pytest.mark.parametrize(
    "field,value",
    [
        ("poll_seconds", 0),
        ("startup_timeout_seconds", float("inf")),
        ("shutdown_timeout_seconds", True),
        ("max_control_bytes", 0),
    ],
)
def test_invalid_process_limits_are_refused(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(config(tmp_path), **{field: value})


def test_create_intent_is_once_only_and_bound_to_the_live_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("entropic_substrate.ladder.run.h_probe_process._run_policy", fake_policy)
    directory, spec = root(tmp_path), launch()
    with supervise_probe_watchdog(
        directory, spec, config(tmp_path), expected_account_id="test-account"
    ) as supervisor:
        supervisor.record_create_intent()
        intent = read_probe_control(
            directory, spec, "controller", "create-intent", max_bytes=10_000
        )
        assert intent["guard_pid"] == supervisor.pid
        with pytest.raises(ValueError):
            supervisor.record_create_intent()
        supervisor.control.write("stop", {"reason": "fixture_only"}, max_bytes=10_000)
        supervisor.wait_result(timeout_seconds=5)


def test_active_guard_is_retained_after_intent_when_controller_scope_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("entropic_substrate.ladder.run.h_probe_process._run_policy", fake_policy)
    directory, spec = root(tmp_path), launch()
    with pytest.raises(RuntimeError, match="retained alive"):
        with supervise_probe_watchdog(
            directory, spec, config(tmp_path), expected_account_id="test-account"
        ) as supervisor:
            supervisor.record_create_intent()
    assert supervisor.alive
    with own_probe_role(directory, spec, "controller") as control:
        control.write("stop", {"reason": "fixture_only"}, max_bytes=10_000)
    supervisor.wait_result(timeout_seconds=5)
    assert not supervisor.alive


@pytest.mark.parametrize("condition", ["stop", "changed_ready", "deadline"])
def test_late_or_changed_admission_cannot_record_create_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str
) -> None:
    monkeypatch.setattr("entropic_substrate.ladder.run.h_probe_process._run_policy", fake_policy)
    directory = root(tmp_path)
    spec = replace(launch(), controller_timeout_seconds=1, watchdog_timeout_seconds=4)
    with supervise_probe_watchdog(
        directory, spec, config(tmp_path), expected_account_id="test-account"
    ) as supervisor:
        if condition == "stop":
            supervisor.control.write("stop", {"reason": "fixture_only"}, max_bytes=10_000)
        elif condition == "changed_ready":
            supervisor.ready["account_id"] = "tampered"
        else:
            time.sleep(1.05)
        with pytest.raises((ValueError, TimeoutError)):
            supervisor.record_create_intent()
        assert (
            read_probe_control(directory, spec, "controller", "create-intent", max_bytes=10_000)
            is None
        )
        if condition != "stop":
            supervisor.control.write("stop", {"reason": "fixture_only"}, max_bytes=10_000)
        supervisor.wait_result(timeout_seconds=5)


def test_supervisor_keeps_absolute_control_identity_across_parent_chdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("entropic_substrate.ladder.run.h_probe_process._run_policy", fake_policy)
    root(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(tmp_path)
    with supervise_probe_watchdog(
        Path("probe"), launch(), config(tmp_path), expected_account_id="test-account"
    ) as supervisor:
        monkeypatch.chdir(elsewhere)
        supervisor.control.write("stop", {"reason": "fixture_only"}, max_bytes=10_000)
        assert supervisor.wait_result(timeout_seconds=5)["status"] == "test_only_observation"


def test_process_isolation_failure_is_retained_in_private_log(tmp_path: Path) -> None:
    directory = root(tmp_path)
    code = """import sys
from pathlib import Path
from entropic_substrate.ladder.run import h_probe_process as process
def fail():
    raise OSError('injected isolation failure')
process.os.setsid = fail
config = process.WatchdogProcessConfig(Path(sys.executable), 1, 10000, 10000, 10000, .01, 1, 1)
process._process_main(Path(sys.argv[1]), sys.stdin.buffer.read(), config, process._run_policy)
"""
    runner = BoundedSubprocessD1CommandRunner(timeout_seconds=30, max_output_bytes=10_000)
    result = runner.run(
        (sys.executable, "-c", code, str(directory)),
        cwd=Path.cwd(),
        stdin=launch().payload.decode(),
    )
    assert result.returncode == 2
    assert '"error_type": "OSError"' in (directory / "watchdog-process.log").read_text()
