"""Spawn and observe an isolated watchdog before any H0 allocation is submitted."""

import json
import math
import multiprocessing
import os
import signal
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import FrameType
from typing import NoReturn

from entropic_substrate.__legacy.d1_artifact_snapshot import normalize_d1_artifact_root
from entropic_substrate.__legacy.d1_command_runner import BoundedSubprocessD1CommandRunner
from entropic_substrate.__legacy.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.ladder.run.h_probe_control import (
    ProbeControlPublisher,
    own_probe_role,
    read_probe_control,
)
from entropic_substrate.ladder.run.h_probe_launch import H0ProbeLaunch, parse_probe_launch
from entropic_substrate.ladder.run.h_probe_watchdog import ProbeWatchdogFailure, run_probe_watchdog
from entropic_substrate.ladder.run.h_runpod_transport import HRunpodTransport


@dataclass(frozen=True, slots=True)
class WatchdogProcessConfig:
    executable: Path
    provider_timeout_seconds: float
    max_response_bytes: int
    max_launch_bytes: int
    max_control_bytes: int
    poll_seconds: float
    startup_timeout_seconds: float
    shutdown_timeout_seconds: float

    def __post_init__(self) -> None:
        runner = BoundedSubprocessD1CommandRunner(
            timeout_seconds=self.provider_timeout_seconds, max_output_bytes=self.max_response_bytes
        )
        HRunpodTransport(
            executable=self.executable,
            runner=runner,
            max_response_bytes=self.max_response_bytes,
            max_launch_bytes=self.max_launch_bytes,
        )
        if type(self.max_control_bytes) is not int or self.max_control_bytes < 1:
            raise ValueError("watchdog control limit must be positive")
        for value in (
            self.poll_seconds,
            self.startup_timeout_seconds,
            self.shutdown_timeout_seconds,
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("watchdog process time bounds must be finite and positive")


def _run_policy(root: Path, launch: H0ProbeLaunch, config: WatchdogProcessConfig) -> None:
    provider = HRunpodTransport(
        executable=config.executable,
        runner=BoundedSubprocessD1CommandRunner(
            timeout_seconds=config.provider_timeout_seconds,
            max_output_bytes=config.max_response_bytes,
        ),
        max_response_bytes=config.max_response_bytes,
        max_launch_bytes=config.max_launch_bytes,
    )
    run_probe_watchdog(
        root,
        launch,
        provider,
        poll_seconds=config.poll_seconds,
        max_control_bytes=config.max_control_bytes,
    )


def _interrupt(signum: int, _frame: FrameType | None) -> NoReturn:
    # Let the bounded command runner sweep/reap its subprocess before this exits.
    raise SystemExit(128 + signum)


def _process_main(
    root: Path,
    payload: bytes,
    config: WatchdogProcessConfig,
    body: Callable[[Path, H0ProbeLaunch, WatchdogProcessConfig], None],
) -> None:
    descriptor = os.open(
        root / "watchdog-process.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    os.close(descriptor)
    try:
        os.setsid()
        signal.signal(signal.SIGTERM, _interrupt)
        body(root, parse_probe_launch(payload, max_bytes=config.max_launch_bytes), config)
    except BaseException as error:
        sys.stderr.write(
            json.dumps({"status": "watchdog_process_failed", "error_type": type(error).__name__})
            + "\n"
        )
        sys.stderr.flush()
        raise SystemExit(2) from None


class ProbeWatchdogProcess:
    """Own a real child handle; ready/result records alone never establish liveness."""

    def __init__(
        self,
        root: Path,
        launch: H0ProbeLaunch,
        config: WatchdogProcessConfig,
        control: ProbeControlPublisher,
        process: BaseProcess,
        started: float,
    ) -> None:
        self.root, self.launch, self.config, self.control = root, launch, config, control
        self._process, self._started, self._closed = process, started, False
        if process.pid is None:
            raise ValueError("watchdog process was not started")
        self._pid = process.pid
        self.ready: dict[str, object] | None = None
        self.exitcode: int | None = None

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def alive(self) -> bool:
        return not self._closed and self._process.is_alive()

    def require_alive(self) -> None:
        if not self.alive:
            raise ValueError("watchdog process is not alive")
        if os.getpgid(self.pid) != self.pid:
            raise ValueError("watchdog is not isolated from the controller process group")

    def await_ready(self, account_id: str) -> None:
        deadline = self._started + self.config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if not self.alive:
                raise ValueError("watchdog exited before ready")
            value = read_probe_control(
                self.root, self.launch, "watchdog", "ready", max_bytes=self.config.max_control_bytes
            )
            if value is not None:
                row = recovery_record(
                    value,
                    "schema pid account_id monotonic_origin controller_deadline watchdog_deadline observed_at_utc",
                )
                origin = row["monotonic_origin"]
                if type(origin) is not int and type(origin) is not float:
                    raise ValueError("watchdog ready has an invalid clock")
                if not math.isfinite(origin):
                    raise ValueError("watchdog ready has an invalid clock")
                at = row["observed_at_utc"]
                if (
                    row["schema"] != "h-probe-watchdog-ready-v1"
                    or type(row["pid"]) is not int
                    or row["pid"] != self.pid
                    or row["account_id"] != account_id
                    or not self._started <= origin <= time.monotonic()
                    or row["controller_deadline"] != origin + self.launch.controller_timeout_seconds
                    or row["watchdog_deadline"] != origin + self.launch.watchdog_timeout_seconds
                    or type(at) is not str
                    or datetime.fromisoformat(at).utcoffset() != UTC.utcoffset(None)
                ):
                    raise ValueError(
                        "watchdog ready differs from its process/account/clock binding"
                    )
                self.require_alive()
                self.ready = row
                return
            time.sleep(min(self.config.poll_seconds, max(0, deadline - time.monotonic())))
        raise TimeoutError("watchdog ready deadline expired before allocation")

    def wait_result(self, *, timeout_seconds: float) -> dict[str, object]:
        """Return a bound child observation, not a certificate of workload completion."""
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("watchdog join bound must be finite and positive")
        if self._closed:
            raise ValueError("watchdog process was already joined")
        self._process.join(timeout_seconds)
        if self.alive:
            raise TimeoutError("watchdog remains alive; its cleanup responsibility is retained")
        self.exitcode = self._process.exitcode
        try:
            record = read_probe_control(
                self.root,
                self.launch,
                "watchdog",
                "result",
                max_bytes=self.config.max_control_bytes,
            )
            if (
                record is None
                or record.get("schema") != "h-probe-watchdog-result-v1"
                or record.get("scientific_authority") != "none"
            ):
                raise ValueError("watchdog exited without a bound result")
            if self.exitcode != 0:
                raise ProbeWatchdogFailure(record)
            return record
        finally:
            self._process.close()
            self._closed = True

    def record_create_intent(self) -> ArtifactIdentity:
        """Claim one submission only after live-child, ready, stop and deadline checks."""
        self.require_alive()
        ready = read_probe_control(
            self.root, self.launch, "watchdog", "ready", max_bytes=self.config.max_control_bytes
        )
        if ready is None or ready != self.ready:
            raise ValueError("watchdog ready binding changed before create")
        deadline = ready["controller_deadline"]
        if type(deadline) is not int and type(deadline) is not float:
            raise ValueError("watchdog ready deadline has an invalid type")
        if time.monotonic() >= deadline:
            raise TimeoutError("controller deadline expired before create")
        if (
            read_probe_control(
                self.root,
                self.launch,
                "controller",
                "stop",
                max_bytes=self.config.max_control_bytes,
            )
            is not None
        ):
            raise ValueError("stop was already requested before create")
        return self.control.write(
            "create-intent",
            {"schema": "h-probe-create-intent-v1", "guard_pid": self.pid},
            max_bytes=self.config.max_control_bytes,
        )

    def release(self) -> None:
        """Reap pre-allocation failures, but never kill a guardian after intent."""
        if self._closed:
            return
        intent = read_probe_control(
            self.root,
            self.launch,
            "controller",
            "create-intent",
            max_bytes=self.config.max_control_bytes,
        )
        if self._process.is_alive() and intent is None:
            self._process.terminate()
            self._process.join(self.config.shutdown_timeout_seconds)
        if self._process.is_alive():
            raise RuntimeError(f"watchdog {self.pid} retained alive; cleanup is not finished")
        self.exitcode = self._process.exitcode
        self._process.close()
        self._closed = True


@contextmanager
def supervise_probe_watchdog(
    root: Path, launch: H0ProbeLaunch, config: WatchdogProcessConfig, *, expected_account_id: str
) -> Generator[ProbeWatchdogProcess]:
    """Hold the controller lease through the caller's workflow.

    Before intent, a failed startup can be terminated and reaped safely. After
    intent, this context never kills the guardian: callers must request shutdown
    and join it, or retain its independent deadline responsibility on failure.
    """
    if type(config) is not WatchdogProcessConfig:
        raise TypeError("watchdog requires a nominal process configuration")
    config.__post_init__()
    if type(expected_account_id) is not str or not expected_account_id:
        raise ValueError("watchdog requires the independently observed account identity")
    root = normalize_d1_artifact_root(root)
    with own_probe_role(root, launch, "controller") as control:
        log = root / "watchdog-process.log"
        if (
            log.exists()
            or log.is_symlink()
            or any(
                read_probe_control(root, launch, role, name, max_bytes=config.max_control_bytes)
                is not None
                for role, name in (("watchdog", "ready"), ("controller", "create-intent"))
            )
        ):
            raise ValueError("existing attempt requires inspection, not a fresh watchdog")
        started = time.monotonic()
        process = multiprocessing.get_context("spawn").Process(
            target=_process_main, args=(root, launch.payload, config, _run_policy), daemon=False
        )
        process.start()
        supervisor = ProbeWatchdogProcess(root, launch, config, control, process, started)
        try:
            supervisor.await_ready(expected_account_id)
            yield supervisor
        finally:
            supervisor.release()
