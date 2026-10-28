"""Run one already-published, operator-approved H request from a recorded launch."""

import argparse
import json
import math
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from entropic_substrate.kernel.artifacts import (
    normalize_d1_artifact_root,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.runtime.logging import configure_logging
from entropic_substrate.ladder.run.h_probe_launch import parse_probe_launch
from entropic_substrate.ladder.run.h_probe_process import (
    WatchdogProcessConfig,
    supervise_probe_watchdog,
)
from entropic_substrate.ladder.run.h_runpod_transport import HRunpodTransport
from entropic_substrate.ladder.run.provider.d1_command_runner import (
    BoundedSubprocessD1CommandRunner,
)
from entropic_substrate.ladder.run.worker.h_probe_controller import (
    ProbeControllerFailure,
    run_probe_controller,
)
from entropic_substrate.ladder.run.worker_command import build_worker_objects


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, kind in (
        ("launch-root", Path),
        ("launch-file", str),
        ("workspace", Path),
        ("repo-root", Path),
        ("expected-account-id", str),
        ("runpodctl", Path),
        ("provider-timeout", float),
        ("max-provider-bytes", int),
        ("max-launch-bytes", int),
        ("max-control-bytes", int),
        ("poll-seconds", float),
        ("startup-timeout", float),
        ("shutdown-timeout", float),
        ("max-hourly-rate", float),
    ):
        parser.add_argument(f"--{name}", type=kind, required=True)
    return parser


def run_probe_command(
    argv: Sequence[str],
    *,
    environ: Mapping[str, str],
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Spending approval belongs to the invoker; a balance or CLI flag grants none.

    All limits and account identity are explicit. This command does not publish
    inputs, create templates, retry allocation, or admit scientific successors.
    Controller evidence survives an additional context-release error in the output.
    """
    evidence: dict[str, object] | None = None
    try:
        try:
            args = _parser().parse_args(list(argv))
        except SystemExit as error:
            return 0 if error.code == 0 else 2
        settings = WatchdogProcessConfig(
            executable=args.runpodctl,
            provider_timeout_seconds=args.provider_timeout,
            max_response_bytes=args.max_provider_bytes,
            max_launch_bytes=args.max_launch_bytes,
            max_control_bytes=args.max_control_bytes,
            poll_seconds=args.poll_seconds,
            startup_timeout_seconds=args.startup_timeout,
            shutdown_timeout_seconds=args.shutdown_timeout,
        )
        if not math.isfinite(args.max_hourly_rate) or args.max_hourly_rate <= 0:
            raise ValueError("hourly price ceiling must be finite and positive")
        if (
            not args.expected_account_id
            or args.expected_account_id != args.expected_account_id.strip()
        ):
            raise ValueError("expected account identity must be explicit and canonical")
        _, payload, _ = read_d1_artifact_bytes(
            args.launch_root,
            args.launch_file,
            label="H0 launch",
            max_bytes=args.max_launch_bytes,
        )
        launch = parse_probe_launch(payload, max_bytes=args.max_launch_bytes)
        # Rung B: a scientific launch's ceiling must keep the run inside its sealed cap.
        launch.require_hourly_rate_within_cap(args.max_hourly_rate)
        root = normalize_d1_artifact_root(args.workspace)
        metadata = root.stat()
        if (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or any(root.iterdir())
        ):
            raise ValueError("host workspace must be empty, private and owned by the caller")
        provider = HRunpodTransport(
            executable=settings.executable,
            runner=BoundedSubprocessD1CommandRunner(
                timeout_seconds=settings.provider_timeout_seconds,
                max_output_bytes=settings.max_response_bytes,
            ),
            max_response_bytes=settings.max_response_bytes,
            max_launch_bytes=settings.max_launch_bytes,
        )
        objects = build_worker_objects(
            args.repo_root,
            environ,
            max_object_bytes=launch.request.limits.max_object_bytes,
            max_objects=launch.request.limits.max_objects,
        )
        control, capture = root / "control", root / "capture"
        control.mkdir(mode=0o700)
        capture.mkdir(mode=0o700)
        with supervise_probe_watchdog(
            control, launch, settings, expected_account_id=args.expected_account_id
        ) as guard:
            try:
                evidence = run_probe_controller(
                    guard,
                    provider,
                    objects,
                    capture,
                    poll_seconds=settings.poll_seconds,
                    max_hourly_rate=args.max_hourly_rate,
                )
            except ProbeControllerFailure as error:
                evidence = error.record
                raise
        stdout.write(json.dumps(evidence, sort_keys=True, allow_nan=False) + "\n")
        return 0
    except BaseException as error:
        stderr.write(
            json.dumps(
                {
                    "status": "controller_command_failed",
                    "scientific_authority": "none",
                    "error_type": type(error).__name__,
                    "evidence": evidence,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: the log substrate first, then the unchanged command.

    The controller runs inside this process, so its host events (the closed
    vocabulary in ``worker.h_probe_log_events``) land on this stderr and in the
    structured JSON sink. The stdout success JSON and the stderr failure JSON are
    unchanged single lines, and no host event is emitted after either of them.
    """
    configure_logging()
    return run_probe_command(sys.argv[1:] if argv is None else argv, environ=os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
