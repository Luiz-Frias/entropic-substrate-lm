"""The sanctioned operator stop for one already-published H probe request.

Writes the R2 stop marker idempotently, waits for the worker's `stopped`
terminal within a required bound, and prints the published snapshot keys.
This is the only sanctioned way to end an in-flight H probe run (FOR-1617);
it does not resume a stopped run, touch the watchdog, or change the host
`h-probe-stop-v1` control message.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

from entropic_substrate.kernel.artifacts import read_d1_artifact_bytes
from entropic_substrate.kernel.runtime.logging import configure_logging
from entropic_substrate.ladder.run.h_probe_events import read_probe_event
from entropic_substrate.ladder.run.h_probe_launch import parse_probe_launch
from entropic_substrate.ladder.run.object_worker import (
    classify_stop_marker,
    encode_stop_marker,
    stop_marker_key,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker.h_probe_host_log import emit
from entropic_substrate.ladder.run.worker.h_probe_log_events import (
    HOST_STOP_MARKER_CONFIRMED,
    HOST_STOP_MARKER_REQUESTED,
    HOST_STOP_MARKER_UNCONFIRMED,
)
from entropic_substrate.ladder.run.worker_command import build_worker_objects
from entropic_substrate.ladder.run.worker_request import AnyProbeRequest, PairProbeRequest
from entropic_substrate.ladder.run.worker_transport import publish_verified_object


def request_stop(
    objects: D1TrainerMailboxObjectPort,
    request: AnyProbeRequest,
    *,
    target: tuple[str, int] | None = None,
    cell: str | None = None,
    wait_seconds: float,
    poll_seconds: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[str, ...] | None:
    """Publish the marker, then wait for `stopped`; ``None`` at the bound.

    Both time arguments are required with no default (house rule 3): a
    caller that forgets one gets a ``TypeError``, never a silently generous
    wait. Publishing is idempotent: a second call publishes byte-identical
    marker content, which the object port itself treats as success -- this
    function never catches, retries or overrides an immutable collision.

    FOR-1693 (DH-44, D10): ``cell`` names the pair cell a stop lands on, and every refusal
    happens BEFORE ``publish_verified_object`` (the marker is write-once, so one that cannot
    then be confirmed would leave the run with no further stop): a cell given for a
    single-profile request, and a pair given no cell.

    FOR-1694 (h1-l4c): a pair is admitted when ``cell`` names one of its cells AND ``target``
    names the seal ``(phase, step)`` on THAT cell's schedule; the marker published is the
    write-once ``h-probe-stop-marker-v2`` ``{phase, step, cell}``, which stops exactly that cell
    at exactly that seal (the other cell runs on), and the wait reads the pair's ``stopped``
    event. A pair without ``--at-cell`` stays refused, by name.
    """
    if isinstance(request, PairProbeRequest):
        if cell is None:
            raise ValueError(
                "a pair stop names its cell: pass --at-cell donor or --at-cell recipient "
                "with --at-phase and --at-step"
            )
        if target is None:
            raise ValueError(
                "a pair stop names its seal: --at-cell needs --at-phase and --at-step "
                "(a marker without a target would stop no cell)"
            )
        marker_payload = encode_stop_marker(request, (target[0], target[1], cell))
    elif cell is not None:
        raise ValueError("a cell is given for a single-profile request; only a pair names a cell")
    else:
        marker_payload = encode_stop_marker(request, target)
    # A defensive self-check only; the object port is the actual admission
    # predicate for what counts as this request's own marker.
    if not classify_stop_marker(marker_payload, request):
        raise ValueError("stop marker does not classify as its own request")
    publish_verified_object(objects, stop_marker_key(request), marker_payload)
    deadline = monotonic() + wait_seconds
    while True:
        event = read_probe_event(
            objects,
            request,
            "stopped",
            expected_pod_id=None,
            max_bytes=request.limits.max_object_bytes,
        )
        if event is not None:
            if event.snapshot is None:
                raise ValueError("stopped event carries no snapshot identity")
            marker = objects.read(event.snapshot.key)
            if type(marker) is not D1TrainerMailboxObjectRead or marker.identity != event.snapshot:
                raise ValueError("stopped snapshot marker identity mismatch")
            manifest = json.loads(marker.payload)
            files = manifest["files"]
            # h-worker-snapshot-v2 lists a capture reference beside its run twin under the same
            # key, so the distinct keys are the set; one entry per manifest row would count twice.
            keys = tuple(sorted({str(item["key"]) for item in files}))
            return (*keys, event.snapshot.key)
        if monotonic() >= deadline:
            return None
        sleep(poll_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, kind in (
        ("launch-root", Path),
        ("launch-file", str),
        ("repo-root", Path),
        ("max-launch-bytes", int),
        ("wait-seconds", float),
        ("poll-seconds", float),
    ):
        parser.add_argument(f"--{name}", type=kind, required=True)
    # Optional pair: stop at this seal (phase, step) rather than the next one.
    parser.add_argument(
        "--at-phase",
        type=str,
        default=None,
        help="seal phase the stop lands on: codec_warmup or prediction (with --at-step)",
    )
    parser.add_argument(
        "--at-step",
        type=int,
        default=None,
        help=(
            "PHASE-LOCAL optimizer step of that seal, the runner's own (phase, step) "
            "coordinates, same as request.pause: on the 1K schedule global 500 is "
            "--at-phase prediction --at-step 250 (with --at-phase)"
        ),
    )
    # FOR-1693: the pair cell a stop lands on (h-probe-stop-marker-v2). Only a pair request
    # has cells; FOR-1694 launches one, and a pair is stopped only when this names its cell
    # (with --at-phase and --at-step). Refused for a single-profile request.
    parser.add_argument(
        "--at-cell",
        type=str,
        default=None,
        help="pair cell the stop lands on (required for a pair, refused for a single profile)",
    )
    return parser


def run_probe_stop(
    argv: Sequence[str],
    *,
    environ: Mapping[str, str],
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Parse the launch record, publish the marker and wait for its receipt.

    ``build_worker_objects`` is imported, never edited: it builds a live R2
    client from ``environ`` credentials, so callers under test must
    monkeypatch it rather than let it run against production R2.
    """
    try:
        try:
            args = _parser().parse_args(list(argv))
        except SystemExit as error:
            return 0 if error.code == 0 else 2
        if (args.at_phase is None) != (args.at_step is None):
            stderr.write("--at-phase and --at-step must be given together\n")
            return 2
        target = None if args.at_phase is None else (args.at_phase, args.at_step)
        if args.wait_seconds <= 0 or args.poll_seconds <= 0:
            raise ValueError("wait and poll bounds must be positive")
        _, payload, _ = read_d1_artifact_bytes(
            args.launch_root,
            args.launch_file,
            label="H0 launch",
            max_bytes=args.max_launch_bytes,
        )
        launch = parse_probe_launch(payload, max_bytes=args.max_launch_bytes)
        request = launch.request
        objects = build_worker_objects(
            args.repo_root,
            environ,
            max_object_bytes=request.limits.max_object_bytes,
            max_objects=request.limits.max_objects,
        )
        pod_name = launch.pod_name
        at = {} if target is None else {"at_phase": target[0], "at_step": target[1]}
        if args.at_cell is not None:
            at["at_cell"] = args.at_cell
        emit(
            "info",
            HOST_STOP_MARKER_REQUESTED,
            pod_name=pod_name,
            wait_seconds=args.wait_seconds,
            **at,
        )
        keys = request_stop(
            objects,
            request,
            target=target,
            cell=args.at_cell,
            wait_seconds=args.wait_seconds,
            poll_seconds=args.poll_seconds,
            monotonic=time.monotonic,
            sleep=time.sleep,
        )
        if keys is None:
            emit(
                "warning",
                HOST_STOP_MARKER_UNCONFIRMED,
                pod_name=pod_name,
                wait_seconds=args.wait_seconds,
            )
            stderr.write(
                json.dumps(
                    {"status": "stop_not_confirmed", "scientific_authority": "none"},
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            return 1
        emit("info", HOST_STOP_MARKER_CONFIRMED, pod_name=pod_name, keys=len(keys))
        stdout.write(
            json.dumps(
                {"status": "stopped", "scientific_authority": "none", "keys": list(keys)},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        return 0
    except BaseException as error:
        stderr.write(
            json.dumps(
                {
                    "status": "stop_command_failed",
                    "scientific_authority": "none",
                    "error_type": type(error).__name__,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: the log substrate first, then the unchanged stop command.

    The stdout success JSON and the stderr failure JSON stay single lines; no host
    event is emitted after either of them.
    """
    configure_logging()
    return run_probe_stop(sys.argv[1:] if argv is None else argv, environ=os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
