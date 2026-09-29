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
from entropic_substrate.ladder.run.worker_command import build_worker_objects
from entropic_substrate.ladder.run.worker_request import ProbeRequest
from entropic_substrate.ladder.run.worker_transport import publish_verified_object


def request_stop(
    objects: D1TrainerMailboxObjectPort,
    request: ProbeRequest,
    *,
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
    """
    marker_payload = encode_stop_marker(request)
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
            keys = tuple(sorted(str(item["key"]) for item in files))
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
        keys = request_stop(
            objects,
            request,
            wait_seconds=args.wait_seconds,
            poll_seconds=args.poll_seconds,
            monotonic=time.monotonic,
            sleep=time.sleep,
        )
        if keys is None:
            stderr.write(
                json.dumps(
                    {"status": "stop_not_confirmed", "scientific_authority": "none"},
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            return 1
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


if __name__ == "__main__":
    raise SystemExit(run_probe_stop(sys.argv[1:], environ=os.environ))
