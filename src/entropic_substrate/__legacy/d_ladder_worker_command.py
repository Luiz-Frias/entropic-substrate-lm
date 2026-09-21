"""RunPod foreground worker and host publisher over existing R2 objects.

Launch ``serve`` as the pod command after an approved allocation. Read its
``bootstrap.json`` using ``inspect``, prepare the registered RemoteLadderRequest
with that measured runtime origin, then ``publish`` it. The worker waits once
under a bounded deadline and exits after exporting execution evidence.
Allocation, provider termination deadlines and final pod deletion belong to
the existing operator lifecycle, not this trainer process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO, cast

from entropic_substrate.d1_artifact_snapshot import read_d1_artifact_bytes
from entropic_substrate.d1_operator_values import require_d1_pod_id
from entropic_substrate.d1_r2_runtime_composition import build_d1_r2_runtime_composition
from entropic_substrate.d1_system_runtime_probe import SystemD1RuntimeProbe
from entropic_substrate.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.d1_trainer_mailbox_worker import D1TrainerMailboxWaitPolicy
from entropic_substrate.d1_trainer_object_transport import D1TrainerMailboxObjectPort
from entropic_substrate.d1_trainer_process_contracts import (
    parse_d1_runtime_report,
    serialize_d1_runtime_report,
)
from entropic_substrate.d_ladder_remote import publish_remote_request, run_remote_worker
from entropic_substrate.d_ladder_remote_request import RemoteLadderRequest, parse_remote_request
from entropic_substrate.d_ladder_worker_runtime import (
    build_worker_factory,
    measure_hardware,
    verify_worker_source,
    worker_resource_registration,
)
from entropic_substrate.evidence.ladder_campaign import object_fields
from entropic_substrate.evidence.resource_provenance import RunStartProvenance
from entropic_substrate.experiments.d1_primitives import (
    canonical_json,
    require_commit,
    require_sha256,
)


def wait_object(
    objects: D1TrainerMailboxObjectPort, key: str, policy: D1TrainerMailboxWaitPolicy
) -> D1TrainerMailboxObjectRead:
    """Bound waiting for a new command; never republish or synthesize one."""
    start = time.monotonic()
    if not math.isfinite(start):
        raise ValueError("worker monotonic clock is invalid")
    while True:
        value = objects.read(key)
        if value is not None:
            if type(value) is not D1TrainerMailboxObjectRead or value.identity.key != key:
                raise ValueError("worker object identity mismatch")
            return value
        current = time.monotonic()
        if not math.isfinite(current) or current < start:
            raise ValueError("worker monotonic clock is invalid")
        remaining = policy.maximum_seconds - (current - start)
        if remaining <= 0:
            raise TimeoutError("worker request wait budget exhausted")
        time.sleep(min(remaining, policy.poll_interval_seconds))


def _observed_start(request: RemoteLadderRequest, payload: bytes) -> RunStartProvenance:
    row = object_fields(
        json.loads(payload),
        "schema pod_id attempt_sha256 code_commit image_digest runtime accelerator_model accelerator_count resource_activation",
    )
    if (
        row["schema"] != "d-ladder-pod-bootstrap-v1"
        or row["pod_id"] != request.pod_id
        or row["attempt_sha256"] != request.attempt_sha256
        or row["code_commit"] != request.code_commit
        or row["image_digest"] != request.start.image_digest
        or row["resource_activation"] != request.start.resource_activation.manifest
    ):
        raise ValueError("request differs from measured pod bootstrap identity")
    runtime = parse_d1_runtime_report(
        canonical_json(row["runtime"]).encode(),
        pod_id=request.pod_id,
        attempt_sha256=request.attempt_sha256,
    )
    observed = replace(
        request.start,
        trainer_started_at=runtime.trainer_started_at,
        cuda_version=runtime.cuda_version,
        driver_version=runtime.driver_version,
        pytorch_version=runtime.pytorch_version,
        compiler_enabled=runtime.compiler_enabled,
        compiler_version=runtime.compiler_version,
        accelerator_model=cast(str, row["accelerator_model"]),
        accelerator_count=cast(int, row["accelerator_count"]),
    )
    if observed != request.start:
        raise ValueError("request runtime differs from measured pod bootstrap facts")
    return observed


def _serve(
    arguments: argparse.Namespace, environ: Mapping[str, str], objects: D1TrainerMailboxObjectPort
) -> D1TrainerMailboxObjectIdentity:
    pod = require_d1_pod_id(environ.get("RUNPOD_POD_ID"))
    attempt = cast(str, arguments.attempt_sha256)
    require_sha256(attempt, label="worker attempt")
    commit, image = (
        environ.get("D_LADDER_CODE_COMMIT", ""),
        environ.get("D_LADDER_IMAGE_DIGEST", ""),
    )
    require_commit(commit, label="image source commit")
    if len(commit) not in (40, 64) or not image.startswith("sha256:"):
        raise ValueError("worker requires full source commit and immutable image digest")
    require_sha256(image.removeprefix("sha256:"), label="image digest")
    policy = D1TrainerMailboxWaitPolicy(
        arguments.maximum_wait_seconds, arguments.poll_interval_seconds
    )
    runtime = SystemD1RuntimeProbe()(pod)
    model, count = measure_hardware()
    prefix = f"d-ladder/{pod}/{attempt}"
    bootstrap = canonical_json(
        {
            "schema": "d-ladder-pod-bootstrap-v1",
            "pod_id": pod,
            "attempt_sha256": attempt,
            "code_commit": commit,
            "image_digest": image,
            "runtime": json.loads(serialize_d1_runtime_report(runtime, attempt_sha256=attempt)),
            "accelerator_model": model,
            "accelerator_count": count,
            "resource_activation": worker_resource_registration().manifest,
        }
    ).encode()
    expected = D1TrainerMailboxObjectIdentity(
        f"{prefix}/bootstrap.json", len(bootstrap), hashlib.sha256(bootstrap).hexdigest()
    )
    if objects.publish(expected.key, bootstrap, expected_sha256=expected.sha256) != expected:
        raise ValueError("pod bootstrap publication identity mismatch")
    item = wait_object(objects, f"{prefix}/request.json", policy)
    request = parse_remote_request(item.payload)
    observed = _observed_start(request, bootstrap)
    if request.prefix != prefix:
        raise ValueError("request belongs to a different pod attempt")
    verify_worker_source(objects, request, Path(arguments.repo_root))
    refreshed = SystemD1RuntimeProbe()(pod)
    for field in (
        "cuda_version",
        "driver_version",
        "pytorch_version",
        "compiler_enabled",
        "compiler_version",
    ):
        if getattr(refreshed, field) != getattr(runtime, field):
            raise ValueError("runtime changed after bootstrap")
    if measure_hardware() != (model, count):
        raise ValueError("accelerators changed after bootstrap")
    return run_remote_worker(
        objects,
        item.identity,
        Path(arguments.workspace_root),
        observed_start=observed,
        observed_code_commit=commit,
        factory=build_worker_factory(observed.resource_activation),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "publish", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--repo-root", required=True)
        command.add_argument("--max-mailbox-objects", type=int, default=20000)
        if name == "serve":
            command.add_argument("--workspace-root", required=True)
            command.add_argument("--attempt-sha256", required=True)
            command.add_argument("--maximum-wait-seconds", type=float, required=True)
            command.add_argument("--poll-interval-seconds", type=float, default=1.0)
        elif name == "publish":
            command.add_argument("--artifact-root", required=True)
            command.add_argument("--request-manifest", required=True)
        else:
            command.add_argument("--object-key", required=True)
    return parser


def run_worker_command(
    argv: Sequence[str],
    *,
    environ: Mapping[str, str],
    objects: D1TrainerMailboxObjectPort | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Execute the real command composition; emit bounded JSON without credential values."""
    try:
        arguments = _parser().parse_args(list(argv))
        if arguments.command == "serve":
            require_d1_pod_id(environ.get("RUNPOD_POD_ID"))
        selected = (
            objects
            if objects is not None
            else build_d1_r2_runtime_composition(
                repo_root=Path(arguments.repo_root),
                environ=environ,
                max_objects=arguments.max_mailbox_objects,
            ).mailbox
        )
        if arguments.command == "serve":
            result = _serve(arguments, environ, selected)
        elif arguments.command == "publish":
            _, payload, _ = read_d1_artifact_bytes(
                Path(arguments.artifact_root),
                arguments.request_manifest,
                label="remote request manifest",
            )
            request = parse_remote_request(payload)
            bootstrap = selected.read(f"{request.prefix}/bootstrap.json")
            if bootstrap is None:
                raise ValueError("pod bootstrap evidence is not present")
            _observed_start(request, bootstrap.payload)
            result = publish_remote_request(selected, request, Path(arguments.artifact_root))
        else:
            value = selected.read(arguments.object_key)
            if value is None:
                raise ValueError("requested remote object is missing")
            stdout.write(
                canonical_json({"status": "success", "object": json.loads(value.payload)}) + "\n"
            )
            return 0
        stdout.write(
            canonical_json(
                {
                    "status": "success",
                    "key": result.key,
                    "sha256": result.sha256,
                    "size_bytes": result.size_bytes,
                }
            )
            + "\n"
        )
        return 0
    except Exception as error:
        stderr.write(
            canonical_json(
                {"status": "error", "error_type": type(error).__name__, "code": "worker_refused"}
            )
            + "\n"
        )
        return 20


def main(argv: Sequence[str] | None = None) -> int:
    return run_worker_command(sys.argv[1:] if argv is None else argv, environ=os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
