"""Publish, execute and receive H probes through bounded project R2 custody."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TextIO, cast

from entropic_substrate.d1_artifact_snapshot import read_d1_artifact_bytes
from entropic_substrate.d1_dvc_remote_config import load_d1_dvc_s3_remote
from entropic_substrate.d1_operator_values import require_d1_pod_id
from entropic_substrate.d1_trainer_mailbox_contracts import D1TrainerMailboxObjectIdentity
from entropic_substrate.d1_trainer_object_transport import D1TrainerMailboxObjectPort
from entropic_substrate.d1_trainer_s3_object_adapter import (
    Boto3D1TrainerMailboxObjectAdapter,
    D1S3ClientPort,
)
from entropic_substrate.sentence_prediction.object_worker import run_object_worker
from entropic_substrate.sentence_prediction.worker_completion import receive_worker_completion
from entropic_substrate.sentence_prediction.worker_request import ProbeRequest, parse_probe_request
from entropic_substrate.sentence_prediction.worker_transport import (
    publish_probe_request,
    read_published_probe_request,
)


def build_worker_objects(
    repo_root: Path, environ: Mapping[str, str], *, max_object_bytes: int, max_objects: int
) -> Boto3D1TrainerMailboxObjectAdapter:
    """Use committed topology and explicit environment credentials; never ambient fallback."""
    import boto3
    from botocore.config import Config

    if any(type(value) is not int or value < 1 for value in (max_object_bytes, max_objects)):
        raise ValueError("worker transport limits must be positive integers")
    access, private = environ.get("AWS_ACCESS_KEY_ID"), environ.get("AWS_SECRET_ACCESS_KEY")
    if type(access) is not str or not access or type(private) is not str or not private:
        raise ValueError("worker object credentials are missing")
    topology = load_d1_dvc_s3_remote(repo_root)
    client = boto3.client(
        "s3",
        endpoint_url=topology.endpoint_url,
        region_name="auto",
        aws_access_key_id=access,
        aws_secret_access_key=private,
        config=Config(connect_timeout=15, read_timeout=15, retries={"max_attempts": 0}),
    )
    return Boto3D1TrainerMailboxObjectAdapter(
        client=cast(D1S3ClientPort, client),
        bucket=topology.bucket,
        max_objects=max_objects,
        max_object_bytes=max_object_bytes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("publish", "run", "receive"):
        command = commands.add_parser(name)
        command.add_argument("--repo-root", type=Path, required=True)
        command.add_argument("--max-object-bytes", type=int, required=True)
        command.add_argument("--max-objects", type=int, required=True)
        if name in ("publish", "receive"):
            command.add_argument("--artifact-root", type=Path, required=True)
            command.add_argument("--request-file", required=True)
        if name == "run":
            command.add_argument("--workspace-root", type=Path, required=True)
            prefix = "request"
        elif name == "receive":
            command.add_argument("--destination", type=Path, required=True)
            command.add_argument("--pod-id")
            prefix = "terminal"
        else:
            continue
        for field, kind in (("key", str), ("sha256", str), ("size", int)):
            command.add_argument(f"--{prefix}-{field}", type=kind, required=True)
        command.add_argument("--local-tiny", action="store_true")
    return parser


def _scope(request: ProbeRequest, local: bool, object_limit: int, count_limit: int) -> None:
    expected = "local_tiny" if local else "remote_tiny"
    if request.profile.run_class != expected:
        raise ValueError("worker command scope differs from the request")
    if request.limits.max_object_bytes > object_limit or request.limits.max_objects > count_limit:
        raise ValueError("worker request exceeds the configured transport envelope")


def run_worker_command(
    argv: Sequence[str],
    *,
    environ: Mapping[str, str],
    objects: D1TrainerMailboxObjectPort | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run one operation; report structured errors without credential/provider details."""
    try:
        args = _parser().parse_args(list(argv))
        if args.max_object_bytes < 1 or args.max_objects < 1:
            raise ValueError("worker command limits must be positive")
        selected = (
            objects
            if objects is not None
            else build_worker_objects(
                args.repo_root,
                environ,
                max_object_bytes=args.max_object_bytes,
                max_objects=args.max_objects,
            )
        )
        if args.command == "run":
            marker = D1TrainerMailboxObjectIdentity(
                args.request_key, args.request_size, args.request_sha256
            )
            request = read_published_probe_request(
                selected, marker, max_request_bytes=args.max_object_bytes
            )
            _scope(request, args.local_tiny, args.max_object_bytes, args.max_objects)
            if not args.local_tiny:
                require_d1_pod_id(environ.get("RUNPOD_POD_ID"))
            result = run_object_worker(
                selected,
                marker,
                args.workspace_root,
                max_request_bytes=args.max_object_bytes,
                repo_root=args.repo_root,
            )
            response = {
                "status": "worker_returned",
                "scientific_authority": "none",
                "object": asdict(result),
            }
        else:
            _, payload, _ = read_d1_artifact_bytes(
                args.artifact_root,
                args.request_file,
                label="worker command request",
                max_bytes=args.max_object_bytes,
            )
            request = parse_probe_request(payload, max_bytes=args.max_object_bytes)
            if args.command == "publish":
                if (
                    request.limits.max_object_bytes > args.max_object_bytes
                    or request.limits.max_objects > args.max_objects
                ):
                    raise ValueError("request exceeds publisher transport envelope")
                result = publish_probe_request(selected, request, args.artifact_root)
                response = {
                    "status": "request_published",
                    "scientific_authority": "none",
                    "object": asdict(result),
                }
            else:
                _scope(request, args.local_tiny, args.max_object_bytes, args.max_objects)
                terminal = D1TrainerMailboxObjectIdentity(
                    args.terminal_key, args.terminal_size, args.terminal_sha256
                )
                response = receive_worker_completion(
                    selected,
                    request,
                    terminal,
                    args.destination,
                    expected_pod_id=args.pod_id,
                    max_marker_bytes=args.max_object_bytes,
                )
        stdout.write(json.dumps(response, sort_keys=True, allow_nan=False) + "\n")
        return 0
    except Exception as error:
        stderr.write(
            json.dumps(
                {
                    "status": "error",
                    "code": "worker_command_refused",
                    "error_type": type(error).__name__,
                    "scientific_authority": "none",
                },
                sort_keys=True,
            )
            + "\n"
        )
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return run_worker_command(sys.argv[1:] if argv is None else argv, environ=os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
