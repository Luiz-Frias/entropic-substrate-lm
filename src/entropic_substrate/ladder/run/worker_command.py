"""Publish, execute and receive H probes through bounded project R2 custody."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal, TextIO, cast

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.kernel.artifacts import read_d1_artifact_bytes
from entropic_substrate.kernel.runtime.logging import configure_logging, logger
from entropic_substrate.ladder.capture.adapters.composite import build_sink_for_run
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.capture.ports.run_descriptor import TrackingRunDescriptor
from entropic_substrate.ladder.capture.reader.sinks import CONFIG_HASH_TAG, EXP_TAG, h_config_hash
from entropic_substrate.ladder.run.admission import RUN_ADMISSION
from entropic_substrate.ladder.run.campaign.d1_operator_values import require_d1_pod_id
from entropic_substrate.ladder.run.h_metric_extractor import HTrainingStepExtractor
from entropic_substrate.ladder.run.object_worker import run_object_worker
from entropic_substrate.ladder.run.pair_run_tags import (
    LIFECYCLE_TAG,
    PAIR_OUTCOME_TAG,
    CellLifecycle,
    PairOutcome,
)
from entropic_substrate.ladder.run.transport.d1_dvc_remote_config import load_d1_dvc_s3_remote
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.transport.d1_trainer_s3_object_adapter import (
    Boto3D1TrainerMailboxObjectAdapter,
    D1S3ClientPort,
    D1TrainerS3ObjectError,
)
from entropic_substrate.ladder.run.worker_completion import receive_worker_completion
from entropic_substrate.ladder.run.worker_log_events import WORKER_END, WORKER_START, render
from entropic_substrate.ladder.run.worker_request import (
    RESULT_SCHEMA_V1,
    RESULT_SCHEMA_V2,
    AnyProbeRequest,
    PairProbeRequest,
    parse_probe_request,
)
from entropic_substrate.ladder.run.worker_transport import (
    publish_probe_request,
    read_published_probe_request,
)


class UnresolvedWorkerCredentials(ValueError):
    """A provider reference reached the process without secret substitution."""


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
    if any(
        value.strip().startswith("{{") and "RUNPOD_SECRET_" in value for value in (access, private)
    ):
        raise UnresolvedWorkerCredentials("worker credentials contain unresolved references")
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


class _PairCellSinks:
    """The per-cell sink provider of a pair run (FOR-1693, D11, DH-40).

    One MLflow run, one metrics file and one worker log PER CELL, under
    ``<observability>/<cell>/``. ``begin`` is the ONE re-point of the substrate's JSON sink for a
    cell: after it, ``worker.start``, ``mlflow.run`` and every later line (including that cell's
    ``checkpoint.seal`` and ``evaluation.start`` events) land in that cell's log, so each log
    carries exactly one ``worker.start`` and only its own cell's seals. The run name stays
    ``h-worker-<cell profile identity>`` (``h_wave_monitor`` searches by it); the tags add
    ``block``, ``pair``, ``cell`` and ``run_authority``; ``tiny`` stays derived from the authority.

    DH-50 (operator ruling 2026-11-03): every cell run is CREATED tagged ``esl.lifecycle = live``
    (every begin, so every resume, which opens a run of its own); ``lifecycle`` replaces it on
    that cell's run and ``pair_outcome`` sets ``esl.pair_outcome`` on the donor's run, both
    through the cell sink's ``annotate`` (the donor's run may already be terminated: the hand-off
    is decided after the donor's close).
    """

    def __init__(self, request: PairProbeRequest, observability_root: Path) -> None:
        self._request = request
        self._root = observability_root
        self._opened: dict[str, MetricSink] = {}
        self._began: dict[str, float] = {}
        self._finished: set[str] = set()

    def begin(self, cell: str, profile: HProfile) -> MetricSink:
        request = self._request
        directory = self._root / cell
        log = directory / "worker-log.jsonl"
        configure_logging(json_path=log)
        started = {
            "request_sha256": request.digest,
            "image": request.image,
            "code_commit": request.code_commit,
        }
        logger.bind(event=WORKER_START, **started).info(render(WORKER_START, **started))
        chash = h_config_hash(profile)
        params: dict[str, str | int] = {
            EXP_TAG: profile.name,
            "seed": profile.seed,
            CONFIG_HASH_TAG: chash,
            "run_class": profile.run_class,
            "precision": profile.precision,
        }
        if request.code_commit is not None:
            params["git_sha"] = request.code_commit
        if request.image is not None:
            params["image"] = request.image
        tags: dict[str, str | int] = {
            "schema": RESULT_SCHEMA_V2,
            EXP_TAG: profile.name,
            CONFIG_HASH_TAG: chash,
            "request_sha256": request.digest,
            "block": request.binding.block_id,
            "pair": request.binding.pair_id,
            "cell": cell,
            LIFECYCLE_TAG: "live",
        }
        if request.authority is not None:
            params["run_authority"] = request.authority.digest
            tags["run_authority"] = request.authority.digest
        sink = build_sink_for_run(
            TrackingRunDescriptor(
                run_name=f"h-worker-{profile.identity}",
                tiny=request.authority is None,
                params=params,
                tags=tags,
            ),
            results_path=directory / "metrics.jsonl",
            log_path=log,
            metric_extractor=HTrainingStepExtractor(warmup_steps=profile.schedule.warmup_steps),
        )
        self._opened[cell] = sink
        self._began[cell] = time.perf_counter()
        return sink

    def lifecycle(self, cell: str, value: CellLifecycle) -> None:
        self._opened[cell].annotate({LIFECYCLE_TAG: value})

    def pair_outcome(self, value: PairOutcome) -> None:
        self._opened[self._request.tokens[0]].annotate({PAIR_OUTCOME_TAG: value})

    def finish(self, cell: str, status: Literal["FINISHED", "FAILED"]) -> None:
        if cell in self._finished:
            return
        self._finished.add(cell)
        self._opened[cell].finish(status=status)
        # each cell's log closes with its OWN worker.end, written before the next cell
        # re-points the substrate's sink (after sink.finish, exactly as for a single profile)
        ended = {
            "status": "ok" if status == "FINISHED" else "failed",
            "seconds": time.perf_counter() - self._began[cell],
        }
        logger.bind(event=WORKER_END, **ended).info(render(WORKER_END, **ended))

    def finish_open(self) -> None:
        """Terminate every cell run still open as FAILED (an exit that bypassed its cell's end)."""
        for cell in self._opened:
            self.finish(cell, "FAILED")


def _scope(request: AnyProbeRequest, local: bool, object_limit: int, count_limit: int) -> None:
    expected = "local_worker" if local else "remote_worker"
    # a pair is scoped by its cells' shared run class (PairProbeRequest refuses two classes)
    run_class = (
        request.donor.run_class
        if isinstance(request, PairProbeRequest)
        else request.profile.run_class
    )
    if RUN_ADMISSION.worker_surface(run_class) != expected:
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
    stage = "arguments"
    # FOR-1637: set once `worker.start` has been logged (i.e. after the request is
    # admitted and the structured sink points at worker-log.jsonl); `worker.end`
    # is logged exactly once per such run, after sink.finish(), on both exits.
    # A run refused before that point logs neither: nothing was started.
    worker_started: float | None = None
    worker_ended = False
    try:
        args = _parser().parse_args(list(argv))
        if args.max_object_bytes < 1 or args.max_objects < 1:
            raise ValueError("worker command limits must be positive")
        stage = "object_client"
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
            stage = "request_fetch"
            marker = D1TrainerMailboxObjectIdentity(
                args.request_key, args.request_size, args.request_sha256
            )
            request = read_published_probe_request(
                selected, marker, max_request_bytes=args.max_object_bytes
            )
            stage = "request_admission"
            _scope(request, args.local_tiny, args.max_object_bytes, args.max_objects)
            if not args.local_tiny:
                require_d1_pod_id(environ.get("RUNPOD_POD_ID"))
            stage = "sink_construction"
            # A SIBLING of the workspace root, never a path under it:
            # run_object_worker's own materialize_probe_request step requires
            # the workspace root be completely empty at call time (worker_transport.py,
            # normalize_d1_artifact_root's any(target.iterdir()) check) --
            # constructing JsonlSink under the workspace root would mkdir()
            # it into existence first and turn that admission check RED on
            # every run. MEASURED (RUNG 2b): this is exactly the "D1 root
            # is directory-scanned" risk this module's own history flags --
            # confirmed by a direct reproduction before landing this fix.
            observability_root = (
                args.workspace_root.parent / f"{args.workspace_root.name}.observability"
            )
            if isinstance(request, PairProbeRequest):
                # FOR-1693 (D11, DH-40): one MLflow run, metrics file and worker log PER CELL,
                # each built by the provider as its cell begins (the one log re-point happens
                # inside it); the run terminates per cell, and a cell the pair loop never closed
                # (an exit that bypassed it) is finished FAILED here.
                # no command-level worker.start/worker.end: each cell's log carries its own pair
                stage = "worker_execution"
                provider = _PairCellSinks(request, observability_root)
                try:
                    result = run_object_worker(
                        selected,
                        marker,
                        args.workspace_root,
                        max_request_bytes=args.max_object_bytes,
                        repo_root=args.repo_root,
                        cell_sinks=provider,
                    )
                finally:
                    provider.finish_open()
            else:
                # FOR-1637 (G5): the ONE re-point of the substrate's JSON sink, here
                # and nowhere else -- after admission, before build_sink_for_run, so
                # worker.start, mlflow.run and every later line land in the file the
                # MlflowSink uploads at finish(). configure_logging() removes every
                # handler first, so nothing is printed twice.
                worker_log = observability_root / "worker-log.jsonl"
                configure_logging(json_path=worker_log)
                worker_started = time.perf_counter()
                logger.bind(
                    event=WORKER_START,
                    request_sha256=request.digest,
                    image=request.image,
                    code_commit=request.code_commit,
                ).info(
                    render(
                        WORKER_START,
                        request_sha256=request.digest,
                        image=request.image,
                        code_commit=request.code_commit,
                    )
                )
                # FOR-1642: the H run surface. `config_hash` (the seed-aggregation grouping
                # key) is `h_config_hash(profile)`, identical in params and tags. `git_sha`
                # and `image` come from the VERIFIED request, unswapped, and are OMITTED when
                # the request has none (a local_tiny request forbids both): MLflow params
                # cannot be None, and a local `git_sha()` would put an unverified HEAD under
                # a verified key.
                chash = h_config_hash(request.profile)
                params: dict[str, str | int] = {
                    EXP_TAG: request.profile.name,
                    "seed": request.profile.seed,
                    CONFIG_HASH_TAG: chash,
                    "run_class": request.profile.run_class,
                    "precision": request.profile.precision,
                }
                if request.code_commit is not None:
                    params["git_sha"] = request.code_commit
                if request.image is not None:
                    params["image"] = request.image
                # Rung B: a smoke run stays behind the tiny MLflow opt-in
                # (``ENTROPIC_MLFLOW_TINY=1`` in its launch template); a scientific run is
                # tracked whenever a tracking URI is set, because its eight-key template
                # carries no tiny opt-in and ``tiny=True`` would drop its mirror SILENTLY
                # (``sinks.mlflow_enabled``). It tags the sealed authority it runs under.
                if request.authority is not None:
                    params["run_authority"] = request.authority.digest
                sink = build_sink_for_run(
                    TrackingRunDescriptor(
                        run_name=f"h-worker-{request.profile.identity}",
                        tiny=request.authority is None,
                        params=params,
                        tags={
                            "schema": RESULT_SCHEMA_V1,
                            EXP_TAG: request.profile.name,
                            CONFIG_HASH_TAG: chash,
                            "request_sha256": request.digest,
                        },
                    ),
                    results_path=observability_root / "metrics.jsonl",
                    log_path=worker_log,
                    metric_extractor=HTrainingStepExtractor(
                        warmup_steps=request.profile.schedule.warmup_steps
                    ),
                )
                # DN-1643-PORT: the run terminates FINISHED iff run_object_worker RETURNED.
                # Every raised exit, BaseException included (KeyboardInterrupt, SystemExit),
                # leaves the FAILED default in place and finishes the sink FAILED exactly
                # once; the exception itself is untouched and still reaches the handler
                # below (stage worker_execution, exit 2). Finishing here, not after the try,
                # keeps sink.finish before worker.end on both exits.
                terminal_status = "FAILED"
                try:
                    stage = "worker_execution"
                    result = run_object_worker(
                        selected,
                        marker,
                        args.workspace_root,
                        max_request_bytes=args.max_object_bytes,
                        repo_root=args.repo_root,
                        sink=sink,
                    )
                    terminal_status = "FINISHED"
                finally:
                    sink.finish(status=terminal_status)
            worker_ended = True
            if worker_started is not None:
                ended = {"status": "ok", "seconds": time.perf_counter() - worker_started}
                logger.bind(event=WORKER_END, **ended).info(render(WORKER_END, **ended))
            response = {
                "status": "worker_returned",
                "scientific_authority": "none",
                "object": asdict(result),
            }
        else:
            stage = "request_file"
            _, payload, _ = read_d1_artifact_bytes(
                args.artifact_root,
                args.request_file,
                label="worker command request",
                max_bytes=args.max_object_bytes,
            )
            request = parse_probe_request(payload, max_bytes=args.max_object_bytes)
            if args.command == "publish":
                stage = "request_publish"
                # FOR-1694: a pair is published like any request (``publish_probe_request`` takes
                # both kinds and is the only publisher: without it no pair passes the allocation's
                # published-request byte check)
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
                stage = "completion_receive"
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
        if worker_started is not None and not worker_ended:
            worker_ended = True
            # C9/G21: the failing stage and the exception TYPE name only.
            failed = {
                "status": "failed",
                "seconds": time.perf_counter() - worker_started,
                "stage": stage,
                "error_type": type(error).__name__,
            }
            logger.bind(event=WORKER_END, **failed).warning(render(WORKER_END, **failed))
        stderr.write(
            json.dumps(
                {
                    "status": "error",
                    "code": "worker_command_refused",
                    "error_type": type(error).__name__,
                    "scientific_authority": "none",
                    "stage": stage,
                    **(
                        {"object_error": error.safe_diagnostic()}
                        if isinstance(error, D1TrainerS3ObjectError)
                        else {}
                    ),
                    **(
                        {"credential_error": "unresolved_reference"}
                        if isinstance(error, UnresolvedWorkerCredentials)
                        else {}
                    ),
                },
                sort_keys=True,
            )
            + "\n"
        )
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Configure logging, then parse ``argv`` and dispatch. Returns a process exit code."""
    configure_logging()
    return run_worker_command(sys.argv[1:] if argv is None else argv, environ=os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
