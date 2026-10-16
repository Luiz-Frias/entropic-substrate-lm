"""Read-only monitor for a wave of H0 probe arms (FOR-1646).

Promoted from untracked scratch under the tracked contract: it reads MLflow and
narrates one progress line per arm per tick through the log substrate. It never
launches, deletes or terminates anything (the scratch ``--reap`` is NOT promoted),
and it constructs no metric sink.

Arm identity comes only from tracked contracts: a launch record parsed with
``parse_probe_launch`` (the run name is ``h-worker-{request.profile.identity}``,
cited from ``ladder/run/worker_command.py:212``, ``run_name=f"h-worker-{...}"``)
or an explicit ``--run-name``. The experiment is the single in-tree constant
``H_SERIES_EXPERIMENT``; the scratch's hard-coded experiment id is not used.

Progress here is the per-step MLflow evidence source. Each line names its source
and renders a missing run, a failed read or a failed search as its own explicit
state, through the same builder the controller's R2 checkpoint line uses
(``progress_fields``), so the two sources cannot be conflated.

MLflow is reached over stdlib REST, not ``build_mlflow_client``: the client has no
per-call timeout parameter, so the required ``--request-timeout-seconds`` is only
honestly bindable on a request this module issues itself. The tracking URI comes
from ``MLFLOW_TRACKING_URI`` and basic-auth credentials from
``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD``; none of them is ever
rendered, and a URI carrying userinfo is refused unread.

Attempts (DN-1646-ATTEMPTS, 2026-10-16): a fresh-nonce retry keeps ``profile.identity``,
so same-named runs in the experiment are ATTEMPTS of one arm. An arm is terminal only
when EVERY attempt the search returns is terminal. The tick's step and ``run_status``
come from the NEWEST attempt (greatest ``start_time``; the response order, which MLflow
serves newest-first, when no start time is given), never from the highest step across
attempts; a live attempt takes precedence over a newer terminal one, because the arm is
then still live. The tick carries the attempt count as ``attempts``. The newest-first
claim is measured against the installed mlflow-skinny 3.15.0 client-side store
(``mlflow/store/tracking/sqlalchemy_store.py``, ``_get_orderby_clauses`` at line 9372,
which appends ``start_time DESC`` and then ``run_uuid``); the tracking SERVER's version is
not measured here, so the ``start_time`` sort above is the load-bearing pin and the
response order only breaks ties. The search returns at most ``_SEARCH_RESULTS`` (five)
rows: a declared limit of five attempts PER LAUNCH on the tag-bound path (a pod restart
duplicates a run only a few times), with no paging.

Launch identity (FOR-1657, 2026-10-16): the worker tags its run ``request_sha256`` with
the request digest (``ladder/run/worker_command.py``, the descriptor tag dict), and every
``--launch-file`` arm searches ``attributes.run_name = '<name>' AND tags.request_sha256 =
'<digest>'`` (the digest from ``parse_probe_launch(...).request.digest``). The attempts
rule above then ranges over ONE launch: an earlier launch's terminal run of the same name
no longer stands in for a retry whose pod is still allocating (that arm reads
``run_not_found`` and keeps ticking). The residual gap of the name-only search is therefore
CLOSED for launch-file arms and still OPEN for ``--run-name`` arms, which keep the bare
name filter and so can mix attempts from different launches. Compatibility boundary: a
launch-file arm finds only runs from a worker image built at or after this slice; a run
from an older image carries no tag and is visible only through ``--run-name``.

Exit codes (three honest exits plus usage and interrupt): 0 every arm's run reached
a terminal MLflow status; 3 the tick budget was exhausted with arms still running; 1
failure (unusable launch, tracking URI or source); 2 usage; 130 interrupt. Nothing
runs and no environment is read at import.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from entropic_substrate.kernel.artifacts import read_d1_artifact_bytes
from entropic_substrate.kernel.runtime.logging import configure_logging
from entropic_substrate.ladder.run.h_probe_launch import H_SERIES_EXPERIMENT, parse_probe_launch
from entropic_substrate.ladder.run.worker.h_probe_host_log import emit
from entropic_substrate.ladder.run.worker.h_probe_log_events import (
    HOST_MONITOR_END,
    HOST_MONITOR_TICK,
    SOURCE_MLFLOW_METRIC,
    STATE_LIST_FAILED,
    STATE_NO_PROGRESS,
    STATE_READ_FAILED,
    STATE_RUN_NOT_FOUND,
    STATE_STEP,
    progress_fields,
)

__all__ = [
    "EXIT_EXHAUSTED",
    "EXIT_FAILED",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "EXIT_USAGE",
    "MlflowRestSource",
    "RunReading",
    "RunSource",
    "main",
    "run_name_for",
    "run_wave_monitor",
]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_EXHAUSTED = 3
EXIT_INTERRUPTED = 130

# The MLflow metric the H training extractor logs on every committed step
# (ladder/run/h_metric_extractor.py, HTrainingStepExtractor._training: "phase_index"),
# on the global optimizer-step axis. System metrics also carry steps (their own sample
# counter), so the step is read from this key only, never from "any metric".
STEP_METRIC_KEY = "phase_index"
_TERMINAL_RUN_STATUSES = frozenset({"FINISHED", "FAILED", "KILLED"})
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}")
# The worker's ``request_sha256`` tag: a sha256 hex digest (full-string match only).
_REQUEST_SHA256 = re.compile(r"[0-9a-f]{64}")
_SEARCH_RESULTS = 5
# A response bound, not a retry or timeout: a hostile or broken server cannot make this
# process buffer without limit.
_MAX_RESPONSE_BYTES = 4_194_304


@dataclass(frozen=True, slots=True)
class RunReading:
    """What one read of one arm established. ``step`` exists iff ``state == 'step'``.

    ``step`` and ``run_status`` are the displayed attempt's. ``attempts`` (same-named runs
    seen) and ``arm_terminal`` (every attempt terminal; ``None`` = judge ``run_status``
    alone) are additive facts that never take part in equality.
    """

    state: str
    step: int | None = None
    run_status: str | None = None
    attempts: int | None = field(default=None, compare=False)
    arm_terminal: bool | None = field(default=None, compare=False)


class RunSource(Protocol):
    def latest(
        self, experiment_name: str, run_name: str, *, request_sha256: str | None = None
    ) -> RunReading: ...


def run_name_for(profile_identity: str) -> str:
    """The worker's MLflow run name (ladder/run/worker_command.py:212)."""
    return f"h-worker-{profile_identity}"


class MlflowRestSource:
    """The newest reading of a named run over stdlib REST; credentials never leave it."""

    def __init__(
        self,
        tracking_uri: str,
        *,
        username: str | None,
        password: str | None,
        timeout_seconds: float,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        parts = urllib.parse.urlsplit(tracking_uri)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            raise ValueError("tracking URI must be an http(s) URL without credentials")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("request timeout must be finite and positive")
        if (username is None) != (password is None):
            raise ValueError("tracking credentials must be provided together")
        self._base = tracking_uri.rstrip("/")
        self._timeout = timeout_seconds
        self._opener = opener
        self._authorization = (
            None
            if username is None or password is None
            else "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
        )

    def _call(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        body: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        url = self._base + path + ("?" + urllib.parse.urlencode(query) if query else "")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._authorization is not None:
            headers["Authorization"] = self._authorization
        request = urllib.request.Request(
            url,
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers=headers,
        )
        with self._opener(request, timeout=self._timeout) as response:
            raw = cast(bytes, response.read(_MAX_RESPONSE_BYTES + 1))
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("tracking response exceeds its byte bound")
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError("tracking response must be an object")
        return cast(dict[str, object], value)

    def latest(
        self, experiment_name: str, run_name: str, *, request_sha256: str | None = None
    ) -> RunReading:
        if _RUN_NAME.fullmatch(run_name) is None:
            raise ValueError("run name must be a safe identifier")
        if request_sha256 is not None and _REQUEST_SHA256.fullmatch(request_sha256) is None:
            raise ValueError("request digest must be 64 lowercase hex characters")
        run_filter = f"attributes.run_name = '{run_name}'"
        if request_sha256 is not None:
            run_filter += f" AND tags.request_sha256 = '{request_sha256}'"
        try:
            experiment = self._call(
                "GET",
                "/api/2.0/mlflow/experiments/get-by-name",
                query={"experiment_name": experiment_name},
            )
            runs = self._call(
                "POST",
                "/api/2.0/mlflow/runs/search",
                body={
                    "experiment_ids": [
                        cast(dict[str, object], experiment["experiment"])["experiment_id"]
                    ],
                    "filter": run_filter,
                    "max_results": _SEARCH_RESULTS,
                },
            )
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return RunReading(STATE_RUN_NOT_FOUND)
            return RunReading(STATE_LIST_FAILED)
        except (urllib.error.URLError, TimeoutError, OSError):
            return RunReading(STATE_LIST_FAILED)
        except (ValueError, KeyError, TypeError):
            return RunReading(STATE_READ_FAILED)
        try:
            return _arm_reading(runs.get("runs", []))
        except (ValueError, KeyError, TypeError):
            return RunReading(STATE_READ_FAILED)


def _arm_reading(runs: object) -> RunReading:
    """One arm from its same-named runs (attempts); none at all is 'not found'.

    The displayed attempt is the newest, a live attempt taking precedence over a newer
    terminal one; the arm is terminal only when every attempt is.
    """
    if type(runs) is not list:
        raise ValueError("runs must be a list")
    if not runs:
        return RunReading(STATE_RUN_NOT_FOUND)
    readings: list[tuple[int, RunReading]] = []
    for run in runs:
        row = cast(dict[str, object], run)
        info = cast(dict[str, object], row["info"])
        status = info["status"]
        if type(status) is not str:
            raise ValueError("run status must be text")
        started = info.get("start_time", 0)
        if type(started) is not int:
            raise ValueError("run start time must be an integer")
        metrics = cast(dict[str, object], row["data"]).get("metrics", [])
        if type(metrics) is not list:
            raise ValueError("run metrics must be a list")
        steps = [
            cast(int, metric["step"])
            for metric in cast(list[dict[str, object]], metrics)
            if metric["key"] == STEP_METRIC_KEY
        ]
        if any(type(step) is not int or step < 1 for step in steps):
            raise ValueError("metric steps must be positive integers")
        readings.append(
            (
                started,
                RunReading(STATE_STEP, max(steps), status)
                if steps
                else RunReading(STATE_NO_PROGRESS, None, status),
            )
        )
    newest_first = [reading for _, reading in sorted(readings, key=lambda item: -item[0])]  # stable
    shown = next(
        (r for r in newest_first if r.run_status not in _TERMINAL_RUN_STATUSES), newest_first[0]
    )
    return RunReading(
        shown.state,
        shown.step,
        shown.run_status,
        attempts=len(newest_first),
        arm_terminal=all(r.run_status in _TERMINAL_RUN_STATUSES for r in newest_first),
    )


@dataclass(frozen=True, slots=True)
class _Arm:
    run_name: str
    total: int | None
    request_sha256: str | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="h-wave-monitor", description="read-only progress monitor for H0 probe arms"
    )
    parser.add_argument("--launch-file", type=Path, action="append", default=[])
    parser.add_argument("--max-launch-bytes", type=int)
    parser.add_argument("--run-name", action="append", default=[])
    for name, kind in (
        ("interval-seconds", float),
        ("request-timeout-seconds", float),
        ("ticks", int),
    ):
        parser.add_argument(f"--{name}", type=kind, required=True)
    return parser


def _arms(args: argparse.Namespace) -> list[_Arm]:
    arms: list[_Arm] = []
    for path in cast(list[Path], args.launch_file):
        if args.max_launch_bytes is None:
            raise ValueError("--max-launch-bytes is required with --launch-file")
        _, payload, _ = read_d1_artifact_bytes(
            path.parent, path.name, label="H0 launch", max_bytes=args.max_launch_bytes
        )
        request = parse_probe_launch(payload, max_bytes=args.max_launch_bytes).request
        schedule = request.profile.schedule
        arms.append(
            _Arm(
                run_name_for(request.profile.identity),
                schedule.warmup_steps + schedule.steps,
                request.digest,
            )
        )
    for name in cast(list[str], args.run_name):
        if _RUN_NAME.fullmatch(name) is None:
            raise ValueError("run name must be a safe identifier")
        arms.append(_Arm(name, None))
    if not arms:
        raise ValueError("at least one --launch-file or --run-name is required")
    if len({arm.run_name for arm in arms}) != len(arms):
        raise ValueError("arms must be distinct")
    return arms


def _default_source(environ: Mapping[str, str], timeout_seconds: float) -> RunSource:
    uri = environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise ValueError("MLFLOW_TRACKING_URI is required")
    return MlflowRestSource(
        uri,
        username=environ.get("MLFLOW_TRACKING_USERNAME"),
        password=environ.get("MLFLOW_TRACKING_PASSWORD"),
        timeout_seconds=timeout_seconds,
    )


def run_wave_monitor(
    argv: Sequence[str],
    *,
    environ: Mapping[str, str],
    source_factory: Callable[[Mapping[str, str], float], RunSource] = _default_source,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Read-only. The tick budget and the interval are required arguments with no default."""
    try:
        args = _parser().parse_args(list(argv))
    except SystemExit as error:
        return EXIT_OK if error.code == 0 else EXIT_USAGE
    ticks, interval, timeout = (
        cast(int, args.ticks),
        cast(float, args.interval_seconds),
        cast(float, args.request_timeout_seconds),
    )
    if (
        ticks < 1
        or not math.isfinite(interval)
        or interval <= 0
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        return EXIT_USAGE
    done = 0
    try:
        arms = _arms(args)
        source = source_factory(environ, timeout)
        for done in range(1, ticks + 1):
            terminal = 0
            for arm in arms:
                try:
                    reading = (
                        source.latest(H_SERIES_EXPERIMENT, arm.run_name)
                        if arm.request_sha256 is None
                        else source.latest(
                            H_SERIES_EXPERIMENT, arm.run_name, request_sha256=arm.request_sha256
                        )
                    )
                except Exception:
                    reading = RunReading(STATE_READ_FAILED)
                fields = progress_fields(
                    source=SOURCE_MLFLOW_METRIC,
                    state=reading.state,
                    step=reading.step,
                    total=arm.total,
                )
                if reading.run_status is not None:
                    fields["run_status"] = reading.run_status
                if reading.attempts is not None:
                    fields["attempts"] = reading.attempts
                emit("info", HOST_MONITOR_TICK, run_name=arm.run_name, tick=done, **fields)
                terminal += (
                    reading.arm_terminal
                    if reading.arm_terminal is not None
                    else reading.run_status in _TERMINAL_RUN_STATUSES
                )
            if terminal == len(arms):
                emit("info", HOST_MONITOR_END, outcome="all_runs_terminal", ticks=done)
                return EXIT_OK
            if done < ticks:
                sleep(interval)
        emit("warning", HOST_MONITOR_END, outcome="budget_exhausted", ticks=done)
        return EXIT_EXHAUSTED
    except KeyboardInterrupt:
        emit("warning", HOST_MONITOR_END, outcome="interrupted", ticks=done)
        return EXIT_INTERRUPTED
    except Exception as error:
        emit(
            "error", HOST_MONITOR_END, outcome="failed", error_type=type(error).__name__, ticks=done
        )
        return EXIT_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    return run_wave_monitor(sys.argv[1:] if argv is None else argv, environ=os.environ)


if __name__ == "__main__":
    sys.exit(main())
