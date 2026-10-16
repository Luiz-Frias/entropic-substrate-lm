"""FOR-1646: the tracked wave monitor is read-only, layer-clean and shares the progress renderer.

Everything is driven through the monitor's two seams, the injected ``RunSource`` (a fake
client for the entry point) and the injected ``opener`` (a fake HTTP layer for the stdlib REST
source). Nothing here reaches a network, a provider or a clock.
"""

from __future__ import annotations

import ast
import base64
import json
import subprocess
import sys
import urllib.error
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from tests.test_h_probe_launch import launch

import entropic_substrate
from entropic_substrate.composition.entrypoints import h_wave_monitor as monitor
from entropic_substrate.composition.entrypoints.h_wave_monitor import (
    EXIT_EXHAUSTED,
    EXIT_FAILED,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_USAGE,
    MlflowRestSource,
    RunReading,
    run_name_for,
    run_wave_monitor,
)
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.run.h_probe_launch import H_SERIES_EXPERIMENT
from entropic_substrate.ladder.run.worker import h_probe_controller, h_probe_log_events

pytestmark = pytest.mark.unit

PKG = Path(entropic_substrate.__file__).resolve().parent
MODULE = PKG / "composition/entrypoints/h_wave_monitor.py"
SECRET_PASSWORD = "p4ssw0rd-never-rendered"  # pragma: allowlist secret
SECRET_USER = "user-never-rendered"  # pragma: allowlist secret
URI = "https://tracking.example.test/base"


@contextmanager
def captured() -> Any:
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


class Source:
    """A scripted client: one reading per call, the last one repeating."""

    def __init__(self, *readings: RunReading | Exception) -> None:
        self.readings, self.calls = list(readings), []

    def latest(self, experiment: str, run_name: str) -> RunReading:
        self.calls.append((experiment, run_name))
        item = self.readings[min(len(self.calls) - 1, len(self.readings) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


def argv(*extra: str, ticks: str = "3") -> list[str]:
    return [
        "--interval-seconds", "7", "--request-timeout-seconds", "5", "--ticks", ticks, *extra,
    ]  # fmt: skip


def go(
    source: Source, *extra: str, ticks: str = "3", sleeps: list[float] | None = None
) -> tuple[int, list[dict[str, Any]]]:
    pauses = [] if sleeps is None else sleeps
    with captured() as records:
        code = run_wave_monitor(
            argv(*extra, ticks=ticks),
            environ={},
            source_factory=lambda _env, _timeout: source,
            sleep=pauses.append,
        )
    return code, [r for r in records if r.get("event", "").startswith("host.monitor")]


# ----------------------------------------------------- the entry point and its exits
def test_every_tick_names_the_arm_and_the_source_and_exhaustion_is_its_own_exit() -> None:
    source = Source(RunReading("step", 5, "RUNNING"))
    sleeps: list[float] = []
    code, records = go(source, "--run-name", "h-worker-a", ticks="3", sleeps=sleeps)
    assert code == EXIT_EXHAUSTED
    ticks = [r for r in records if r["event"] == "host.monitor.tick"]
    assert len(ticks) == 3 and sleeps == [
        7.0,
        7.0,
    ]  # the interval between ticks, none after the last
    assert source.calls == [(H_SERIES_EXPERIMENT, "h-worker-a")] * 3
    for tick, number in zip(ticks, (1, 2, 3), strict=True):
        assert tick["run_name"] == "h-worker-a" and tick["tick"] == number
        assert tick["source"] == "mlflow_metric" and tick["state"] == "step" and tick["step"] == 5
        assert tick["run_status"] == "RUNNING" and tick["level"] == "INFO"
    (end,) = [r for r in records if r["event"] == "host.monitor.end"]
    assert end["outcome"] == "budget_exhausted" and end["ticks"] == 3


def test_all_arms_terminal_ends_early_with_success() -> None:
    source = Source(RunReading("step", 6, "RUNNING"), RunReading("step", 6, "FINISHED"))
    code, records = go(source, "--run-name", "h-worker-a", ticks="9")
    assert code == EXIT_OK
    (end,) = [r for r in records if r["event"] == "host.monitor.end"]
    assert end["outcome"] == "all_runs_terminal" and end["ticks"] == 2


@pytest.mark.parametrize(
    "reading,state",
    [
        (RunReading("run_not_found"), "run_not_found"),
        (RunReading("list_failed"), "list_failed"),
        (RunReading("read_failed"), "read_failed"),
        (RunReading("no_progress_yet", None, "RUNNING"), "no_progress_yet"),
    ],
)
def test_a_missing_run_or_failed_read_is_an_explicit_state_never_a_step(
    reading: RunReading, state: str
) -> None:
    _code, records = go(Source(reading), "--run-name", "h-worker-a", ticks="1")
    (tick,) = [r for r in records if r["event"] == "host.monitor.tick"]
    assert tick["state"] == state and "step" not in tick and "step=" not in tick["message"]


def test_a_source_that_raises_is_a_read_failed_tick_not_a_monitor_failure() -> None:
    code, records = go(
        Source(RuntimeError("boom"), RunReading("step", 2, "FINISHED")), "--run-name", "h-worker-a"
    )
    ticks = [r for r in records if r["event"] == "host.monitor.tick"]
    assert [t["state"] for t in ticks] == ["read_failed", "step"] and code == EXIT_OK
    assert "boom" not in json.dumps(records)


def test_each_exit_has_its_own_code_and_a_named_end_event() -> None:
    codes = {EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_EXHAUSTED, EXIT_INTERRUPTED}
    assert len(codes) == 5 and EXIT_INTERRUPTED == 130

    # failure: the source cannot be built
    def refuse(_env: Mapping[str, str], _timeout: float) -> Any:
        raise ValueError("MLFLOW_TRACKING_URI is required")

    with captured() as records:
        code = run_wave_monitor(
            argv("--run-name", "h-worker-a"),
            environ={},
            source_factory=refuse,
            sleep=lambda _s: None,
        )
    (end,) = [r for r in records if r.get("event") == "host.monitor.end"]
    assert code == EXIT_FAILED and end["outcome"] == "failed" and end["error_type"] == "ValueError"
    assert end["level"] == "ERROR"
    # interrupt
    interrupted, records = go(Source(KeyboardInterrupt()), "--run-name", "h-worker-a")
    assert interrupted == EXIT_INTERRUPTED
    assert next(r for r in records if r["event"] == "host.monitor.end")["outcome"] == "interrupted"


@pytest.mark.parametrize(
    "arguments",
    [
        [],  # nothing required is present
        [
            "--run-name",
            "h-worker-a",
            "--request-timeout-seconds",
            "5",
            "--ticks",
            "3",
        ],  # no interval
        ["--run-name", "h-worker-a", "--interval-seconds", "7", "--ticks", "3"],  # no http timeout
        [
            "--run-name",
            "h-worker-a",
            "--interval-seconds",
            "7",
            "--request-timeout-seconds",
            "5",
        ],  # no bound
        [
            "--run-name",
            "h-worker-a",
            "--interval-seconds",
            "0",
            "--request-timeout-seconds",
            "5",
            "--ticks",
            "3",
        ],
        [
            "--run-name",
            "h-worker-a",
            "--interval-seconds",
            "7",
            "--request-timeout-seconds",
            "nan",
            "--ticks",
            "3",
        ],
        [
            "--run-name",
            "h-worker-a",
            "--interval-seconds",
            "7",
            "--request-timeout-seconds",
            "5",
            "--ticks",
            "0",
        ],
    ],
)
def test_the_interval_timeout_and_bound_are_required_positive_arguments_with_no_default(
    arguments: list[str],
) -> None:
    source = Source(RunReading("step", 1, "FINISHED"))
    code = run_wave_monitor(
        arguments, environ={}, source_factory=lambda _e, _t: source, sleep=lambda _s: None
    )
    assert code == EXIT_USAGE and not source.calls


@pytest.mark.parametrize(
    "extra",
    [
        [],  # no arm
        ["--run-name", "bad name"],
        ["--run-name", "h-worker-a", "--run-name", "h-worker-a"],  # duplicate arm
        ["--launch-file", "x.json"],  # a launch file without its byte bound
    ],
)
def test_arms_must_be_declared_distinct_and_safe(extra: list[str]) -> None:
    code, records = go(Source(RunReading("step", 1, "FINISHED")), *extra)
    assert code == EXIT_FAILED
    assert next(r for r in records if r["event"] == "host.monitor.end")["outcome"] == "failed"


def test_an_arm_comes_from_a_parsed_launch_record_with_the_workers_run_name(tmp_path: Path) -> None:
    value = launch()
    (tmp_path / "launch.json").write_bytes(value.payload)
    source = Source(RunReading("step", 4, "RUNNING"))
    code, records = go(
        source,
        "--launch-file",
        str(tmp_path / "launch.json"),
        "--max-launch-bytes",
        "1000000",
        ticks="1",
    )
    expected = f"h-worker-{value.request.profile.identity}"  # ladder/run/worker_command.py:212
    assert run_name_for(value.request.profile.identity) == expected
    assert source.calls == [(H_SERIES_EXPERIMENT, expected)] and code == EXIT_EXHAUSTED
    (tick,) = [r for r in records if r["event"] == "host.monitor.tick"]
    schedule = value.request.profile.schedule
    assert tick["run_name"] == expected and tick["total"] == schedule.warmup_steps + schedule.steps


def test_the_experiment_is_the_one_in_tree_constant() -> None:
    from entropic_substrate.ladder.run.h_probe_launch import named_experiment

    assert named_experiment(H_SERIES_EXPERIMENT).env_value == H_SERIES_EXPERIMENT == "h-series"
    tree = ast.parse(MODULE.read_text())
    literals = [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "h-series"
    ]
    assert not literals, "the experiment name must be read from the constant, never repeated"


# --------------------------------------------------- one renderer, two sources
def test_the_controller_and_the_monitor_build_their_lines_with_the_one_shared_builder() -> None:
    assert monitor.progress_fields is h_probe_log_events.progress_fields
    assert h_probe_controller.progress_fields is h_probe_log_events.progress_fields
    assert monitor.render is h_probe_log_events.render is h_probe_controller.render
    builder_defs = [
        path.name for path in PKG.rglob("*.py") if "def progress_fields" in path.read_text()
    ]
    assert builder_defs == ["h_probe_log_events.py"]


def test_the_two_sources_render_distinguishable_lines_for_the_same_step() -> None:
    r2 = h_probe_log_events.progress_fields(
        source="r2_checkpoint_event", state="step", step=4, total=6
    )
    mlflow = h_probe_log_events.progress_fields(
        source="mlflow_metric", state="step", step=4, total=6
    )
    first = h_probe_log_events.render("host.poll.tick", **r2)
    second = h_probe_log_events.render("host.monitor.tick", **mlflow)
    assert "r2_checkpoint_event" in first and "mlflow_metric" in second and first != second


# ----------------------------------------------------- the stdlib REST source
class Response:
    def __init__(self, payload: object) -> None:
        self.raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_a: object) -> None:
        return None

    def read(self, limit: int = -1) -> bytes:
        return self.raw if limit < 0 else self.raw[:limit]


class Opener:
    def __init__(self, *answers: object) -> None:
        self.answers, self.requests = list(answers), []

    def __call__(self, request: Any, timeout: float) -> Response:
        self.requests.append((request, timeout))
        answer = self.answers[min(len(self.requests) - 1, len(self.answers) - 1)]
        if isinstance(answer, BaseException):
            raise answer
        return Response(answer)


def rest(
    opener: Opener, *, user: str | None = SECRET_USER, password: str | None = SECRET_PASSWORD
) -> MlflowRestSource:
    return MlflowRestSource(
        URI, username=user, password=password, timeout_seconds=4.5, opener=opener
    )


EXPERIMENT = {"experiment": {"experiment_id": "7"}}


def run_row(
    status: str, *metrics: tuple[str, int], start_time: int | None = None
) -> dict[str, Any]:
    return {
        "info": {
            "run_id": "r",
            "status": status,
            **({} if start_time is None else {"start_time": start_time}),
        },
        "data": {
            "metrics": [{"key": k, "value": 1.0, "step": s, "timestamp": 0} for k, s in metrics]
        },
    }


def test_the_rest_source_asks_the_h_series_experiment_for_the_named_run_with_the_required_timeout() -> (
    None
):
    opener = Opener(EXPERIMENT, {"runs": [run_row("RUNNING", ("phase_index", 12))]})
    reading = rest(opener).latest("h-series", "h-worker-abc")
    assert reading == RunReading("step", 12, "RUNNING")
    (lookup, _), (search, _) = opener.requests
    assert lookup.get_method() == "GET" and "experiment_name=h-series" in lookup.full_url
    assert lookup.full_url.startswith(URI + "/api/2.0/mlflow/experiments/get-by-name")
    assert search.get_method() == "POST" and search.full_url == URI + "/api/2.0/mlflow/runs/search"
    body = json.loads(search.data)
    assert body["experiment_ids"] == ["7"]
    assert body["filter"] == "attributes.run_name = 'h-worker-abc'"
    assert {t for _, t in opener.requests} == {4.5}
    expected = "Basic " + base64.b64encode(f"{SECRET_USER}:{SECRET_PASSWORD}".encode()).decode()
    assert lookup.get_header("Authorization") == expected


def test_the_step_is_the_per_step_training_metric_only_and_the_best_same_named_run_wins() -> None:
    runs = [
        run_row("FAILED", ("phase_index", 3), ("system/cpu_utilization_percentage", 900)),
        run_row("RUNNING", ("phase_index", 8)),
        run_row("RUNNING", ("system/cpu_utilization_percentage", 5)),  # no training step at all
    ]
    reading = rest(Opener(EXPERIMENT, {"runs": runs})).latest("h-series", "h-worker-abc")
    assert reading == RunReading("step", 8, "RUNNING")
    only_system = rest(Opener(EXPERIMENT, {"runs": [runs[2]]})).latest("h-series", "h-worker-abc")
    assert only_system == RunReading("no_progress_yet", None, "RUNNING")


def attempts_wave(*runs: dict[str, Any], ticks: str = "2") -> tuple[int, list[dict[str, Any]]]:
    """The real monitor over the real REST source; the search answers repeat per tick."""
    answers: list[object] = []
    for _ in range(int(ticks)):
        answers += [EXPERIMENT, {"runs": list(runs)}]
    opener = Opener(*answers)
    with captured() as records:
        code = run_wave_monitor(
            argv("--run-name", "h-worker-abc", ticks=ticks),
            environ={},
            source_factory=lambda _env, _timeout: rest(opener),
            sleep=lambda _seconds: None,
        )
    return code, [r for r in records if r.get("event", "").startswith("host.monitor")]


def test_a_live_retry_beside_an_older_terminal_attempt_is_not_a_terminal_arm() -> None:
    """DN-1646-ATTEMPTS: same-named runs are attempts of one arm (a fresh-nonce retry keeps
    ``profile.identity``). The newest-first search answer is [newer RUNNING step 3, older
    FAILED step 8]: the arm is live, and the tick reads the newest attempt, not step 8."""
    code, records = attempts_wave(
        run_row("RUNNING", ("phase_index", 3), start_time=2000),
        run_row("FAILED", ("phase_index", 8), start_time=1000),
    )
    assert code == EXIT_EXHAUSTED
    ticks = [r for r in records if r["event"] == "host.monitor.tick"]
    assert len(ticks) == 2
    for tick in ticks:
        assert tick["state"] == "step" and tick["step"] == 3
        assert tick["run_status"] == "RUNNING" and tick["attempts"] == 2
        assert "step=3" in tick["message"] and "attempts=2" in tick["message"]
    (end,) = [r for r in records if r["event"] == "host.monitor.end"]
    assert end["outcome"] == "budget_exhausted" and end["ticks"] == 2


def test_the_newest_attempt_is_by_start_time_not_by_the_order_of_the_answer() -> None:
    code, records = attempts_wave(  # oldest-first answer; start_time decides
        run_row("FAILED", ("phase_index", 8), start_time=1000),
        run_row("RUNNING", ("phase_index", 3), start_time=2000),
        ticks="1",
    )
    (tick,) = [r for r in records if r["event"] == "host.monitor.tick"]
    assert code == EXIT_EXHAUSTED
    assert tick["step"] == 3 and tick["run_status"] == "RUNNING"


def test_an_arm_is_terminal_only_when_every_attempt_is_and_reads_the_newest() -> None:
    code, records = attempts_wave(
        run_row("FINISHED", ("phase_index", 6), start_time=2000),
        run_row("FAILED", ("phase_index", 8), start_time=1000),
        ticks="3",
    )
    assert code == EXIT_OK
    (tick,) = [r for r in records if r["event"] == "host.monitor.tick"]
    assert tick["step"] == 6 and tick["run_status"] == "FINISHED" and tick["attempts"] == 2
    (end,) = [r for r in records if r["event"] == "host.monitor.end"]
    assert end["outcome"] == "all_runs_terminal" and end["ticks"] == 1


def test_a_live_older_attempt_keeps_the_arm_live_beside_a_newer_terminal_one() -> None:
    code, records = attempts_wave(
        run_row("FAILED", ("phase_index", 2), start_time=2000),
        run_row("RUNNING", ("phase_index", 5), start_time=1000),
        ticks="1",
    )
    (tick,) = [r for r in records if r["event"] == "host.monitor.tick"]
    assert code == EXIT_EXHAUSTED and tick["run_status"] == "RUNNING" and tick["attempts"] == 2


def test_a_single_run_reads_as_one_attempt_and_a_failed_read_carries_no_count() -> None:
    _code, records = attempts_wave(run_row("RUNNING", ("phase_index", 4)), ticks="1")
    (tick,) = [r for r in records if r["event"] == "host.monitor.tick"]
    assert tick["attempts"] == 1
    _code, failed = go(Source(RunReading("read_failed")), "--run-name", "h-worker-a", ticks="1")
    (bad,) = [r for r in failed if r["event"] == "host.monitor.tick"]
    assert "attempts" not in bad


@pytest.mark.parametrize(
    "answers,expected",
    [
        ([urllib.error.HTTPError(URI, 404, "nf", None, None)], "run_not_found"),  # type: ignore[arg-type]
        ([EXPERIMENT, {"runs": []}], "run_not_found"),
        ([urllib.error.HTTPError(URI, 500, "boom", None, None)], "list_failed"),  # type: ignore[arg-type]
        ([urllib.error.URLError("down")], "list_failed"),
        ([TimeoutError()], "list_failed"),
        ([b"not json"], "read_failed"),
        ([b"[1]"], "read_failed"),
        ([{"experiment": {}}], "read_failed"),
        ([EXPERIMENT, {"runs": [{"info": {}}]}], "read_failed"),
        ([EXPERIMENT, {"runs": [run_row("RUNNING", ("phase_index", 0))]}], "read_failed"),
        ([EXPERIMENT, {"runs": "nope"}], "read_failed"),
    ],
)
def test_each_failure_is_its_own_state(answers: list[object], expected: str) -> None:
    assert rest(Opener(*answers)).latest("h-series", "h-worker-abc").state == expected


def test_an_oversized_response_is_refused_not_buffered() -> None:
    huge = b"x" * (monitor._MAX_RESPONSE_BYTES + 10)  # pyright: ignore[reportPrivateUsage]
    assert rest(Opener(huge)).latest("h-series", "h-worker-abc").state == "read_failed"


@pytest.mark.parametrize("name", ["", "a b", "x'; DROP", "../x", "-lead", "a" * 300])
def test_a_run_name_that_could_break_the_filter_is_refused_before_any_request(name: str) -> None:
    opener = Opener(EXPERIMENT)
    with pytest.raises(ValueError):
        rest(opener).latest("h-series", name)
    assert not opener.requests


@pytest.mark.parametrize(
    "uri",
    [
        "ftp://x.example/",
        "https://user:pw@x.example/",  # pragma: allowlist secret
        "https://user@x.example/",
        "https://x.example/?a=b",
        "https://x.example/#frag",
        "not a uri",
        "",
    ],
)
def test_a_tracking_uri_with_credentials_or_a_foreign_scheme_is_refused_unread(uri: str) -> None:
    with pytest.raises(ValueError) as caught:
        MlflowRestSource(uri, username=None, password=None, timeout_seconds=1, opener=Opener())
    assert "pw" not in str(caught.value)


def test_credentials_travel_together_and_the_timeout_must_be_finite() -> None:
    with pytest.raises(ValueError):
        MlflowRestSource(URI, username="u", password=None, timeout_seconds=1, opener=Opener())
    for timeout in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            MlflowRestSource(
                URI, username=None, password=None, timeout_seconds=timeout, opener=Opener()
            )
    anonymous = Opener(EXPERIMENT, {"runs": []})
    MlflowRestSource(URI, username=None, password=None, timeout_seconds=1, opener=anonymous).latest(
        "h-series", "h-worker-a"
    )
    assert anonymous.requests[0][0].get_header("Authorization") is None


def test_the_default_source_reads_the_environment_at_call_time_and_never_renders_it() -> None:
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        monitor._default_source({}, 1.0)  # pyright: ignore[reportPrivateUsage]
    source = monitor._default_source(  # pyright: ignore[reportPrivateUsage]
        {
            "MLFLOW_TRACKING_URI": URI,
            "MLFLOW_TRACKING_USERNAME": SECRET_USER,
            "MLFLOW_TRACKING_PASSWORD": SECRET_PASSWORD,
        },
        2.0,
    )
    assert isinstance(source, MlflowRestSource)


def test_no_credential_or_uri_reaches_a_record_or_the_failure_report() -> None:
    opener = Opener(urllib.error.URLError(f"cannot reach {URI} as {SECRET_USER}:{SECRET_PASSWORD}"))
    with captured() as records:
        code = run_wave_monitor(
            argv("--run-name", "h-worker-a", ticks="1"),
            environ={
                "MLFLOW_TRACKING_URI": URI,
                "MLFLOW_TRACKING_USERNAME": SECRET_USER,
                "MLFLOW_TRACKING_PASSWORD": SECRET_PASSWORD,
            },
            source_factory=lambda env, timeout: MlflowRestSource(
                env["MLFLOW_TRACKING_URI"],
                username=env["MLFLOW_TRACKING_USERNAME"],
                password=env["MLFLOW_TRACKING_PASSWORD"],
                timeout_seconds=timeout,
                opener=opener,
            ),
            sleep=lambda _s: None,
        )
    blob = json.dumps(records)
    assert code == EXIT_EXHAUSTED
    assert [r["state"] for r in records if r.get("event") == "host.monitor.tick"] == ["list_failed"]
    for secret in (
        SECRET_PASSWORD,
        SECRET_USER,
        URI,
        "tracking.example.test",
        "Authorization",
        "Basic ",
    ):
        assert secret not in blob


# ------------------------------------------------ read-only, layer-clean, cold import
def test_the_monitor_defines_no_create_delete_or_terminate_path_and_never_prints() -> None:
    source = MODULE.read_text()
    tree = ast.parse(source)
    imported = {
        a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    imported |= {
        (n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    }
    assert not imported & {"subprocess", "mlflow", "runpod", "boto3", "requests", "shutil"}
    calls = {
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert "print" not in calls and "system" not in calls
    assert not calls & {"JsonlSink", "MlflowSink", "CompositeSink", "bind_report"}
    lowered = " ".join(
        name.lower()
        for name in (
            n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        )
    )
    for verb in ("delete", "terminate", "reap", "create", "launch", "kill"):
        assert verb not in lowered
    assert "report=True" not in source.replace(" ", "")
    assert "sys.stdout" not in source and "sys.stderr" not in source


def test_importing_the_monitor_runs_nothing_reads_no_environment_and_binds_no_mlflow() -> None:
    tree = ast.parse(MODULE.read_text())
    allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
    for node in tree.body:
        is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        is_guard = isinstance(node, ast.If) and "__main__" in ast.dump(node.test)
        assert isinstance(node, allowed) or is_docstring or is_guard, ast.dump(node)[:80]
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            # module constants may be built only by pure constructors (no I/O, no environment)
            for call in (n for n in ast.walk(node.value) if isinstance(n, ast.Call)):
                func = call.func
                assert (isinstance(func, ast.Name) and func.id == "frozenset") or (
                    isinstance(func, ast.Attribute) and func.attr == "compile"
                ), ast.dump(node)[:80]
    code = (
        "import sys\n"
        "import entropic_substrate.composition.entrypoints.h_wave_monitor as m\n"
        "assert 'mlflow' not in sys.modules, 'mlflow imported'\n"
        "print('cold-import-ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,  # HANG guard (measured: a few seconds for the torch-bearing import); never tuned
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert completed.returncode == 0 and "cold-import-ok" in completed.stdout, completed.stderr[
        -1500:
    ]


def test_the_monitor_has_a_main_that_configures_logging_first() -> None:
    tree = ast.parse(MODULE.read_text())
    (main,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    first = main.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert getattr(first.value.func, "id", None) == "configure_logging"
