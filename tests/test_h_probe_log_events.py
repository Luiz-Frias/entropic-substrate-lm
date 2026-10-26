"""FOR-1645/1646/1638: the H host narrates itself through the log substrate.

Four things are pinned here, all by observing what an operator can observe
(the substrate's records and a real process's stderr), never by reading module
state:

1. The host vocabulary is a closed, stdlib-only leaf, disjoint from the worker
   vocabulary, pinned in BOTH directions against the events the host code emits.
2. The checkpoint schedule arithmetic names exactly the ``checkpoint-N`` events a
   REAL worker run publishes, and the controller's progress cursor finds the newest
   one with a bounded number of reads that does not grow with the namespace.
3. The provider-aware controller (FOR-1638): ``get_owned`` returning ``None`` for the
   attributed pod is a terminal ``provider_absent`` observation, never shadowing a
   published terminal, with the attributed id immutable after allocation custody.
4. Each host entry point's REAL ``main()`` configures the substrate first, shown on a
   real child process's stderr (M1).

Child hermeticity (precedent ``tests/test_h_worker_process_logs.py``): ``sys.executable``,
a tmp cwd (``main()`` writes the cwd-relative default ``logs/entropic-substrate.jsonl``),
the repo root, ``src`` and ``tests`` on ``PYTHONPATH``, stdin closed, and an environment
with every MLflow and AWS variable removed so no gate env can reach a live service.
"""

from __future__ import annotations

import ast
import io
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_probe_controller import Guard, Provider, setup
from tests.test_h_probe_events import _run_and_fetch_checkpoint_key
from tests.test_h_probe_launch import launch
from tests.test_h_probe_operator_stop import _stop_argv, _synthetic_stopped_terminal

import entropic_substrate
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.run import checkpoint
from entropic_substrate.ladder.run.h_probe_process import ProbeWatchdogProcess
from entropic_substrate.ladder.run.h_runpod_transport import ProbeAccount, ProbeProviderError
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.transport.d1_trainer_s3_object_adapter import (
    D1TrainerS3ObjectError,
)
from entropic_substrate.ladder.run.worker import h_probe_controller as controller
from entropic_substrate.ladder.run.worker import h_probe_log_events as events
from entropic_substrate.ladder.run.worker.h_probe_controller import (
    ProbeControllerFailure,
    _CheckpointProgress,
    run_probe_controller,
)
from entropic_substrate.ladder.run.worker.h_probe_stop import run_probe_stop
from entropic_substrate.ladder.run.worker_log_events import H_WORKER_LOG_EVENTS

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
PKG = Path(entropic_substrate.__file__).resolve().parent
SRC = PKG.parent
HOST_MODULES = (
    PKG / "ladder/run/worker/h_probe_command.py",
    PKG / "ladder/run/worker/h_probe_controller.py",
    PKG / "ladder/run/worker/h_probe_stop.py",
    PKG / "composition/entrypoints/h_wave_monitor.py",
)

# MEASURED 2026-10-16 on the development Mac while a sibling suite was idle: each child
# below (interpreter + torch import + a fake-adapter command run) is a few seconds, and the
# tracer child additionally runs one tiny CPU training (about 20 s, like the worker child in
# tests/test_h_worker_process_logs.py). 300 s is a HANG guard, not a tuned budget; it is never
# raised to make a test pass (house rule 3).
_CHILD_TIMEOUT_S = 300


# ---------------------------------------------------------------- observation
@contextmanager
def captured() -> Iterator[list[dict[str, Any]]]:
    """Every substrate record emitted inside the block: level, message and bound extras."""
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


def names(records: list[dict[str, Any]]) -> list[str]:
    return [r["event"] for r in records if "event" in r]


def only(records: list[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    return [r for r in records if r.get("event") == event]


# ------------------------------------------------------- 1. the vocabulary leaf
def test_the_vocabulary_is_a_stdlib_only_leaf_outside_the_reader_identity() -> None:
    tree = ast.parse(Path(cast(str, events.__file__)).read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"__future__"}
    assert not any("h_probe_log_events" in m for m in checkpoint._READER_MODULES)  # pyright: ignore[reportPrivateUsage]


def test_the_host_vocabulary_is_closed_and_disjoint_from_the_worker_vocabulary() -> None:
    # RETRACTED 2026-10-17 (FOR-1665): 17 was the previous measured count; the host gained
    # host.terminal_probe.transient, and `len(events.H_HOST_LOG_EVENTS)` now measures 18.
    assert len(events.H_HOST_LOG_EVENTS) == 18
    assert events.HOST_TERMINAL_PROBE_TRANSIENT == "host.terminal_probe.transient"
    assert all(name.startswith("host.") for name in events.H_HOST_LOG_EVENTS)
    assert not events.H_HOST_LOG_EVENTS & H_WORKER_LOG_EVENTS
    constants = {
        getattr(events, n) for n in events.__all__ if n.startswith("HOST_") and n.isupper()
    }
    assert constants == events.H_HOST_LOG_EVENTS


def test_render_is_the_event_then_name_value_pairs() -> None:
    assert events.render("host.x") == "host.x"
    assert events.render("host.x", a=1, b="two", c=0.5) == "host.x a=1 b=two c=0.500"


def test_every_host_event_literal_in_the_host_modules_is_in_the_vocabulary() -> None:
    for path in HOST_MODULES:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("host.") and " " not in node.value:
                    assert node.value in events.H_HOST_LOG_EVENTS, (path.name, node.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": events.SOURCE_R2_CHECKPOINT, "state": events.STATE_STEP},  # step missing
        {"source": events.SOURCE_R2_CHECKPOINT, "state": events.STATE_NO_PROGRESS, "step": 3},
        {"source": events.SOURCE_R2_CHECKPOINT, "state": events.STATE_READ_FAILED, "step": 0},
        {"source": events.SOURCE_R2_CHECKPOINT, "state": events.STATE_STEP, "step": 0},
        {"source": events.SOURCE_R2_CHECKPOINT, "state": events.STATE_STEP, "step": True},
        {"source": "somewhere", "state": events.STATE_STEP, "step": 1},
        {"source": events.SOURCE_MLFLOW_METRIC, "state": "stalled"},
        {"source": events.SOURCE_MLFLOW_METRIC, "state": events.STATE_STEP, "step": 1, "total": 0},
    ],
)
def test_a_progress_line_refuses_an_undeclared_or_inconsistent_state(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        events.progress_fields(**kwargs)


@pytest.mark.parametrize(
    "state",
    [
        events.STATE_NO_PROGRESS,
        events.STATE_RUN_NOT_FOUND,
        events.STATE_READ_FAILED,
        events.STATE_LIST_FAILED,
    ],
)
@pytest.mark.parametrize("source", sorted(events.PROGRESS_SOURCES))
def test_a_non_step_state_never_renders_a_step_so_it_cannot_read_as_the_previous_one(
    source: str, state: str
) -> None:
    fields = events.progress_fields(source=source, state=state, total=6)
    assert "step" not in fields and fields["state"] == state and fields["source"] == source
    assert "step=" not in events.render(events.HOST_POLL_TICK, **fields)
    stepped = events.progress_fields(source=source, state=events.STATE_STEP, step=4, total=6)
    assert events.render(events.HOST_POLL_TICK, **stepped).endswith(
        "source=" + source + " state=step step=4 total=6"
    )


@pytest.mark.parametrize(
    "schedule,expected",
    [
        ((2, 4, 2, 2), (2, 4, 6)),
        ((5, 7, 3, 2), (3, 5, 7, 8, 9, 11, 12)),
        ((0, 3, 5, 5), (3,)),
        ((4, 0, 2, 2), (2, 4)),
        ((0, 0, 1, 1), ()),
    ],
)
def test_the_seal_schedule_arithmetic(
    schedule: tuple[int, int, int, int], expected: tuple[int, ...]
) -> None:
    warmup, steps, every, evaluate = schedule
    assert (
        events.checkpoint_steps(
            warmup_steps=warmup,
            steps=steps,
            checkpoint_interval=every,
            evaluation_interval=evaluate,
        )
        == expected
    )


@pytest.mark.parametrize(
    "bad", [(-1, 1, 1, 1), (1, -1, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0), (True, 1, 1, 1)]
)
def test_the_seal_schedule_refuses_nonsense_bounds(bad: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        events.checkpoint_steps(
            warmup_steps=bad[0],
            steps=bad[1],
            checkpoint_interval=bad[2],
            evaluation_interval=bad[3],
        )


# ----------------------------------- 2. the schedule vs a REAL worker run; reads
@pytest.fixture(scope="module")
def real_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[_ObjectStore, Any]:
    """The real worker pipeline, run ONCE for this module (the slow part)."""
    objects, request, _key = _run_and_fetch_checkpoint_key(tmp_path_factory.mktemp("host-events"))
    return objects, request


def _schedule(request: Any) -> tuple[int, ...]:
    s = request.profile.schedule
    return events.checkpoint_steps(
        warmup_steps=s.warmup_steps,
        steps=s.steps,
        checkpoint_interval=s.checkpoint_interval,
        evaluation_interval=s.evaluation_interval,
    )


def test_the_schedule_names_exactly_the_checkpoint_events_a_real_worker_publishes(
    real_run: tuple[_ObjectStore, Any],
) -> None:
    objects, request = real_run
    published = {
        key.rsplit("/", 1)[1].removesuffix(".json")
        for key in objects.values
        if key.startswith(f"{request.prefix}/checkpoint-")
    }
    assert published == {f"checkpoint-{n:08d}" for n in _schedule(request)}
    assert len(published) >= 2


class _Counting:
    """An object port that counts reads and refuses to list (progress never lists)."""

    def __init__(self, store: _ObjectStore, junk: int = 0) -> None:
        self.store, self.reads = store, 0
        for index in range(junk):
            store.values[f"unrelated/{index:06d}"] = b"x"

    def read(self, key: str) -> Any:
        self.reads += 1
        return self.store.read(key)

    def list(self, prefix: str) -> Any:
        raise AssertionError("progress must never list the namespace")


def _subset(objects: _ObjectStore, request: Any, upto: int) -> _ObjectStore:
    keep = {
        key: value
        for key, value in objects.values.items()
        if not key.startswith(f"{request.prefix}/checkpoint-")
        or int(key.rsplit("-", 1)[1].removesuffix(".json")) <= upto
    }
    return _ObjectStore(values=keep)


def test_the_progress_cursor_finds_the_newest_real_checkpoint_by_parsed_step(
    real_run: tuple[_ObjectStore, Any],
) -> None:
    objects, request = real_run
    schedule = _schedule(request)
    assert (
        _CheckpointProgress(request).advance(cast(Any, _Counting(_subset(objects, request, 0))))
        is None
    )
    for upto in schedule:
        port = _Counting(_subset(objects, request, upto))
        assert _CheckpointProgress(request).advance(cast(Any, port)) == upto
    assert (
        _CheckpointProgress(request).total
        == request.profile.schedule.warmup_steps + request.profile.schedule.steps
    )


def test_reads_per_poll_are_bounded_and_do_not_grow_with_the_namespace(
    real_run: tuple[_ObjectStore, Any],
) -> None:
    """G2: a counting port whose namespace grows across polls shows no growth in reads; the
    port refuses ``list``, so a per-poll listing would fail this test outright."""
    objects, request = real_run
    schedule = _schedule(request)
    bound = 1 + math.ceil(math.log2(len(schedule)))
    cursor = _CheckpointProgress(request)
    per_poll: list[int] = []
    for upto, junk in zip((0, schedule[0], schedule[-1]), (0, 500, 5000), strict=True):
        port = _Counting(_subset(objects, request, upto), junk=junk)
        cursor.advance(cast(Any, port))
        per_poll.append(port.reads)
    assert max(per_poll) <= bound
    again = _Counting(_subset(objects, request, schedule[-1]), junk=50_000)
    assert _CheckpointProgress(request).advance(cast(Any, again)) == schedule[-1]
    assert again.reads <= bound


def test_a_long_schedule_costs_logarithmic_reads_and_a_quiet_poll_costs_one() -> None:
    request = SimpleNamespace(
        profile=SimpleNamespace(
            schedule=SimpleNamespace(
                warmup_steps=0, steps=6000, checkpoint_interval=10, evaluation_interval=10
            )
        ),
        limits=SimpleNamespace(max_object_bytes=1000),
        prefix="p",
        resume=None,
    )
    present: set[str] = set()
    reads: list[str] = []

    def fake(_objects: object, _request: object, name: str, **_kw: object) -> object | None:
        reads.append(name)
        return object() if name in present else None

    cursor = _CheckpointProgress(cast(Any, request))
    candidates = events.checkpoint_steps(
        warmup_steps=0, steps=6000, checkpoint_interval=10, evaluation_interval=10
    )
    assert len(candidates) == 600
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(controller, "read_probe_event", fake)
        assert cursor.advance(cast(Any, object())) is None and len(reads) == 1  # quiet poll
        for count in (1, 37, 38, 599, 600):
            present.update(f"checkpoint-{n:08d}" for n in candidates[:count])
            reads.clear()
            assert cursor.advance(cast(Any, object())) == candidates[count - 1]
            assert len(reads) <= 1 + math.ceil(math.log2(len(candidates)))
        reads.clear()
        assert cursor.advance(cast(Any, object())) == 6000 and not reads  # nothing left to probe


def _resumed_request(resume_name: str | None, *, warmup: int = 250) -> Any:
    return SimpleNamespace(
        profile=SimpleNamespace(
            schedule=SimpleNamespace(
                warmup_steps=warmup, steps=1000, checkpoint_interval=250, evaluation_interval=250
            )
        ),
        limits=SimpleNamespace(max_object_bytes=1000),
        prefix="p",
        resume=None if resume_name is None else SimpleNamespace(path=f"run/{resume_name}"),
    )


def _probe_log(present: set[str]) -> tuple[list[str], Any]:
    reads: list[str] = []

    def fake(_objects: object, _request: object, name: str, **_kw: object) -> object | None:
        reads.append(name)
        return object() if name in present else None

    return reads, fake


def test_a_resumed_request_reports_progress_from_its_first_seal_after_the_stop() -> None:
    """2026-10-26 arm B: stopped at global 500 (prediction step 250 + warmup 250), the
    resumed prefix starts at checkpoint-750, so candidate 250 is never present."""
    request = _resumed_request("h0-prediction-00000250-progress.json")
    reads, fake = _probe_log({"checkpoint-00000750"})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(controller, "read_probe_event", fake)
        assert _CheckpointProgress(request).advance(cast(Any, object())) == 750
    assert all(int(name.rsplit("-", 1)[1]) > 500 for name in reads)


def test_a_resumed_request_with_nothing_published_probes_only_above_the_stop() -> None:
    request = _resumed_request("h0-prediction-00000250-progress.json")
    reads, fake = _probe_log(set())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(controller, "read_probe_event", fake)
        assert _CheckpointProgress(request).advance(cast(Any, object())) is None
    assert reads == ["checkpoint-00000750"]


def test_a_warmup_stop_maps_to_the_global_axis_without_the_warmup_offset() -> None:
    request = _resumed_request("h0-warmup-00000250-progress.json")
    reads, fake = _probe_log({"checkpoint-00000500"})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(controller, "read_probe_event", fake)
        assert _CheckpointProgress(request).advance(cast(Any, object())) == 500
    assert reads[0] == "checkpoint-00000500"


def test_a_request_without_resume_still_probes_from_the_first_candidate() -> None:
    request = _resumed_request(None)
    reads, fake = _probe_log(set())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(controller, "read_probe_event", fake)
        assert _CheckpointProgress(request).advance(cast(Any, object())) is None
    assert reads == ["checkpoint-00000250"]


@pytest.mark.parametrize(
    "name",
    [
        "h0-prediction-00000200-progress.json",  # global 450: not on the seal ladder
        "h0-prediction-00000250-receipt.json",  # not a progress receipt name
        "h0-cooldown-00000250-progress.json",  # unknown phase
    ],
)
def test_a_resume_whose_stop_is_not_on_the_seal_ladder_is_refused(name: str) -> None:
    request = _resumed_request(name)
    with pytest.raises(ValueError, match="resume"):
        _CheckpointProgress(request).advance(cast(Any, object()))


# --------------------------------------------- 3. the provider-aware controller
def _drive(
    guard: Guard,
    provider: Provider,
    clock: list[float],
    destination: Path,
    objects: object | None = None,
) -> dict[str, object]:
    return run_probe_controller(
        cast(ProbeWatchdogProcess, guard),
        provider,
        cast(D1TrainerMailboxObjectPort, object() if objects is None else objects),
        destination,
        poll_seconds=1,
        max_hourly_rate=1,
        monotonic=lambda: clock[0],
        wait=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )


TERMINAL: dict[str, object] = {
    "reported_outcome": "probe_workload_complete",
    "workload_verified": True,
    "capture_status": "native_completion_receipt_verified",
}


def _collect_sequence(monkeypatch: pytest.MonkeyPatch, outcomes: list[object]) -> list[int]:
    """Replace terminal capture with a scripted sequence; returns the call counter."""
    calls = [0]

    def collect(*_args: object, **_kwargs: object) -> object:
        index = min(calls[0], len(outcomes) - 1)
        calls[0] += 1
        outcome = outcomes[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(controller, "collect_probe_terminal", collect)
    return calls


def test_an_absent_attributed_pod_ends_the_loop_before_the_deadline_as_a_named_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard, provider, calls, clock, _run = setup(tmp_path, monkeypatch, None)
    provider.present = False
    with captured() as records, pytest.raises(ProbeControllerFailure) as caught:
        _drive(guard, provider, clock, tmp_path / "capture")
    record = caught.value.record
    assert clock[0] == 0, "the loop must end on the first absent observation, not wait for 3.0"
    assert calls == ["create", "collect"], "terminal capture is attempted AFTER the None"
    assert record["failure_stages"] == ["provider_absent"]
    assert record["error_types"] == ["ProbeProviderAbsent"]
    assert record["status"] == "controller_failed" and record["workload_verified"] is False
    assert record["pod_id"] == "owned-pod" and record["termination_verified"] is True
    assert guard.control.records["stop"]["reason"] == "cancelled"
    assert guard.control.records["stop"]["pod_id"] == "owned-pod"
    assert only(records, events.HOST_PROVIDER_ABSENT_OBSERVED)[0]["pod_id"] == "owned-pod"


def test_an_absent_pod_never_shadows_a_published_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard, provider, _calls, clock, _run = setup(tmp_path, monkeypatch, TERMINAL)
    provider.present = False
    with captured() as records:
        result = _drive(guard, provider, clock, tmp_path / "capture")
    assert result["status"] == "terminal_collected" and result["workload_verified"] is True
    assert result["failure_stages"] == []
    assert not only(records, events.HOST_PROVIDER_ABSENT_OBSERVED)
    assert (
        only(records, events.HOST_TERMINAL_OBSERVED)[0]["reported_outcome"]
        == "probe_workload_complete"
    )


def test_an_absence_on_a_different_account_is_not_provider_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard, provider, _calls, clock, _run = setup(tmp_path, monkeypatch, None)
    provider.present = False
    monkeypatch.setattr(provider, "account", lambda: ProbeAccount("someone-else", 20, 0))
    with captured() as records, pytest.raises(ProbeControllerFailure) as caught:
        _drive(guard, provider, clock, tmp_path / "capture")
    assert caught.value.record["failure_stages"] == ["provider_ownership"]
    assert not only(records, events.HOST_PROVIDER_ABSENT_OBSERVED)
    assert caught.value.record["pod_id"] == "owned-pod"


def test_a_present_pod_is_polled_to_the_deadline_exactly_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard, provider, _calls, clock, _run = setup(tmp_path, monkeypatch, None)
    with pytest.raises(ProbeControllerFailure) as caught:
        _drive(guard, provider, clock, tmp_path / "capture")
    assert clock[0] == 3 and caught.value.record["failure_stages"] == ["terminal_capture"]


@pytest.mark.parametrize("code", sorted(controller._TRANSIENT_PROVIDER_CODES))  # pyright: ignore[reportPrivateUsage]
def test_a_transient_provider_code_is_logged_and_the_poll_continues_at_the_same_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    guard, provider, _calls, clock, _run = setup(tmp_path, monkeypatch, None)
    _collect_sequence(monkeypatch, [None, TERMINAL])
    attempts = [0]
    original = provider.get_owned

    def flaky(spec: object, pod_id: str) -> Any:
        attempts[0] += 1
        if attempts[0] == 1:
            raise ProbeProviderError(code)
        return original(spec, pod_id)

    monkeypatch.setattr(provider, "get_owned", flaky)
    with captured() as records:
        result = _drive(guard, provider, clock, tmp_path / "capture")
    assert result["status"] == "terminal_collected" and attempts[0] == 2
    assert clock[0] == 1, "one poll interval passed, no extra sleep or counter"
    assert only(records, events.HOST_PROVIDER_TRANSIENT)[0]["code"] == code


def test_the_transient_set_equals_the_watchdogs_tolerated_set() -> None:
    """The controller duplicates h_probe_watchdog.py:191-196 (outside this phase's envelope)."""
    source = (PKG / "ladder/run/h_probe_watchdog.py").read_text()
    window = source[
        source.index("def note_error") : source.index(
            "writer.write(", source.index("def note_error")
        )
    ]
    tolerated = {
        node.value
        for node in ast.walk(
            ast.parse("if 1:\n" + "\n".join("    " + line for line in window.splitlines()[1:]))
        )
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert set(controller._TRANSIENT_PROVIDER_CODES) <= tolerated  # pyright: ignore[reportPrivateUsage]
    assert tolerated >= {"network_error", "timeout", "rate_limited", "server_error"}


@pytest.mark.parametrize("possible", [None, "someone-elses-pod"])
def test_a_nontransient_probe_error_cannot_rebind_the_attributed_pod_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, possible: str | None
) -> None:
    """C5: the id is immutable after allocation custody. Before the guard, an error carrying
    possible_pod_id=None rebound pod_id to None and the stop handoff then disagreed with the
    allocation record."""
    guard, provider, _calls, clock, _run = setup(tmp_path, monkeypatch, None)

    def refuse(_spec: object, _pod_id: str) -> Any:
        raise ProbeProviderError("unauthorized", possible_pod_id=possible)

    monkeypatch.setattr(provider, "get_owned", refuse)
    with pytest.raises(ProbeControllerFailure) as caught:
        _drive(guard, provider, clock, tmp_path / "capture")
    record = caught.value.record
    assert record["pod_id"] == "owned-pod" and record["provider_error_code"] == "unauthorized"
    assert record["failure_stages"][0] == "provider_ownership"
    assert guard.control.records["stop"]["pod_id"] == "owned-pod"
    assert guard.control.records["stop"]["reason"] == "cancelled"


def test_the_result_record_keeps_its_keys_and_status_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard, provider, _calls, clock, _run = setup(tmp_path, monkeypatch, TERMINAL)
    keys = set(_drive(guard, provider, clock, tmp_path / "capture"))
    assert keys == {
        "schema", "scientific_authority", "request_sha256", "launch_sha256", "pod_id", "pod",
        "status", "workload_verified", "termination_verified", "collection", "watchdog",
        "emergency_cleanup", "error_types", "provider_error_code", "failure_stages",
        # ADDITIVE 2026-10-17 (FOR-1665): every key above is kept, these two are new.
        "terminal_probe_transient_count", "last_terminal_probe_error",
    }  # fmt: skip


# ------------------------------------------------------- controller progress line
def _presence(monkeypatch: pytest.MonkeyPatch, present: set[str]) -> None:
    def fake(_objects: object, request: Any, name: str, **_kw: object) -> object | None:
        return SimpleNamespace(name=name) if name in present else None

    monkeypatch.setattr(controller, "read_probe_event", fake)


def test_one_info_progress_line_per_poll_carries_the_newest_step_and_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard, provider, _calls, clock, _run = setup(tmp_path, monkeypatch, None)
    guard.ready = {"account_id": "fixture", "controller_deadline": 10.0, "watchdog_deadline": 20.0}
    _collect_sequence(monkeypatch, [None, None, None, TERMINAL])
    polls = [0]
    sealed = {"checkpoint-00000002", "checkpoint-00000004", "checkpoint-00000006"}
    # The fixture schedule seals at 2, 4 and 6: nothing after poll 1, the first after poll 2,
    # all of them after poll 3.
    visible = {1: set[str](), 2: {"checkpoint-00000002"}, 3: sealed}

    def read(_objects: object, _request: object, name: str, **_kw: object) -> object | None:
        return SimpleNamespace() if name in visible[polls[0]] else None

    monkeypatch.setattr(controller, "read_probe_event", read)
    original = controller._progress_tick  # pyright: ignore[reportPrivateUsage]

    def tick(progress: Any, objects: Any, pod_name: str) -> None:
        polls[0] += 1
        original(progress, objects, pod_name)

    monkeypatch.setattr(controller, "_progress_tick", tick)
    with captured() as records:
        _drive(guard, provider, clock, tmp_path / "capture")
    ticks = only(records, events.HOST_POLL_TICK)
    assert [(t["state"], t.get("step")) for t in ticks] == [
        ("no_progress_yet", None),
        ("step", 2),
        ("step", 6),
    ]
    assert all(t["source"] == events.SOURCE_R2_CHECKPOINT and t["total"] == 6 for t in ticks)
    assert ticks[2]["last_event"] == "checkpoint-00000006"
    assert "step=6" in ticks[2]["message"] and "step=" not in ticks[0]["message"]
    assert all(t["level"] == "INFO" for t in ticks)


def test_a_progress_failure_is_its_own_warning_and_changes_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C8: the ``object()`` port has no ``read``; the poll warns each time and the run is
    exactly what it was without progress."""
    guard, provider, _calls, clock, _run = setup(tmp_path, monkeypatch, None)
    with captured() as records, pytest.raises(ProbeControllerFailure) as caught:
        _drive(guard, provider, clock, tmp_path / "capture")
    warnings = only(records, events.HOST_PROGRESS_UNAVAILABLE)
    assert len(warnings) == 3 and clock[0] == 3
    assert all(w["level"] == "WARNING" and w["state"] == "read_failed" for w in warnings)
    assert all(w["error_type"] == "AttributeError" and "step" not in w for w in warnings)
    assert not only(records, events.HOST_POLL_TICK)
    assert caught.value.record["failure_stages"] == ["terminal_capture"]


def test_the_progress_line_is_built_by_the_one_shared_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M6 teeth: a builder that dropped the step would fail the step assertions above; here the
    builder itself is observed, and the controller must route through it."""
    guard, provider, _calls, clock, _run = setup(tmp_path, monkeypatch, None)
    _collect_sequence(monkeypatch, [None, TERMINAL])
    _presence(monkeypatch, {"checkpoint-00000002"})
    seen: list[dict[str, object]] = []
    real = events.progress_fields

    def spy(**kwargs: Any) -> dict[str, object]:
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(controller, "progress_fields", spy)
    _drive(guard, provider, clock, tmp_path / "capture")
    assert seen and seen[0]["source"] == events.SOURCE_R2_CHECKPOINT and seen[0]["step"] == 2


# ------------------------------- narration: attribution, discipline, both ways
def _scenario_controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Happy path with a progress tick and a transient code; an absent pod; a blind port; a lost
    guardian with an emergency cleanup: together every controller-side event."""
    out: list[dict[str, Any]] = []
    with pytest.MonkeyPatch.context() as patch:
        (tmp_path / "ok").mkdir(parents=True)
        guard, provider, _c, clock, _r = setup(tmp_path / "ok", patch, None)
        # FOR-1665: the first tick's marker read raises (the terminal-probe transient event)
        _collect_sequence(
            patch, [D1TrainerS3ObjectError(operation="read", code="provider_failure"), TERMINAL]
        )
        _presence(patch, {"checkpoint-00000002"})
        tries = [0]
        base = provider.get_owned

        def flaky(spec: object, pod_id: str) -> Any:
            tries[0] += 1
            if tries[0] == 1:
                raise ProbeProviderError("timeout")
            return base(spec, pod_id)

        patch.setattr(provider, "get_owned", flaky)
        with captured() as records:
            _drive(guard, provider, clock, tmp_path / "ok" / "capture")
        out += records
    with pytest.MonkeyPatch.context() as patch:
        (tmp_path / "absent").mkdir(parents=True)
        guard, provider, _c, clock, _r = setup(tmp_path / "absent", patch, None)
        provider.present = False
        with captured() as records, pytest.raises(ProbeControllerFailure):
            _drive(guard, provider, clock, tmp_path / "absent" / "capture")
        out += records
    with pytest.MonkeyPatch.context() as patch:
        # an object port without read(): progress is advisory, so it warns and the run times out
        (tmp_path / "blind").mkdir(parents=True)
        guard, provider, _c, clock, _r = setup(tmp_path / "blind", patch, None)
        with captured() as records, pytest.raises(ProbeControllerFailure):
            _drive(guard, provider, clock, tmp_path / "blind" / "capture")
        out += records
    with pytest.MonkeyPatch.context() as patch:
        (tmp_path / "dead").mkdir(parents=True)
        guard, provider, _c, clock, _r = setup(tmp_path / "dead", patch, None)

        def lost(*_a: object, **_k: object) -> None:
            guard.alive = False
            raise ValueError("guardian died after allocation")

        patch.setattr(controller, "collect_probe_terminal", lost)
        patch.setattr(guard, "wait_result", lambda **_k: (_ for _ in ()).throw(ValueError("gone")))
        with captured() as records, pytest.raises(ProbeControllerFailure):
            _drive(guard, provider, clock, tmp_path / "dead" / "capture")
        out += records
    return out


def _scenario_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    request = launch().request
    confirmed = _ObjectStore()
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir(parents=True)
    _synthetic_stopped_terminal(confirmed, request, fixture_root, steps=4, pod_id="owned-pod")
    for store, wait in ((confirmed, "5"), (_ObjectStore(), "0.05")):
        work = tmp_path / f"stop-{wait}"
        work.mkdir()
        monkeypatch.setattr(
            "entropic_substrate.ladder.run.worker.h_probe_stop.build_worker_objects",
            lambda *_a, _s=store, **_k: _s,
        )
        with captured() as records:
            run_probe_stop(
                _stop_argv(work, wait_seconds=wait, poll_seconds="0.01"),
                environ={},
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        out += records
    return out


def _scenario_monitor() -> list[dict[str, Any]]:
    from entropic_substrate.composition.entrypoints import h_wave_monitor as monitor

    class Source:
        def latest(self, experiment: str, run_name: str) -> monitor.RunReading:
            return monitor.RunReading("step", 3, "FINISHED")

    with captured() as records:
        monitor.run_wave_monitor(
            ["--run-name", "h-worker-a", "--interval-seconds", "1",
             "--request-timeout-seconds", "1", "--ticks", "2"],
            environ={},
            source_factory=lambda _env, _timeout: Source(),
            sleep=lambda _s: None,
        )  # fmt: skip
    return records


def test_the_emitted_events_equal_the_vocabulary_in_both_directions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        *_scenario_controller(tmp_path / "c", monkeypatch),
        *_scenario_stop(tmp_path / "s", monkeypatch),
        *_scenario_monitor(),
    ]
    observed = set(names(records))
    # The stop verb's marker publish also emits the WORKER vocabulary (worker_transport); nothing
    # may fall outside the two closed sets, and the host set is matched exactly both ways.
    assert observed <= events.H_HOST_LOG_EVENTS | H_WORKER_LOG_EVENTS
    host = {name for name in observed if name.startswith("host.")}
    assert host <= events.H_HOST_LOG_EVENTS, host - events.H_HOST_LOG_EVENTS
    assert host >= events.H_HOST_LOG_EVENTS, events.H_HOST_LOG_EVENTS - host


def test_every_host_record_is_attributed_and_carries_only_safe_field_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        r
        for r in [
            *_scenario_controller(tmp_path / "c", monkeypatch),
            *_scenario_stop(tmp_path / "s", monkeypatch),
        ]
        if r.get("event") in events.H_HOST_LOG_EVENTS
    ]
    assert records
    pod_name = launch().pod_name
    for record in records:
        assert record["pod_name"] == pod_name, record["event"]
        assert record.get("report") is not True
        assert record["message"].startswith(record["event"] + " ")
        for key, value in record.items():
            assert type(value) in (str, int, float, bool, type(None)), (record["event"], key)
        error_type = record.get("error_type")
        assert error_type is None or error_type.isidentifier()
        blob = json.dumps(record)
        for forbidden in ("Traceback", "RUNPOD_SECRET", "MLFLOW", "AWS_", "guardian died"):
            assert forbidden not in blob, (record["event"], forbidden)


# ------------------------------------------------ 4. REAL main() on a real stderr
def _child_env() -> dict[str, str]:
    scrubbed = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MLFLOW_", "ENTROPIC_MLFLOW", "AWS_"))
    }
    scrubbed["PYTHONPATH"] = os.pathsep.join([str(REPO), str(SRC), str(REPO / "tests")])
    scrubbed["PYTHONUNBUFFERED"] = "1"
    return scrubbed


def _run_child(script: str, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    cwd.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=cwd,
        env=_child_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
        check=False,
    )


def _third_fields(stderr: str) -> list[str]:
    """``<local ts> | LEVEL | <message>``: the substrate's format; the message leads with the event."""
    return [line.split(" | ", 2)[2] for line in stderr.splitlines() if line.count(" | ") >= 2]


# The child patches ONLY I/O adapters, then runs the module's REAL main():
#   HRunpodTransport (provider transport), build_worker_objects (the R2 client builder),
#   supervise_probe_watchdog (the watchdog subprocess supervisor), allocate_probe_once (provider
#   create), collect_probe_terminal (R2 terminal reads) and read_probe_control (control-dir reads).
# run_probe_command and run_probe_controller run for real. The fake Guard/Provider are this
# suite's own (tests/test_h_probe_controller.py).
_COMMAND_CHILD = """
import sys, time
from contextlib import contextmanager
from pathlib import Path

import pytest
from tests.test_h_probe_controller import Guard, Provider

from entropic_substrate.ladder.run.worker import h_probe_command as command
from entropic_substrate.ladder.run.worker import h_probe_controller as ctl

mode, root = sys.argv[1], Path(sys.argv[2])
guard = Guard(root)
now = time.monotonic()
guard.ready = {"account_id": "fixture", "controller_deadline": now + 1000, "watchdog_deadline": now + 2000}
provider = Provider(guard)
if mode == "absent":
    provider.present = False


def allocate(*_a):
    guard.control.records["create-intent"] = {"guard_pid": guard.pid}
    guard.control.records["allocation"] = {"pod_id": provider.pod.pod_id}
    return provider.pod


def collect(*_a, **_k):
    if mode == "ok":
        return {"reported_outcome": "probe_workload_complete", "workload_verified": True,
                "capture_status": "native_completion_receipt_verified"}
    if mode == "fail":
        raise ValueError("private detail that must not reach the log")
    return None


@contextmanager
def supervise(*_a, **_k):
    yield guard


ctl.allocate_probe_once = allocate
ctl.collect_probe_terminal = collect
ctl.read_probe_control = lambda _r, _s, _role, name, **_kw: guard.control.records.get(name)
command.HRunpodTransport = lambda **_: provider
command.build_worker_objects = lambda *_a, **_k: object()
command.supervise_probe_watchdog = supervise
raise SystemExit(command.main(sys.argv[3:]))
"""


def _command_argv(work: Path) -> list[str]:
    work.mkdir(parents=True, exist_ok=True)
    (work / "launch.json").write_bytes(launch().payload)
    (work / "workspace").mkdir(mode=0o700)
    return [
        "--launch-root", str(work), "--launch-file", "launch.json",
        "--workspace", str(work / "workspace"), "--repo-root", str(work),
        "--expected-account-id", "fixture", "--runpodctl", "/fixture/runpodctl",
        "--provider-timeout", "30", "--max-provider-bytes", "1000000",
        "--max-launch-bytes", "1000000", "--max-control-bytes", "1000000",
        "--poll-seconds", "5", "--startup-timeout", "120", "--shutdown-timeout", "30",
        "--max-hourly-rate", "0.6",
    ]  # fmt: skip


@pytest.mark.parametrize("mode", ["ok", "fail", "absent"])
def test_the_real_command_main_puts_the_host_events_on_the_substrate_stderr(
    tmp_path: Path, mode: str
) -> None:
    work = tmp_path / "work"
    argv = _command_argv(work)
    completed = _run_child(_COMMAND_CHILD, tmp_path / "cwd", mode, str(tmp_path / "guard"), *argv)
    third = _third_fields(completed.stderr)
    host = [t for t in third if t.startswith("host.")]
    assert host, completed.stderr[-2000:]
    assert host[0].startswith("host.allocation.start pod_name=" + launch().pod_name)
    assert all(t.split(" ", 1)[0] in events.H_HOST_LOG_EVENTS for t in host)
    # the structured sink sits beside the child's cwd, with the bound fields as flat keys
    rows = [
        json.loads(line)
        for line in (tmp_path / "cwd/logs/entropic-substrate.jsonl").read_text().splitlines()
    ]
    assert {r["event"] for r in rows if "event" in r} == {t.split(" ", 1)[0] for t in host}
    assert all(r["pod_name"] == launch().pod_name for r in rows if "event" in r)
    lines = [line for line in completed.stderr.splitlines() if line.strip()]
    if mode == "ok":
        assert completed.returncode == 0
        (result,) = [line for line in completed.stdout.splitlines() if line.strip()]
        assert json.loads(result)["status"] == "terminal_collected"
        assert "host.terminal.observed" in " ".join(host)
    else:
        assert completed.returncode == 1 and completed.stdout == ""
        # G10: the machine-readable failure JSON is the LAST stderr line; nothing follows it.
        failure = json.loads(lines[-1])
        assert failure["status"] == "controller_command_failed"
        assert "private detail" not in completed.stderr
        if mode == "absent":
            assert "host.provider_absent.observed" in " ".join(host)


_STOP_CHILD = """
import sys
from pathlib import Path

from tests.test_d1_trainer_object_transport import _ObjectStore

from entropic_substrate.ladder.run.worker import h_probe_stop as stop

stop.build_worker_objects = lambda *_a, **_k: _ObjectStore()
raise SystemExit(stop.main(sys.argv[1:]))
"""


def test_the_real_stop_main_puts_the_host_events_on_the_substrate_stderr(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    argv = _stop_argv(tmp_path / "work", wait_seconds="0.05", poll_seconds="0.01")
    completed = _run_child(_STOP_CHILD, tmp_path / "cwd", *argv)
    host = [t for t in _third_fields(completed.stderr) if t.startswith("host.")]
    assert [t.split(" ", 1)[0] for t in host] == [
        "host.stop_marker.requested",
        "host.stop_marker.unconfirmed",
    ], completed.stderr[-2000:]
    assert completed.returncode == 1 and completed.stdout == ""
    last = [line for line in completed.stderr.splitlines() if line.strip()][-1]
    assert json.loads(last) == {"status": "stop_not_confirmed", "scientific_authority": "none"}


_MONITOR_CHILD = """
import sys
from entropic_substrate.composition.entrypoints import h_wave_monitor as monitor
raise SystemExit(monitor.main(sys.argv[1:]))
"""


def test_the_real_monitor_main_puts_its_events_on_the_substrate_stderr(tmp_path: Path) -> None:
    completed = _run_child(
        _MONITOR_CHILD, tmp_path / "cwd",
        "--run-name", "h-worker-a", "--interval-seconds", "1",
        "--request-timeout-seconds", "1", "--ticks", "1",
    )  # fmt: skip
    host = [t for t in _third_fields(completed.stderr) if t.startswith("host.")]
    assert host and host[0].startswith("host.monitor.end outcome=failed error_type=ValueError")
    assert completed.returncode == 1 and completed.stdout == ""


def test_the_real_tracer_main_puts_its_line_on_the_substrate_stderr(tmp_path: Path) -> None:
    from tests.test_h_tiny_tracer import _foreign_capture, _remote
    from tests.test_sentence_profile import profile

    foreign, _identity = _foreign_capture(tmp_path)
    remote = tmp_path / "remote-profile.json"
    remote.write_text(json.dumps(_remote(profile()).manifest))
    child = (
        "import sys\n"
        "from entropic_substrate.composition.wiring import h_tiny_tracer as tracer\n"
        "raise SystemExit(tracer.main(sys.argv[1:]))\n"
    )
    completed = _run_child(
        child, tmp_path / "cwd",
        "--profile", str(remote), "--corpus", str(foreign), "--out", str(tmp_path / "trace"),
        "--precision", "fp32",
    )  # fmt: skip
    assert completed.returncode == 0, completed.stderr[-2000:]
    lines = _third_fields(completed.stderr)
    assert any(t.startswith("h-tiny-trace training_complete") for t in lines), completed.stderr[
        -2000:
    ]


def test_the_tracer_configures_logging_in_main_and_not_again_under_the_module_guard() -> None:
    tree = ast.parse((PKG / "composition/wiring/h_tiny_tracer.py").read_text())
    guard = [n for n in tree.body if isinstance(n, ast.If)]
    assert guard and "configure_logging" not in ast.dump(guard[-1])
    (main,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    first = main.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert getattr(first.value.func, "id", None) == "configure_logging"


def test_the_stop_main_configures_the_substrate_before_it_runs_the_verb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from entropic_substrate.ladder.run.worker import h_probe_stop

    order: list[object] = []
    monkeypatch.setattr(h_probe_stop, "configure_logging", lambda: order.append("logging"))

    def run(argv: object, *, environ: object) -> int:
        order.append(("run", list(cast(Any, argv))))
        return 5

    monkeypatch.setattr(h_probe_stop, "run_probe_stop", run)
    assert h_probe_stop.main(["--x", "1"]) == 5
    assert order == ["logging", ("run", ["--x", "1"])]
