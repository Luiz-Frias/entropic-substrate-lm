"""Host suite lock: one xdist suite per host, the colliding run refuses with its holder."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import suite_lock
from suite_lock import (
    EXIT_LOCK_HELD,
    LOCK_PATH,
    Process,
    SuiteLockRefused,
    acquire,
    ancestor_pids,
    parse_holder,
    process_table,
    release,
)

pytestmark = pytest.mark.unit

STARTED = datetime(2026, 10, 30, 16, 53, 33, tzinfo=UTC)
NOW = datetime(2026, 10, 30, 16, 56, 52, tzinfo=UTC)
EARLIER = STARTED - timedelta(minutes=5)
ME, PARENT, HOLDER, ORPHAN = 5000, 4999, 7000, 7100


def proc(pid: int, ppid: int = 1, *, started: datetime = EARLIER, command: str = "zsh") -> Process:
    return Process(pid=pid, ppid=ppid, started=started, command=command)


# This run: a pytest controller under a shell; nothing else on the host.
ALONE = (proc(PARENT), proc(ME, PARENT, command="python -m pytest -n auto"))


def hold(lock: Path, label: str, pid: int, started: datetime = STARTED) -> None:
    lock.mkdir()
    (lock / "holder").write_text(f"{label} {pid} {started:%Y-%m-%dT%H:%M:%SZ}\n")


def take(lock: Path, processes: tuple[Process, ...]) -> suite_lock.Acquisition:
    return acquire(lock, label="pytest", pid=ME, now=NOW, processes=processes)


def test_lock_path_is_the_one_existing_wrappers_take() -> None:
    # Workflow briefs and drivers already serialize with `mkdir /tmp/esl-suite.lock`.
    assert Path("/tmp/esl-suite.lock") == LOCK_PATH
    assert EXIT_LOCK_HELD == 75  # sysexits EX_TEMPFAIL: try again later


def test_holder_line_written_by_existing_wrappers_parses() -> None:
    holder = parse_holder("restack-gates 39025 2026-10-30T16:53:33Z\n")
    assert (holder.label, holder.pid, holder.started) == ("restack-gates", 39025, STARTED)
    assert parse_holder("w2 lane p8 4242 2026-10-30T16:53:33Z").label == "w2 lane p8"


@pytest.mark.parametrize("text", ["", "39025", "x notapid 2026-10-30T16:53:33Z", "x 1 yesterday"])
def test_malformed_holder_line_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        parse_holder(text)


def test_free_lock_is_acquired_with_a_wrapper_compatible_holder(tmp_path: Path) -> None:
    lock = tmp_path / "suite.lock"
    assert take(lock, ALONE) == suite_lock.Acquisition(held=True)
    holder = parse_holder((lock / "holder").read_text())
    assert (holder.label, holder.pid, holder.started) == ("pytest", ME, NOW)


def test_live_foreign_holder_refuses_the_new_run_with_its_start_time(tmp_path: Path) -> None:
    lock = tmp_path / "suite.lock"
    hold(lock, "restack-gates", HOLDER)
    with pytest.raises(SuiteLockRefused) as refused:
        take(lock, (*ALONE, proc(HOLDER)))
    message = str(refused.value)
    assert f"restack-gates (pid {HOLDER})" in message
    assert "12:53:33 EDT" in message  # Eastern wall clock, DST-aware
    assert "2026-10-30T16:53:33Z" in message
    assert "3m 19s" in message
    assert "did not start" in message and "Wait" in message
    # The running suite's lock is untouched.
    assert parse_holder((lock / "holder").read_text()).pid == HOLDER


def test_eastern_label_follows_daylight_saving(tmp_path: Path) -> None:
    lock = tmp_path / "suite.lock"
    started = datetime(2026, 11, 3, 15, 0, 0, tzinfo=UTC)
    hold(lock, "gate", HOLDER, started)
    with pytest.raises(SuiteLockRefused, match="10:00:00 EST"):
        acquire(
            lock,
            label="pytest",
            pid=ME,
            now=started + timedelta(seconds=5),
            processes=(*ALONE, proc(HOLDER, started=started)),
        )


def test_holder_in_own_lineage_passes_through_and_keeps_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "suite.lock"
    hold(lock, "restack-gates", PARENT)
    assert take(lock, ALONE) == suite_lock.Acquisition(held=False)
    release(lock, pid=ME)
    assert parse_holder((lock / "holder").read_text()).pid == PARENT


@pytest.mark.parametrize("holder_state", ["exited", "pid_reused_after_lock"])
def test_stale_lock_with_no_suite_running_is_recovered(tmp_path: Path, holder_state: str) -> None:
    lock = tmp_path / "suite.lock"
    hold(lock, "restack-gates", HOLDER)
    host = ALONE
    if holder_state == "pid_reused_after_lock":
        host = (*ALONE, proc(HOLDER, started=STARTED + timedelta(seconds=1), command="Safari"))
    acquisition = take(lock, host)
    assert acquisition.held is True
    assert acquisition.recovered == parse_holder("restack-gates 7000 2026-10-30T16:53:33Z")
    assert parse_holder((lock / "holder").read_text()).pid == ME
    assert sorted(p.name for p in tmp_path.iterdir()) == ["suite.lock"]  # stale copy removed


def test_dead_holder_with_a_suite_still_running_is_refused_never_stolen(tmp_path: Path) -> None:
    # MEASURED 2026-10-22: a killed driver's pytest kept running under its dead-pid holder.
    lock = tmp_path / "suite.lock"
    hold(lock, "restack-gates", HOLDER)
    orphan = proc(ORPHAN, command="/repo/.venv/bin/python -m pytest -q -n auto")
    with pytest.raises(SuiteLockRefused, match="is gone") as refused:
        take(lock, (*ALONE, orphan))
    assert f"{ORPHAN} " in str(refused.value)
    assert f"rm -r {lock}" in str(refused.value)
    assert parse_holder((lock / "holder").read_text()).pid == HOLDER


@pytest.mark.parametrize("shape", ["regular_file", "directory_without_holder", "garbled_holder"])
def test_invalid_lock_is_refused_loudly_never_stolen(tmp_path: Path, shape: str) -> None:
    lock = tmp_path / "suite.lock"
    if shape == "regular_file":
        lock.write_bytes(b"")  # MEASURED 2026-10-16: a 0-byte FILE wedged every mkdir loop
    else:
        lock.mkdir()
        if shape == "garbled_holder":
            (lock / "holder").write_text("garbled\n")
    with pytest.raises(SuiteLockRefused, match="not a valid lock"):
        take(lock, ALONE)
    assert lock.exists()


def test_release_removes_only_a_lock_this_process_holds(tmp_path: Path) -> None:
    lock = tmp_path / "suite.lock"
    assert take(lock, ALONE).held
    release(lock, pid=ME)
    assert not lock.exists()
    hold(lock, "restack-gates", HOLDER)
    release(lock, pid=ME)
    assert lock.exists()


def test_lineage_is_a_process_and_its_ancestors() -> None:
    host = (*ALONE, proc(HOLDER, PARENT), proc(ORPHAN, HOLDER))
    assert ancestor_pids(ME, host) == {ME, PARENT}
    assert ancestor_pids(ORPHAN, host) == {ORPHAN, HOLDER, PARENT}


def test_process_table_snapshots_this_process_and_its_parent() -> None:
    table = {p.pid: p for p in process_table()}
    me = table[os.getpid()]
    assert me.ppid == os.getppid() and os.getppid() in table
    assert me.started <= datetime.now(UTC)
    assert "python" in me.command.lower() or "pytest" in me.command


@pytest.fixture
def live_pid() -> Iterator[int]:
    """A live process that is not an ancestor of this test."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield child.pid
    finally:
        child.kill()
        child.wait()


def stub_config(numprocesses: object, *, worker: bool = False) -> SimpleNamespace:
    config = SimpleNamespace(
        option=SimpleNamespace(numprocesses=numprocesses),
        stash=pytest.Stash(),
        addinivalue_line=lambda *_args: None,
    )
    config.getoption = lambda name, default=None: getattr(config.option, name, default)
    if worker:
        config.workerinput = {"workerid": "gw0"}
    return config


def repo_conftest(request: pytest.FixtureRequest) -> object:
    conftest = Path(__file__).with_name("conftest.py").resolve()
    for plugin in request.config.pluginmanager.get_plugins():
        if Path(getattr(plugin, "__file__", "") or "/").resolve() == conftest:
            return plugin
    raise AssertionError("tests/conftest.py is not a registered plugin")


def test_repo_conftest_takes_and_releases_the_lock_for_an_xdist_controller(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "suite.lock"
    monkeypatch.setattr(suite_lock, "LOCK_PATH", lock)
    conftest, config = repo_conftest(request), stub_config(8)
    conftest.pytest_configure(config)
    assert parse_holder((lock / "holder").read_text()).pid == os.getpid()
    conftest.pytest_unconfigure(config)
    assert not lock.exists()


@pytest.mark.parametrize("numprocesses,worker", [(None, False), (0, False), (8, True)])
def test_serial_runs_and_xdist_workers_never_touch_the_lock(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    numprocesses: object,
    worker: bool,
) -> None:
    lock = tmp_path / "suite.lock"
    monkeypatch.setattr(suite_lock, "LOCK_PATH", lock)
    conftest, config = repo_conftest(request), stub_config(numprocesses, worker=worker)
    conftest.pytest_configure(config)
    assert not lock.exists()
    conftest.pytest_unconfigure(config)


def test_repo_conftest_exits_the_colliding_run_with_tempfail(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_pid: int
) -> None:
    lock = tmp_path / "suite.lock"
    hold(lock, "restack-gates", live_pid, datetime.now(UTC) + timedelta(seconds=1))
    monkeypatch.setattr(suite_lock, "LOCK_PATH", lock)
    conftest, config = repo_conftest(request), stub_config(8)
    with pytest.raises(pytest.exit.Exception) as exited:
        conftest.pytest_configure(config)
    assert exited.value.returncode == EXIT_LOCK_HELD
    assert "restack-gates" in str(exited.value)
    conftest.pytest_unconfigure(config)
    assert parse_holder((lock / "holder").read_text()).pid == live_pid


def test_recovery_puts_back_a_lock_that_changed_hands_after_inspection(tmp_path: Path) -> None:
    lock = tmp_path / "suite.lock"
    hold(lock, "fresh-run", ORPHAN, NOW)  # took the lock after this run judged HOLDER stale
    judged = parse_holder(f"restack-gates {HOLDER} 2026-10-30T16:53:33Z")
    with pytest.raises(SuiteLockRefused, match="changed hands"):
        suite_lock._set_aside(lock, judged)
    assert parse_holder((lock / "holder").read_text()).label == "fresh-run"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["suite.lock"]


CHILD_SEES_ITS_PARENT_HOLDING = (
    "import os, pathlib, sys; "
    "holder = pathlib.Path(sys.argv[1], 'holder').read_text().split(); "
    "pathlib.Path(sys.argv[2]).write_text(holder[-2]); "
    "sys.exit(int(holder[-2] != str(os.getppid())) or int(sys.argv[3]))"
)


def held_command(lock: Path, marker: Path, code: int = 0) -> list[str]:
    return [sys.executable, "-c", CHILD_SEES_ITS_PARENT_HOLDING, str(lock), str(marker), str(code)]


def test_hold_runs_the_whole_command_under_the_lock_then_releases(tmp_path: Path) -> None:
    lock, marker = tmp_path / "suite.lock", tmp_path / "ran"
    assert suite_lock.hold(lock, "ship-it", held_command(lock, marker)) == 0
    assert marker.read_text() == str(os.getpid())  # the gate's children see this holder
    assert not lock.exists()


def test_hold_propagates_the_command_exit_code_and_still_releases(tmp_path: Path) -> None:
    lock, marker = tmp_path / "suite.lock", tmp_path / "ran"
    assert suite_lock.hold(lock, "ship-it", held_command(lock, marker, 3)) == 3
    assert not lock.exists()


def test_hold_refuses_a_collision_without_running_the_command(
    tmp_path: Path, live_pid: int, capsys: pytest.CaptureFixture[str]
) -> None:
    lock, marker = tmp_path / "suite.lock", tmp_path / "ran"
    hold(lock, "restack-gates", live_pid, datetime.now(UTC) + timedelta(seconds=1))
    assert suite_lock.hold(lock, "ship-it", held_command(lock, marker)) == EXIT_LOCK_HELD
    assert not marker.exists()
    assert "restack-gates" in capsys.readouterr().err
    assert parse_holder((lock / "holder").read_text()).pid == live_pid


@pytest.mark.parametrize("argv", [[], ["ship-it"], ["ship-it", "--"], ["ship-it", "just", "check"]])
def test_cli_without_label_separator_and_command_is_a_usage_error(argv: list[str]) -> None:
    assert suite_lock.main(argv) == 2


def test_ship_it_holds_the_lock_across_check_container_test_and_submit() -> None:
    justfile = (Path(__file__).resolve().parents[1] / "justfile").read_text(encoding="utf-8")
    header = re.search(r"^ship-it:.*$", justfile, flags=re.MULTILINE)
    assert header is not None and header.group(0) == "ship-it:", (
        "prerequisites on ship-it run before its body, outside the lock"
    )
    body = justfile[header.end() : header.end() + 400].split("\n\n")[0]
    held = body.split("tests/suite_lock.py ship-it --", 1)
    assert len(held) == 2, "ship-it must run its steps under tests/suite_lock.py"
    steps = [held[1].index(step) for step in ("just check", "just container-test", "gt submit")]
    assert steps == sorted(steps)
