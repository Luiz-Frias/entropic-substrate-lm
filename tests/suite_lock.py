"""Host suite lock: at most one xdist suite per host; a colliding run refuses, never races.

MEASURED 2026-10-30: a terminal ``gt submit`` pre-push suite went red on
``tests/test_h_probe_process.py`` (watchdog ready deadline, 10 s) while the
host sat at swap 96% and 15-minute load 22 on 8 GiB / 8 cores, with the podman
VM holding 4 GiB and another worktree's ``-n auto`` hook suite resident. The
same test passed 3/3 alone at the same HEAD; its spawn-to-ready latency was
0.70-0.80 s idle and 2.07-2.26 s under 16 CPU burners. Same failure class as
the 2026-09-23 three-lane livelock (load 15-19, swap over 90%). The lock that
followed lived only in workflow prompts as ``mkdir /tmp/esl-suite.lock``, so
terminal pushes and other sessions' hooks never took it.

This module takes that same lock from ``tests/conftest.py`` on the xdist
controller of every ``-n`` run, so every entry point (hooks, ``just``
recipes, agents) passes through it:

- free: ``mkdir`` it and write the one-line holder the existing wrappers
  write, ``<label> <pid> <UTC ISO>``;
- held by this process or an ancestor (a wrapper that took the lock and
  spawned this suite): pass through, and leave the lock to that holder;
- held by a live process that started before the lock was taken: refuse THIS
  run before collection with the holder's label, pid and Eastern start time,
  exit 75 (EX_TEMPFAIL); the running suite is untouched;
- holder gone (pid absent, or reused by a process that started after the
  lock): recover the lock only when no ``pytest`` process outside this run's
  lineage is alive, else refuse and list them. MEASURED 2026-10-22: a killed
  driver left its suite children running under a dead-pid holder, so a dead
  holder alone does not prove the suite is gone; no clock threshold stands in
  for that evidence, since a live gate legitimately holds the lock 30-40 min;
- not a valid lock: refuse loudly, never recover. MEASURED 2026-10-16: a
  0-byte regular FILE at the lock path wedged every ``until mkdir`` loop, and
  a directory without a holder line may be a wrapper between its ``mkdir``
  and its ``echo``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

LOCK_PATH = Path("/tmp/esl-suite.lock")
EXIT_LOCK_HELD = 75  # sysexits EX_TEMPFAIL: try again later
EASTERN = ZoneInfo("America/New_York")
_HELD = pytest.StashKey[Path]()


class SuiteLockRefused(Exception):
    """This run must not start; the message tells a person or agent what to do."""


@dataclass(frozen=True, slots=True)
class Holder:
    label: str
    pid: int
    started: datetime


@dataclass(frozen=True, slots=True)
class Process:
    pid: int
    ppid: int
    started: datetime
    command: str


@dataclass(frozen=True, slots=True)
class Acquisition:
    held: bool  # True: this run owns the lock and must release it
    recovered: Holder | None = None  # a stale holder this run cleared


def parse_holder(text: str) -> Holder:
    parts = text.strip().rsplit(" ", 2)
    if len(parts) != 3 or not parts[0] or not parts[1].isdigit():
        raise ValueError(f"holder line is not '<label> <pid> <UTC ISO>': {text!r}")
    started = datetime.fromisoformat(parts[2])
    if started.utcoffset() is None:
        raise ValueError(f"holder start time has no offset: {text!r}")
    return Holder(parts[0], int(parts[1]), started.astimezone(UTC))


def process_table() -> tuple[Process, ...]:
    """One ``ps`` snapshot: pid, parent, start time, command line."""
    rows = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,lstart=,command="],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    ).stdout
    table = []
    for row in rows.splitlines():
        fields = row.split(None, 7)
        if len(fields) < 7:
            continue
        started = datetime.strptime(" ".join(fields[2:7]), "%a %b %d %H:%M:%S %Y")
        command = fields[7] if len(fields) == 8 else ""
        table.append(Process(int(fields[0]), int(fields[1]), started.astimezone(UTC), command))
    return tuple(table)


def ancestor_pids(pid: int, processes: Sequence[Process]) -> frozenset[int]:
    """``pid`` and every ancestor in ``processes``."""
    parents = {p.pid: p.ppid for p in processes}
    lineage: set[int] = set()
    while pid > 1 and pid not in lineage:
        lineage.add(pid)
        pid = parents.get(pid, 0)
    return frozenset(lineage)


def acquire(
    lock: Path, *, label: str, pid: int, now: datetime, processes: Sequence[Process]
) -> Acquisition:
    """Take ``lock``, pass through an ancestor's, recover a stale one, or refuse."""
    recovered = None
    for _attempt in range(3):
        try:
            lock.mkdir()
        except FileExistsError:
            holder = _inspect(lock)
            lineage = ancestor_pids(pid, processes)
            if holder.pid in lineage:
                return Acquisition(held=False)
            running = {p.pid: p for p in processes}.get(holder.pid)
            if running is not None and running.started <= holder.started:
                raise SuiteLockRefused(_held(lock, holder, now)) from None
            suites = [p for p in processes if "pytest" in p.command and p.pid not in lineage]
            if suites:
                raise SuiteLockRefused(_orphaned(lock, holder, suites)) from None
            _set_aside(lock, holder)
            recovered = holder
            continue
        (lock / "holder").write_text(f"{label} {pid} {now.astimezone(UTC):%Y-%m-%dT%H:%M:%SZ}\n")
        return Acquisition(held=True, recovered=recovered)
    raise SuiteLockRefused(_invalid(lock, "it changed hands during recovery; rerun"))


def release(lock: Path, *, pid: int) -> None:
    """Remove ``lock`` only if its holder line names ``pid``."""
    try:
        if parse_holder((lock / "holder").read_text()).pid != pid:
            return
    except (OSError, ValueError):
        return
    (lock / "holder").unlink()
    lock.rmdir()


def configure_suite_lock(config: pytest.Config) -> None:
    """Take the lock on an xdist controller; serial runs and workers never touch it."""
    if hasattr(config, "workerinput") or config.getoption("numprocesses", None) in (None, 0):
        return
    lock = LOCK_PATH
    try:
        acquisition = acquire(
            lock,
            label=f"pytest@{Path.cwd().name}",
            pid=os.getpid(),
            now=datetime.now(UTC),
            processes=process_table(),
        )
    except SuiteLockRefused as refused:
        pytest.exit(str(refused), returncode=EXIT_LOCK_HELD)
    if acquisition.recovered is not None:
        gone = acquisition.recovered
        # Global capture is already active here; write past it so the notice is seen.
        sys.__stderr__.write(
            f"[suite-lock] Recovered a stale lock from {gone.label} (pid {gone.pid}), "
            f"started {_when(gone.started)}: holder gone, no pytest running.\n"
        )
    if acquisition.held:
        config.stash[_HELD] = lock


def release_suite_lock(config: pytest.Config) -> None:
    lock = config.stash.get(_HELD, None)
    if lock is not None:
        release(lock, pid=os.getpid())


def _inspect(lock: Path) -> Holder:
    if lock.is_symlink() or not lock.is_dir():
        raise SuiteLockRefused(_invalid(lock, "it is not a directory"))
    try:
        text = (lock / "holder").read_text()
    except FileNotFoundError:
        raise SuiteLockRefused(
            _invalid(lock, "it has no holder line: still being acquired, or orphaned")
        ) from None
    try:
        return parse_holder(text)
    except ValueError:
        raise SuiteLockRefused(_invalid(lock, f"its holder line is unreadable: {text!r}")) from None


def _set_aside(lock: Path, judged: Holder) -> None:
    """Rename the stale lock away atomically, then confirm it was the one judged stale."""
    stale = lock.with_name(f"{lock.name}.stale-{os.getpid()}")
    try:
        lock.rename(stale)
    except FileNotFoundError:
        return  # another run set it aside first; the caller retries mkdir
    try:
        moved = parse_holder((stale / "holder").read_text())
    except (OSError, ValueError):
        moved = None
    if moved != judged:
        stale.rename(lock)  # a fresh holder took it between inspection and rename
        raise SuiteLockRefused(_invalid(lock, "it changed hands during recovery; rerun"))
    shutil.rmtree(stale)


def _when(started: datetime) -> str:
    return f"{started.astimezone(EASTERN):%H:%M:%S %Z} ({started:%Y-%m-%dT%H:%M:%SZ})"


def _held(lock: Path, holder: Holder, now: datetime) -> str:
    seconds = max(0, int((now - holder.started).total_seconds()))
    hours, rest = divmod(seconds, 3600)
    elapsed = (
        f"{hours}h {rest // 60:02d}m {rest % 60:02d}s"
        if hours
        else f"{rest // 60}m {rest % 60:02d}s"
    )
    return (
        f"[suite-lock] Another test suite holds the host suite lock: {holder.label} "
        f"(pid {holder.pid}), started {_when(holder.started)}, running {elapsed}.\n"
        "[suite-lock] This run did not start; the running suite is unaffected. "
        "Wait for it to finish, or queue behind it, then rerun.\n"
        f"[suite-lock] Holder: {lock / 'holder'}"
    )


def _orphaned(lock: Path, holder: Holder, suites: Sequence[Process]) -> str:
    listing = "\n".join(f"  {p.pid} {p.command[:120]}" for p in suites)
    return (
        f"[suite-lock] The host suite lock names {holder.label} (pid {holder.pid}), "
        f"started {_when(holder.started)}, but that process is gone while pytest is "
        f"still running:\n{listing}\n"
        "[suite-lock] This run did not start. Once those finish, the next run recovers "
        f"the lock by itself; to clear it by hand: rm -r {lock}"
    )


def _invalid(lock: Path, reason: str) -> str:
    return (
        f"[suite-lock] {lock} is not a valid lock: {reason}.\n"
        "[suite-lock] This run did not start. If no suite is running, remove it: "
        f"rm -r {lock}"
    )
