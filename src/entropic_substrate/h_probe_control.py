"""Single-writer H0 control mailboxes; immutable custody is not execution authority."""

import fcntl
import json
import os
import re
import stat
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from entropic_substrate.d1_artifact_snapshot import (
    normalize_d1_artifact_root,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.h_probe_launch import H0ProbeLaunch, parse_probe_launch
from entropic_substrate.sentence_prediction.recovery_state import recovery_record


def _root(root: Path) -> Path:
    directory = normalize_d1_artifact_root(root)
    metadata = directory.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("probe control root must be private and owned by the current user")
    return directory


def _scope(launch: H0ProbeLaunch, role: str) -> H0ProbeLaunch:
    if role not in ("controller", "watchdog") or type(role) is not str:
        raise ValueError("unknown probe control role")
    if type(launch) is not H0ProbeLaunch:
        raise TypeError("probe control requires a nominal launch")
    return parse_probe_launch(launch.payload, max_bytes=launch.request.limits.max_object_bytes)


def _name(role: str, name: str) -> str:
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name) is None:
        raise ValueError("probe control name is unsafe")
    return f"{role}-{name}.json"


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _verify_launch(root: Path, launch: H0ProbeLaunch) -> None:
    _, raw, _ = read_d1_artifact_bytes(
        root, "launch.json", label="probe launch", max_bytes=launch.request.limits.max_object_bytes
    )
    if raw != launch.payload:
        raise ValueError("probe control directory belongs to another launch")


def read_probe_control(
    root: Path, launch: H0ProbeLaunch, role: str, name: str, *, max_bytes: int
) -> dict[str, object] | None:
    """Read bound custody bytes; the caller still admits the message-specific payload."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("probe control read requires a positive byte bound")
    launch = _scope(launch, role)
    directory, relative = _root(root), _name(role, name)
    path = directory / relative
    if not path.exists() and not path.is_symlink():
        return None
    _verify_launch(directory, launch)
    _, raw, _ = read_d1_artifact_bytes(
        directory, relative, label="probe control", max_bytes=max_bytes
    )
    row = recovery_record(json.loads(raw), "schema authority launch_sha256 role name payload")
    if (
        row["schema"] != "h-probe-control-v1"
        or row["authority"] != "byte_custody_only"
        or row["launch_sha256"] != launch.digest
        or row["role"] != role
        or row["name"] != name
        or _json(row) != raw
        or type(row["payload"]) is not dict
    ):
        raise ValueError("probe control identity, encoding or scope mismatch")
    return row["payload"]


class _ProbeControlWriter:
    """Valid only while its role lock is held in this process; never claims readiness."""

    def __init__(self, root: Path, launch: H0ProbeLaunch, role: str) -> None:
        self._root, self._launch, self._role = root, launch, role
        self._pid = os.getpid()
        self._active = True

    def write(self, name: str, payload: dict[str, object], *, max_bytes: int) -> ArtifactIdentity:
        if not self._active or self._pid != os.getpid():
            raise ValueError("probe role was released or belongs to another process")
        _verify_launch(_root(self._root), self._launch)
        if type(max_bytes) is not int or max_bytes < 1 or type(payload) is not dict:
            raise ValueError("probe control writes require an explicit bound and object payload")
        relative = _name(self._role, name)
        raw = _json(
            {
                "schema": "h-probe-control-v1",
                "authority": "byte_custody_only",
                "launch_sha256": self._launch.digest,
                "role": self._role,
                "name": name,
                "payload": payload,
            }
        )
        if len(raw) > max_bytes:
            raise ValueError("probe control exceeds its byte bound")
        return publish_d1_atomic_bytes(self._root, relative, raw, label="probe control")

    def release(self) -> None:
        self._active = False


@contextmanager
def _locked_file(directory: Path, name: str, *, wait: bool) -> Generator[None]:
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                directory / f"{name}.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
            ):
                raise ValueError("probe role lock is unsafe")
            fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            raise ValueError("probe role is already owned") from None
        except OSError:
            raise ValueError("probe role lock could not be safely opened") from None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def own_probe_role(root: Path, launch: H0ProbeLaunch, role: str) -> Generator[_ProbeControlWriter]:
    """Lease one role without waiting; process exit releases the OS lock, not history.

    The short shared initialization lock serializes the two roles' manifest bind.
    A restarted controller must inspect existing intent before any create call.
    A stale ready file never proves its original watchdog process is alive.
    The bound launch uses the request's explicit artifact-size envelope.
    """
    launch, directory = _scope(launch, role), _root(root)
    with _locked_file(directory, role, wait=False):
        with _locked_file(directory, "launch", wait=True):
            manifest = directory / "launch.json"
            if manifest.exists() or manifest.is_symlink():
                _verify_launch(directory, launch)
            else:
                publish_d1_atomic_bytes(
                    directory, "launch.json", launch.payload, label="probe launch"
                )
        writer = _ProbeControlWriter(directory, launch, role)
        try:
            yield writer
        finally:
            writer.release()
