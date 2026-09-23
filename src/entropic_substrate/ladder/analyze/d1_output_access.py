"""UID-specific Linux ACL strategy for the D1 worker output mount."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from stat import S_IMODE
from typing import Protocol

__all__ = [
    "D1AclCommandResult",
    "D1OutputAccessError",
    "D1SetfaclOutputAccess",
]

_UID = 65532
_GID = 65532


class D1OutputAccessError(RuntimeError):
    """ACL grant/revoke failed without weakening ordinary directory mode."""


@dataclass(frozen=True, slots=True)
class D1AclCommandResult:
    invoked_argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        if (
            type(self.invoked_argv) is not tuple
            or not self.invoked_argv
            or any(type(item) is not str or not item for item in self.invoked_argv)
        ):
            raise TypeError("D1 ACL command result requires nonempty string argv")
        if type(self.returncode) is not int:
            raise TypeError("D1 ACL command result returncode must be integer")
        if type(self.stdout) is not str or type(self.stderr) is not str:
            raise TypeError("D1 ACL command result streams must be strings")


class _AclCommandPort(Protocol):
    def run(self, argv: tuple[str, ...]) -> D1AclCommandResult: ...


def _path(value: object) -> Path:
    if not isinstance(value, Path):
        raise TypeError("D1 ACL output path must be Path")
    lexical = Path(os.path.abspath(value))
    if (
        lexical.is_symlink()
        or not lexical.is_dir()
        or lexical.resolve() != lexical
        or any(character in str(lexical) for character in ("\n", "\r"))
    ):
        raise D1OutputAccessError("D1 ACL output path must be a real safe directory")
    mode = S_IMODE(lexical.stat().st_mode)
    if mode & 0o700 != 0o700 or mode & 0o077:
        raise D1OutputAccessError("D1 ACL output path requires private mode 0700")
    return lexical


def _identity(uid: int, gid: int) -> None:
    if type(uid) is not int or type(gid) is not int or (uid, gid) != (_UID, _GID):
        raise D1OutputAccessError("D1 ACL output identity must be 65532:65532")


class D1SetfaclOutputAccess:
    """Grant one verified ACL lease to the fixed container identity."""

    def __init__(self, *, runner: _AclCommandPort) -> None:
        if not hasattr(runner, "run"):
            raise TypeError("D1 ACL access requires a command runner")
        self._runner = runner
        self._lock = threading.Lock()
        self._active: tuple[Path, int, int, int] | None = None

    def _invoke(self, argv: tuple[str, ...]) -> D1AclCommandResult:
        try:
            result = self._runner.run(argv)
        except Exception as error:
            raise D1OutputAccessError("D1 ACL command execution failed") from error
        if type(result) is not D1AclCommandResult:
            raise D1OutputAccessError("D1 ACL runner returned unsupported evidence")
        if result.invoked_argv != argv:
            raise D1OutputAccessError("D1 ACL observed argv differs from request")
        if result.returncode != 0:
            raise D1OutputAccessError("D1 ACL command returned nonzero")
        return result

    def _entry(self, path: Path, uid: int) -> str | None:
        result = self._invoke(("getfacl", "-cpn", "--", str(path)))
        prefix = f"user:{uid}:"
        named = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("user:") and not line.startswith("user::")
        ]
        if any(not line.startswith(prefix) for line in named):
            raise D1OutputAccessError("D1 ACL output contains a nonnumeric user identity")
        entries = [line.removeprefix(prefix) for line in named]
        if len(entries) > 1:
            raise D1OutputAccessError("D1 ACL output has duplicate named user entries")
        return None if not entries else entries[0]

    def _set(self, path: Path, uid: int) -> None:
        self._invoke(("setfacl", "-m", f"u:{uid}:rwx", "--", str(path)))

    def _remove(self, path: Path, uid: int) -> None:
        self._invoke(("setfacl", "-x", f"u:{uid}", "--", str(path)))

    def grant(self, path: Path, *, uid: int, gid: int) -> None:
        _identity(uid, gid)
        output = _path(path)
        with self._lock:
            if self._active is not None:
                raise D1OutputAccessError("D1 ACL output lease is already active")
            if self._entry(output, uid) is not None:
                raise D1OutputAccessError("D1 ACL output has a pre-existing identity entry")
            mode = S_IMODE(output.stat().st_mode)
            self._set(output, uid)
            if self._entry(output, uid) != "rwx":
                self._remove(output, uid)
                raise D1OutputAccessError("D1 ACL grant verification failed")
            if S_IMODE(output.stat().st_mode) != mode:
                self._remove(output, uid)
                raise D1OutputAccessError("D1 ACL grant changed ordinary directory mode")
            self._active = (output, uid, gid, mode)

    def revoke(self, path: Path, *, uid: int, gid: int) -> None:
        _identity(uid, gid)
        output = _path(path)
        with self._lock:
            if self._active != (output, uid, gid, S_IMODE(output.stat().st_mode)):
                raise D1OutputAccessError("D1 ACL revoke differs from the active lease")
            self._remove(output, uid)
            if self._entry(output, uid) is not None:
                raise D1OutputAccessError("D1 ACL revoke verification failed")
            self._active = None
