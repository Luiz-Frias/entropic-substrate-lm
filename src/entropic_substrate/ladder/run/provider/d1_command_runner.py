"""Shared shell-free subprocess runner for D1 operator adapters."""

from __future__ import annotations

import hashlib
import math
import os
import selectors
import signal
import subprocess  # nosec B404 -- fixed argv adapter; shell execution is never used
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, NoReturn, cast

from entropic_substrate.ladder.run.values.d1_operator_ports import D1CommandResult

__all__ = [
    "BoundedSubprocessD1CommandRunner",
    "D1BoundedProcessError",
    "D1BoundedProcessResult",
    "D1BoundedProcessRunner",
    "SubprocessD1CommandRunner",
]

_REAP_TIMEOUT_SECONDS = 1.0
_READ_CHUNK_BYTES = 64 * 1024


def _argv(value: tuple[str, ...]) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise TypeError("D1 command runner argv must be a nonempty tuple")
    if any(type(part) is not str or not part for part in value):
        raise TypeError("D1 command runner argv entries must be nonempty strings")
    return value


def _cwd(value: Path | None) -> Path | None:
    if value is None:
        return None
    if not isinstance(cast(object, value), Path):
        raise TypeError("D1 command runner cwd must be a Path or None")
    lexical = Path(os.path.abspath(value))
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("D1 command runner cwd must be a real directory")
    try:
        root = lexical.resolve(strict=True)
    except OSError as error:
        raise ValueError("D1 command runner cwd must be a real directory") from error
    if not root.is_dir():
        raise ValueError("D1 command runner cwd must be a real directory")
    return root


def _stdin(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError("D1 command runner stdin must be a string or None")
    return value


@dataclass(frozen=True, slots=True)
class D1BoundedProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class D1BoundedProcessError(RuntimeError):
    """Stable process-bound failure carrying hashes instead of raw output."""

    def __init__(
        self,
        code: str,
        *,
        stdout: bytes,
        stderr: bytes,
        process_group_swept: bool,
        process_reaped: bool,
    ) -> None:
        self.code = code
        self.stdout_sha256 = hashlib.sha256(stdout).hexdigest()
        self.stderr_sha256 = hashlib.sha256(stderr).hexdigest()
        self.output_bytes = len(stdout) + len(stderr)
        self.process_group_swept = process_group_swept
        self.process_reaped = process_reaped
        super().__init__(f"D1 bounded process {code}; output_bytes={self.output_bytes}")


class D1BoundedProcessRunner:
    """Drain one shell-free process group under time and output bounds."""

    def __init__(self, *, timeout_seconds: float, max_output_bytes: int) -> None:
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("D1 bounded process timeout must be positive")
        if type(max_output_bytes) is not int or max_output_bytes < 1:
            raise ValueError("D1 bounded process output limit must be positive")
        self._timeout_seconds = float(timeout_seconds)
        self._max_output_bytes = max_output_bytes

    @staticmethod
    def _close(stream: IO[bytes] | None) -> None:
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass

    @staticmethod
    def _signal_group(pgid: int) -> bool:
        """Report whether SIGKILL reached the group or the group was already gone.

        MEASURED 2026-09-22 on Darwin 27.0.0 (three-arm probe over one
        `start_new_session=True` child, signal 0 and SIGKILL, live arm as the
        control): killpg answers 0 while a member is alive, ESRCH only once the
        group itself no longer exists, and EPERM while the group still exists but
        holds no signalable member -- that is, while the child has exited and has
        not yet been reaped. Delivery therefore cannot be decided from EPERM: it
        is neither proof of a sweep nor proof of a live survivor.
        """
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return True

    @staticmethod
    def _group_absent(pgid: int) -> bool:
        """Observe whether the group still exists, using a signal-0 probe.

        Absence is ESRCH and nothing else, per the same measurement recorded on
        `_signal_group`. The probe runs only after our own child is reaped, so a
        recycled pid can at worst make a swept group read as present; it can
        never make a surviving group read as absent.
        """
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return False

    @staticmethod
    def _reap(process: subprocess.Popen[bytes]) -> bool:
        try:
            process.wait(timeout=_REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _sweep_and_reap(self, process: subprocess.Popen[bytes]) -> tuple[bool, bool]:
        """Kill the group, reap our child, then confirm the group is gone.

        The sweep is confirmed by the group's observed absence after the reap,
        never by a single killpg sample taken before it: an exited-but-unreaped
        child answers EPERM, which the sample cannot tell apart from a genuine
        failure to signal. `wait` is a synchronizing event, so the confirmation
        needs no window of its own.
        """
        delivered = self._signal_group(process.pid)
        reaped = self._reap(process)
        if delivered:
            return True, reaped
        return reaped and self._group_absent(process.pid), reaped

    def _kill_group(
        self,
        process: subprocess.Popen[bytes],
        *,
        code: str,
        stdout: bytearray,
        stderr: bytearray,
    ) -> NoReturn:
        swept, reaped = self._sweep_and_reap(process)
        raise D1BoundedProcessError(
            code,
            stdout=bytes(stdout),
            stderr=bytes(stderr),
            process_group_swept=swept,
            process_reaped=reaped,
        )

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        input: bytes,
        env: Mapping[str, str],
    ) -> D1BoundedProcessResult:
        if (
            not isinstance(cast(object, cwd), Path)
            or type(input) is not bytes
            or not isinstance(cast(object, env), Mapping)
        ):
            raise TypeError("D1 bounded process requires Path, bytes, and environment mapping")
        stdout, stderr = bytearray(), bytearray()
        pending = memoryview(input)
        process = subprocess.Popen(  # nosec B603 -- explicit argv and shell-free contract
            _argv(argv),
            cwd=_cwd(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=dict(env),
            close_fds=True,
            start_new_session=True,
        )
        selector: selectors.BaseSelector | None = None
        primary: BaseException | None = None
        group_swept = False
        try:
            if process.stdin is None or process.stdout is None or process.stderr is None:
                self._kill_group(
                    process,
                    code="subprocess_failure",
                    stdout=stdout,
                    stderr=stderr,
                )
            selector = selectors.DefaultSelector()
            for stream in (process.stdin, process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
            if pending:
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                self._close(process.stdin)
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            deadline = time.monotonic() + self._timeout_seconds
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_group(process, code="timeout", stdout=stdout, stderr=stderr)
                for key, _ in selector.select(remaining):
                    stream = cast(IO[bytes], key.fileobj)
                    if key.data == "stdin":
                        try:
                            written = os.write(stream.fileno(), pending)
                            pending = pending[written:]
                        except BrokenPipeError:
                            pending = pending[len(pending) :]
                        if not pending:
                            selector.unregister(stream)
                            self._close(stream)
                        continue
                    try:
                        chunk = os.read(stream.fileno(), _READ_CHUNK_BYTES)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(stream)
                        self._close(stream)
                        continue
                    target = stdout if key.data == "stdout" else stderr
                    available = self._max_output_bytes + 1 - len(stdout) - len(stderr)
                    target.extend(chunk[: max(0, available)])
                    if len(stdout) + len(stderr) > self._max_output_bytes:
                        self._kill_group(
                            process,
                            code="output_limit",
                            stdout=stdout,
                            stderr=stderr,
                        )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_group(process, code="timeout", stdout=stdout, stderr=stderr)
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                self._kill_group(process, code="timeout", stdout=stdout, stderr=stderr)
            group_swept, _ = self._sweep_and_reap(process)
            if not group_swept:
                self._kill_group(
                    process,
                    code="subprocess_failure",
                    stdout=stdout,
                    stderr=stderr,
                )
            return D1BoundedProcessResult(returncode, bytes(stdout), bytes(stderr))
        except BaseException as error:
            primary = error
            raise
        finally:
            selector_close_failed = False
            if selector is not None:
                try:
                    selector.close()
                except OSError:
                    selector_close_failed = True
            if group_swept:
                swept, reaped = True, self._reap(process)
            else:
                swept, reaped = self._sweep_and_reap(process)
            self._close(process.stdin)
            self._close(process.stdout)
            self._close(process.stderr)
            if primary is not None:
                if selector_close_failed:
                    primary.add_note("selector cleanup also failed")
                if not swept or not reaped:
                    primary.add_note("process cleanup could not be proven")
            elif selector_close_failed or not swept or not reaped:
                raise D1BoundedProcessError(
                    "subprocess_failure",
                    stdout=bytes(stdout),
                    stderr=bytes(stderr),
                    process_group_swept=swept,
                    process_reaped=reaped,
                )


class BoundedSubprocessD1CommandRunner:
    """Adapt the shared binary process owner to the text command port."""

    def __init__(self, *, timeout_seconds: float, max_output_bytes: int) -> None:
        self._runner = D1BoundedProcessRunner(
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        stdin: str | None,
    ) -> D1CommandResult:
        validated_stdin = _stdin(stdin)
        result = self._runner.run(
            _argv(argv),
            cwd=Path.cwd() if cwd is None else cwd,
            input=b"" if validated_stdin is None else validated_stdin.encode("utf-8"),
            env=dict(os.environ),
        )
        try:
            return D1CommandResult(
                stdout=result.stdout.decode("utf-8"),
                stderr=result.stderr.decode("utf-8"),
                returncode=result.returncode,
            )
        except UnicodeDecodeError as error:
            raise D1BoundedProcessError(
                "invalid_utf8",
                stdout=result.stdout,
                stderr=result.stderr,
                process_group_swept=True,
                process_reaped=True,
            ) from error


class SubprocessD1CommandRunner:
    """Execute an explicit argv tuple and capture UTF-8 text streams."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        stdin: str | None,
    ) -> D1CommandResult:
        result = subprocess.run(  # nosec B603 -- argv is explicit and shell-free
            _argv(argv),
            cwd=_cwd(cwd),
            input=_stdin(stdin),
            capture_output=True,
            text=True,
            check=False,
        )
        return D1CommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
