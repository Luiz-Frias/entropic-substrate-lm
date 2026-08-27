"""TDD contract for the shared shell-free D1 command runner."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import cast

import pytest

import entropic_substrate.d1_command_runner as runner_module
from entropic_substrate.d1_command_runner import (
    BoundedSubprocessD1CommandRunner,
    D1BoundedProcessError,
    D1BoundedProcessRunner,
    SubprocessD1CommandRunner,
)
from entropic_substrate.d1_operator_ports import D1CommandResult

pytestmark = pytest.mark.unit


def test_command_result_requires_exact_integer_exit_status() -> None:
    with pytest.raises(TypeError, match="returncode"):
        D1CommandResult("", "", True)  # type: ignore[arg-type]


def _process_running(pid: int) -> bool:
    result = subprocess.run(  # nosec B603 -- fixed diagnostic argv in a unit test
        ("ps", "-o", "stat=", "-p", str(pid)),
        capture_output=True,
        check=False,
        shell=False,
    )
    state = result.stdout.decode().strip()
    return result.returncode == 0 and bool(state) and not state.startswith("Z")


def test_runner_captures_stdout_stderr_cwd_and_stdin(tmp_path: Path) -> None:
    script = (
        "import os,sys;"
        "sys.stdout.write(os.getcwd() + '\\n');"
        "sys.stdout.write(sys.stdin.read());"
        "sys.stderr.write('warning-stream')"
    )

    result = SubprocessD1CommandRunner().run(
        (sys.executable, "-c", script),
        cwd=tmp_path,
        stdin="payload",
    )

    assert result.stdout == f"{tmp_path}\npayload"
    assert result.stderr == "warning-stream"
    assert result.returncode == 0


def test_runner_uses_shell_free_checked_text_subprocess_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _fake_run(argv: tuple[str, ...], **kwargs: object) -> _Completed:
        observed["argv"] = argv
        observed.update(kwargs)
        return _Completed()

    monkeypatch.setattr(runner_module.subprocess, "run", _fake_run)

    result = SubprocessD1CommandRunner().run(("echo", "ok"), cwd=tmp_path, stdin=None)

    assert result.stdout == "ok"
    assert observed["argv"] == ("echo", "ok")
    assert observed["capture_output"] is True
    assert observed["check"] is False
    assert observed["cwd"] == tmp_path.resolve()
    assert observed["input"] is None
    assert observed["text"] is True
    assert observed.get("shell", False) is False


@pytest.mark.parametrize(
    ("argv", "cwd", "stdin", "error"),
    [
        ((), None, None, "argv"),
        ((sys.executable, ""), None, None, "argv"),
        ((sys.executable, "-c", "print('x')"), Path("/definitely/missing"), None, "cwd"),
        ((sys.executable, "-c", "print('x')"), None, cast(object, 1), "stdin"),
    ],
)
def test_runner_rejects_invalid_invocation_shapes(
    argv: tuple[str, ...],
    cwd: Path | None,
    stdin: object,
    error: str,
) -> None:
    runner = SubprocessD1CommandRunner()

    with pytest.raises((TypeError, ValueError), match=error):
        runner.run(argv, cwd=cwd, stdin=stdin)  # type: ignore[arg-type]


def test_runner_refuses_a_working_directory_below_a_symlinked_parent(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    nested = real / "nested"
    nested.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="cwd"):
        SubprocessD1CommandRunner().run(
            (sys.executable, "-c", "raise SystemExit(0)"),
            cwd=alias / "nested",
            stdin=None,
        )


def test_runner_returns_nonzero_exit_and_diagnostic_streams_without_raising(tmp_path: Path) -> None:
    script = "import sys; sys.stderr.write('boom'); raise SystemExit(3)"

    result = SubprocessD1CommandRunner().run(
        (sys.executable, "-c", script), cwd=tmp_path, stdin=None
    )

    assert (result.returncode, result.stdout, result.stderr) == (3, "", "boom")


def test_bounded_runner_returns_binary_streams_and_nonzero_exit(tmp_path: Path) -> None:
    result = D1BoundedProcessRunner(timeout_seconds=2, max_output_bytes=1024).run(
        (
            sys.executable,
            "-c",
            "import sys;sys.stdout.buffer.write(sys.stdin.buffer.read());"
            "sys.stderr.buffer.write(b'warning');raise SystemExit(7)",
        ),
        cwd=tmp_path,
        input=b"payload",
        env=dict(os.environ),
    )

    assert (result.returncode, result.stdout, result.stderr) == (7, b"payload", b"warning")


def test_bounded_text_runner_reuses_timeout_output_and_utf8_boundary(tmp_path: Path) -> None:
    runner = BoundedSubprocessD1CommandRunner(timeout_seconds=2, max_output_bytes=1024)
    result = runner.run(
        (sys.executable, "-c", "import sys;print(sys.stdin.read());raise SystemExit(3)"),
        cwd=tmp_path,
        stdin="payload",
    )
    assert (result.returncode, result.stdout, result.stderr) == (3, "payload\n", "")

    with pytest.raises(D1BoundedProcessError) as timeout:
        BoundedSubprocessD1CommandRunner(
            timeout_seconds=0.1,
            max_output_bytes=1024,
        ).run(
            (sys.executable, "-c", "import time;time.sleep(60)"),
            cwd=tmp_path,
            stdin=None,
        )
    assert timeout.value.code == "timeout"

    with pytest.raises(D1BoundedProcessError) as invalid_utf8:
        runner.run(
            (sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'\\xff')"),
            cwd=tmp_path,
            stdin=None,
        )
    assert invalid_utf8.value.code == "invalid_utf8"
    assert invalid_utf8.value.output_bytes == 1
    assert (
        invalid_utf8.value.stdout_sha256
        == (
            "a8100ae6aa1940d0b663bb31cd466142ebbdbd5187131b92d93818987832eb89"  # pragma: allowlist secret
        )
    )
    assert invalid_utf8.value.process_group_swept is True
    assert invalid_utf8.value.process_reaped is True


@pytest.mark.parametrize(
    ("script", "timeout", "limit", "code"),
    [
        ("import time;time.sleep(60)", 0.1, 1024, "timeout"),
        ("import sys;sys.stdout.buffer.write(b'x'*4096)", 2.0, 32, "output_limit"),
    ],
)
def test_bounded_runner_kills_and_reaps_on_bound_violation(
    tmp_path: Path,
    script: str,
    timeout: float,
    limit: int,
    code: str,
) -> None:
    runner = D1BoundedProcessRunner(timeout_seconds=timeout, max_output_bytes=limit)

    with pytest.raises(D1BoundedProcessError) as caught:
        runner.run(
            (sys.executable, "-c", script),
            cwd=tmp_path,
            input=b"",
            env=dict(os.environ),
        )

    assert caught.value.code == code
    assert caught.value.process_group_swept is True
    assert caught.value.process_reaped is True
    assert caught.value.output_bytes <= limit + 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0, "max_output_bytes": 1},
        {"timeout_seconds": float("nan"), "max_output_bytes": 1},
        {"timeout_seconds": float("inf"), "max_output_bytes": 1},
        {"timeout_seconds": 1, "max_output_bytes": 0},
    ],
)
def test_bounded_runner_rejects_invalid_bounds(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        D1BoundedProcessRunner(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("argv", "cwd", "input_payload", "environment"),
    [
        ((), Path("."), b"", {}),
        ((sys.executable,), "not-path", b"", {}),
        ((sys.executable,), Path("."), "not-bytes", {}),
        ((sys.executable,), Path("."), b"", []),
    ],
)
def test_bounded_runner_rejects_invalid_invocation_shapes(
    argv: object,
    cwd: object,
    input_payload: object,
    environment: object,
) -> None:
    runner = D1BoundedProcessRunner(timeout_seconds=1, max_output_bytes=1024)
    with pytest.raises((TypeError, ValueError)):
        runner.run(
            cast(tuple[str, ...], argv),
            cwd=cast(Path, cwd),
            input=cast(bytes, input_payload),
            env=cast(dict[str, str], environment),
        )


def test_bounded_error_never_retains_raw_output(tmp_path: Path) -> None:
    secret = "raw-output-secret-sentinel"  # pragma: allowlist secret
    runner = D1BoundedProcessRunner(timeout_seconds=2, max_output_bytes=32)

    with pytest.raises(D1BoundedProcessError) as caught:
        runner.run(
            (sys.executable, "-c", f"import sys;sys.stdout.write({secret!r}*64)"),
            cwd=tmp_path,
            input=b"",
            env=dict(os.environ),
        )

    rendered = "".join(traceback.format_exception(caught.value))
    assert secret not in rendered
    assert not hasattr(caught.value, "stdout")
    assert not hasattr(caught.value, "stderr")


def test_bounded_runner_owns_process_before_selector_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, subprocess.Popen[bytes]] = {}
    real_popen = runner_module.subprocess.Popen

    def spawn(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)  # type: ignore[arg-type]
        observed["process"] = process
        return process

    def fail_selector() -> object:
        raise RuntimeError("selector sentinel")

    monkeypatch.setattr(runner_module.subprocess, "Popen", spawn)
    monkeypatch.setattr(runner_module.selectors, "DefaultSelector", fail_selector)
    runner = D1BoundedProcessRunner(timeout_seconds=2, max_output_bytes=1024)
    try:
        with pytest.raises(RuntimeError, match="selector sentinel"):
            runner.run(
                (sys.executable, "-c", "import time;time.sleep(60)"),
                cwd=tmp_path,
                input=b"",
                env=dict(os.environ),
            )
        assert observed["process"].poll() is not None
    finally:
        process = observed.get("process")
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_selector_close_failure_cannot_replace_primary_or_skip_reap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, subprocess.Popen[bytes]] = {}
    real_popen = runner_module.subprocess.Popen
    real_selector = runner_module.selectors.DefaultSelector

    def spawn(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)  # type: ignore[arg-type]
        observed["process"] = process
        return process

    class _BrokenSelector:
        def __init__(self) -> None:
            self.delegate = real_selector()

        def __getattr__(self, name: str) -> object:
            if name == "select":
                raise RuntimeError("select sentinel")
            return getattr(self.delegate, name)

        def close(self) -> None:
            self.delegate.close()
            raise OSError("selector close sentinel")

    monkeypatch.setattr(runner_module.subprocess, "Popen", spawn)
    monkeypatch.setattr(runner_module.selectors, "DefaultSelector", _BrokenSelector)
    runner = D1BoundedProcessRunner(timeout_seconds=2, max_output_bytes=1024)
    try:
        with pytest.raises(RuntimeError, match="select sentinel") as caught:
            runner.run(
                (sys.executable, "-c", "import time;time.sleep(60)"),
                cwd=tmp_path,
                input=b"",
                env=dict(os.environ),
            )
        assert observed["process"].poll() is not None
        assert "selector cleanup also failed" in getattr(caught.value, "__notes__", ())
    finally:
        process = observed.get("process")
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_bounded_runner_refuses_unproven_clean_group_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = D1BoundedProcessRunner(timeout_seconds=2, max_output_bytes=1024)
    monkeypatch.setattr(runner, "_sweep_group", lambda _process: False)

    with pytest.raises(D1BoundedProcessError) as caught:
        runner.run(
            (sys.executable, "-c", "pass"),
            cwd=tmp_path,
            input=b"",
            env=dict(os.environ),
        )

    assert caught.value.code == "subprocess_failure"
    assert caught.value.process_group_swept is False


def test_bounded_runner_sweeps_descendant_after_clean_leader_exit(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    script = (
        "import subprocess,sys;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
        "open(sys.argv[1],'w').write(str(child.pid))"
    )
    runner = D1BoundedProcessRunner(timeout_seconds=2, max_output_bytes=1024)

    result = runner.run(
        (sys.executable, "-c", script, str(pid_path)),
        cwd=tmp_path,
        input=b"",
        env=dict(os.environ),
    )

    assert result.returncode == 0
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _process_running(child_pid):
            break
        time.sleep(0.01)
    else:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        pytest.fail("bounded runner left a descendant alive")
