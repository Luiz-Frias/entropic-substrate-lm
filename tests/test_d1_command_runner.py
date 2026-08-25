"""TDD contract for the shared shell-free D1 command runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

import entropic_substrate.d1_command_runner as runner_module
from entropic_substrate.d1_command_runner import SubprocessD1CommandRunner

pytestmark = pytest.mark.unit


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


def test_runner_uses_shell_free_checked_text_subprocess_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _Completed:
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
    assert observed["check"] is True
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


def test_runner_propagates_subprocess_failure_without_fallback(tmp_path: Path) -> None:
    script = "import sys; sys.stderr.write('boom'); raise SystemExit(3)"

    with pytest.raises(subprocess.CalledProcessError) as captured:
        SubprocessD1CommandRunner().run((sys.executable, "-c", script), cwd=tmp_path, stdin=None)

    assert captured.value.returncode == 3
