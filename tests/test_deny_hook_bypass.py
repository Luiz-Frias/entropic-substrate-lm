"""Black-box teeth for ``.claude/hooks/deny-hook-bypass.sh`` (FOR-1536).

The hook is a PreToolUse(Bash) channel: it reads the tool call as JSON on
stdin and prints a deny decision when a ``git commit``/``git push`` carries a
verification skip or a ``core.hooksPath`` override. Each case below pipes a
real payload through the real script under ``sh`` -- the same way Claude
Code invokes it -- so the assertion is on the decision the hook emits, not on
a regex read in isolation.

The quoted and escaped rows are the CodeRabbit finding on PR #291 (2026-09-22):
the first form matched raw command text, so a quoted skip flag and a quoted
``core.hooksPath`` override slipped past it while git still received the
hook-disabling argument. The hook now strips shell quoting and backslashes
before matching; these rows go RED if that normalization is removed (proven
by mutation before landing).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "deny-hook-bypass.sh"
_SKIP = "--no-" + "verify"  # split so this file never carries the literal flag

_DENIED = (
    f"git commit {_SKIP} -m x",
    "cd /r && git commit -n -F msg",
    "git -c core.hooksPath=/dev/null commit -F m",
    f"git push {_SKIP} origin main",
    # quoted / escaped forms (CodeRabbit, PR #291)
    f"git commit '{_SKIP}' -m x",
    f'git commit "{_SKIP}" -m x',
    "git -c 'core.hooksPath=/dev/null' commit -F m",
    'git -c "core.hooksPath=/dev/null" commit -F m',
    "git commit --no\\-verify -m x",
    "git commit -F m '-n'",
)

_ALLOWED = (
    "git commit -F msg",
    f'echo "never use {_SKIP}"',
    "git push origin main",
    'git commit -F "$1" 2>&1 | tail -3; if [ -n "$(git status --porcelain)" ]; then echo retry; fi',
    "grep -n verify tests/test_pc_ci_gitlint.py",
    "git log -n 3",
    "ls",
)


def _decision(command: str) -> str | None:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    proc = subprocess.run(
        ["sh", str(_HOOK)], input=payload, capture_output=True, text=True, check=True
    )
    if not proc.stdout.strip():
        return None
    out = json.loads(proc.stdout)
    return str(out["hookSpecificOutput"]["permissionDecision"])


@pytest.mark.unit
@pytest.mark.parametrize("command", _DENIED)
def test_hook_denies_verification_skips(command: str) -> None:
    assert _decision(command) == "deny", command


@pytest.mark.unit
@pytest.mark.parametrize("command", _ALLOWED)
def test_hook_allows_ordinary_git_and_mentions(command: str) -> None:
    assert _decision(command) is None, command
