"""Contracts for ``scripts/derive_compose_project.sh`` (p5, defect #1).

These are BEHAVIOURAL, not textual: every assertion here runs the real
script as a real subprocess and inspects what it actually printed. A test
that only greps run_container_test.sh's source text for a token (the style
``tests/test_run_container_test_config.py`` otherwise uses throughout) would
be vacuous for this specific defect — the acceptance brief is explicit that
"a test that merely checks a suffix exists is vacuous, because the existing
path hash already provides one and a single run cannot distinguish per-run
from per-worktree entropy." Only two REAL invocations, compared, can show
that.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "derive_compose_project.sh"

_FAMILY_RE = re.compile(r"^entropic-substrate-lm-mlflow-test-[a-z0-9_-]+-\d+$")


def _derive() -> dict[str, str]:
    """Run the script as a fresh subprocess and parse its ``KEY=value`` output."""
    result = subprocess.run(  # nosec B603 B607
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        timeout=15,
    )
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        if key and value:
            parsed[key] = value
    return parsed


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.exists(), "scripts/derive_compose_project.sh is missing"
    assert SCRIPT.stat().st_mode & 0o111, (
        "scripts/derive_compose_project.sh must be executable (chmod +x)"
    )


def test_prints_both_family_and_project_on_direct_execution() -> None:
    parsed = _derive()
    assert "PROJECT_FAMILY" in parsed, parsed
    assert "PROJECT" in parsed, parsed


def test_family_matches_the_expected_charset_and_shape() -> None:
    parsed = _derive()
    assert _FAMILY_RE.match(parsed["PROJECT_FAMILY"]), parsed["PROJECT_FAMILY"]


def test_project_extends_family_with_a_dash_separated_suffix() -> None:
    """PROJECT must be built ON TOP OF the family, not an unrelated string —
    the guard's family-prefix match (tests/helpers.py) depends on this."""
    parsed = _derive()
    family, project = parsed["PROJECT_FAMILY"], parsed["PROJECT"]
    assert project.startswith(f"{family}-"), (family, project)


# --------------------------------------------------------- THE acceptance test
def test_two_invocations_in_the_same_worktree_share_family_but_differ_in_project() -> None:
    """Acceptance #3, verbatim: derive the project name TWICE from the SAME
    worktree, in the SAME second, and show the two differ.

    NOT vacuous: this does not merely check that PROJECT contains *some*
    suffix (the pre-existing PATH_HASH already guarantees that, and would
    make this test pass even with ZERO per-run entropy — exactly the defect
    being fixed). It positively proves TWO THINGS in the same run:

    1. PROJECT_FAMILY is IDENTICAL across both invocations — the per-WORKTREE
       component is still stable, so the guard's family-based matching
       (defects #2/#3) keeps working.
    2. PROJECT differs across both invocations DESPITE an identical family —
       proving the divergence is genuinely PER-RUN, not an artefact of the
       worktree path.

    Two SEPARATE OS processes (not two calls to a Python helper) so each
    gets its own PID — the actual entropy source — and the timestamp
    component is free to coincide (same wall-clock second) without the test
    becoming a no-op, which is exactly the near-miss defect #1 describes.
    """
    first = _derive()
    second = _derive()

    assert first["PROJECT_FAMILY"] == second["PROJECT_FAMILY"], (first, second)
    assert first["PROJECT"] != second["PROJECT"], (
        "two invocations from the SAME worktree derived the SAME compose "
        f"project ({first['PROJECT']!r}) — per-run entropy is not working. "
        f"Full derivations: {first} / {second}"
    )


def test_the_pid_component_is_what_actually_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the ENTROPY SOURCE, not just its effect: strip the shared family
    and shared-if-unlucky timestamp off both PROJECT values and show the
    remaining PID-shaped component is what diverges. Without this, a future
    edit could satisfy the test above by concatenating something else
    per-run (e.g. a counter file) that has its own hazards (a stale counter
    surviving a crashed run) without anyone noticing the entropy source
    changed.
    """
    first = _derive()
    second = _derive()

    family = first["PROJECT_FAMILY"]
    assert family == second["PROJECT_FAMILY"], (first, second)

    suffix1 = first["PROJECT"].removeprefix(f"{family}-")
    suffix2 = second["PROJECT"].removeprefix(f"{family}-")
    # "<pid>-<epoch-seconds>"
    pid1, _, ts1 = suffix1.partition("-")
    pid2, _, ts2 = suffix2.partition("-")
    assert pid1.isdigit() and pid2.isdigit(), (suffix1, suffix2)
    assert ts1.isdigit() and ts2.isdigit(), (suffix1, suffix2)
    assert pid1 != pid2, (
        f"PID component did not differ between two subprocess invocations "
        f"({pid1!r} == {pid2!r}) — that should never happen for two distinct "
        f"OS processes and would mean the entropy source silently changed"
    )
