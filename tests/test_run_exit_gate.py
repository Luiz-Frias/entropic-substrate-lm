"""Tests for ``scripts/run_exit_gate.py`` — the deterministic evidence-producer
CLI ported from AGX's ``run-exit-gate.ts``.

Ported test-first from
``/tmp/entropic-port-ref-04a8fc9c/exit-gate.test.ts``'s two CLI-focused
``describe`` blocks:

* ``describe("run-exit-gate CLI (deterministic evidence producer)")`` — 11
  cases, ``TestRunExitGateCli`` below.
* ``describe("run-exit-gate isolation: linked worktrees + hostile hook
  env")`` — 3 cases (the documented 2026-07-13 ``core.worktree`` poisoning
  incident class), ``TestRunExitGateIsolation`` below.

``tests/test_exit_gate.py``'s own module docstring previously (incorrectly —
see round-2 review) deferred these 14 cases as "deliberately no Python
counterpart, not a missed case". They are ported here instead, against
``scripts/run_exit_gate.py`` (the Python sibling of ``run-exit-gate.ts``,
which itself imports ``scripts/exit_gate.py``'s typestate core rather than
duplicating it).

BLACK-BOX BY DESIGN, matching the reference: every case spawns a real
``python scripts/run_exit_gate.py`` child process (itself running several
``git`` subprocess calls) via ``subprocess.run`` and asserts on its stdout
JSON + exit code — never imports ``run_exit_gate`` as a module. This mirrors
``exit-gate.test.ts``'s own choice to spawn a real ``bun scripts/run-exit-gate.ts``
child rather than call its internals directly: the whole point of "the agent
is a courier, never a witness" is that the runner's git reads and process
exit code are the evidence, so the test needs to observe the same real
subprocess boundary a harness would.

ORDERING NOTE: ``TestRunExitGateCli`` shares ONE temp git repo across its
methods (module-scoped fixture), matching ``exit-gate.test.ts``'s own
``beforeAll``-created repo shared across its ``it()`` cases in the same
``describe`` block. Exactly one method
(``test_detects_tip_moved_mid_gate_commit_inside_gate_command``) advances that
repo's HEAD permanently; every other method reads the current tip fresh at
assertion time (never hardcodes an expected SHA), so method execution order
does not affect correctness — this repo has no ``pytest-randomly`` installed
(verified: ``python -c "import pytest_randomly"`` fails), so methods run in
declaration order regardless, but the tests are written to not depend on
that.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNNER = _REPO_ROOT / "scripts" / "run_exit_gate.py"

_SHA_HEX_LEN = 40
_OTHER_TIP = "b" * _SHA_HEX_LEN
# Mirrors exit-gate.test.ts's own SUBPROCESS_TEST_TIMEOUT_MS budget rationale
# (a real `git`-spawning child under CPU contention needs real headroom, not
# vitest's/pytest's tight default) -- scaled down since this port's child
# does far less work than bun's cold start + several git subprocess calls
# under prek's own concurrent hook load.
_SUBPROCESS_TIMEOUT_S = 30


def _git_safe_env(home: Path) -> dict[str, str]:
    """Mirrors exit-gate.test.ts's ``gitSafeEnv()``: strip every inherited
    ``GIT_*`` var (contagious under a hook env -- see run_exit_gate.py's ENV
    QUARANTINE note), then re-add only deliberate isolation knobs, so the
    scratch git commands THIS TEST issues (not the runner-under-test, which
    does its own independent filtering) can't wander into a real user
    gitconfig or a parent repo."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CEILING_DIRECTORIES"] = tempfile.gettempdir()
    env["HOME"] = str(home)
    env["XDG_CONFIG_HOME"] = str(home)
    return env


def _git(cwd: Path, *args: str, home: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_git_safe_env(home if home is not None else cwd),
        check=True,
    )
    return proc.stdout.strip()


def _run_runner(
    cwd: Path, args: list[str], env: dict[str, str] | None = None
) -> tuple[int, dict[str, Any]]:
    proc = subprocess.run(
        [sys.executable, str(_RUNNER), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_S,
        env=env if env is not None else _git_safe_env(cwd),
    )
    try:
        payload: dict[str, Any] = json.loads(proc.stdout)
    except json.JSONDecodeError as cause:
        msg = (
            f"runner emitted non-JSON stdout (status {proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
        raise AssertionError(msg) from cause
    return proc.returncode, payload


# ─── CLI runner component tests (temp git repo; the runner reads HEAD itself) ─


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("exit-gate-cli-")
    _git(path, "init", "-q")
    _git(
        path,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "--allow-empty",
        "-m",
        "init",
        "-q",
    )
    return path


def _current_tip(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


class TestRunExitGateCli:
    """Mirrors exit-gate.test.ts's describe("run-exit-gate CLI ...")."""

    def test_certifies_a_passing_gate_exit_0_cert_bound_to_head(self, repo: Path) -> None:
        status, payload = _run_runner(repo, ["--cmd", "echo gate-green"])
        assert status == 0
        assert payload["ok"] is True
        assert payload["cert"]["certifiedTip"] == _current_tip(repo)
        assert payload["evidence"]["exitCode"] == 0
        assert "gate-green" in payload["evidence"]["tail"]
        assert payload["evidence"]["dirty"] is False

    def test_fails_a_failing_gate_nonzero_exit_carries_the_real_code(self, repo: Path) -> None:
        status, payload = _run_runner(repo, ["--cmd", "echo boom; exit 3"])
        assert status == 1
        assert payload["ok"] is False
        assert payload["error"]["kind"] == "NONZERO_EXIT"
        assert payload["error"]["exitCode"] == 3
        assert "boom" in payload["evidence"]["tail"]

    def test_refuses_a_silent_success_empty_tail(self, repo: Path) -> None:
        status, payload = _run_runner(repo, ["--cmd", "true"])
        assert status == 1
        assert payload["ok"] is False
        assert payload["error"]["kind"] == "EMPTY_TAIL"

    def test_fails_fast_on_expect_tip_mismatch_without_running_the_gate(self, repo: Path) -> None:
        marker = repo / "must-not-exist"
        status, payload = _run_runner(
            repo, ["--cmd", f"touch {marker}", "--expect-tip", _OTHER_TIP]
        )
        assert status == 1
        assert payload["ok"] is False
        assert payload["error"]["kind"] == "STALE_TIP"
        assert not marker.exists()

    def test_detects_tip_moved_mid_gate_commit_inside_gate_command(self, repo: Path) -> None:
        """The one method in this class that advances the shared repo's HEAD
        permanently -- see the module docstring's ORDERING NOTE."""
        status, payload = _run_runner(
            repo,
            [
                "--cmd",
                "echo output; git -c user.email=t@t -c user.name=t commit --allow-empty -m moved -q",
            ],
        )
        assert status == 1
        assert payload["ok"] is False
        assert payload["error"]["kind"] == "STALE_TIP"

    def test_refuses_a_dirty_post_run_tree(self, repo: Path) -> None:
        autofixed = repo / "autofixed.txt"
        try:
            status, payload = _run_runner(repo, ["--cmd", "echo fixed > autofixed.txt"])
            assert status == 1
            assert payload["ok"] is False
            assert payload["error"]["kind"] == "DIRTY_TREE"
        finally:
            autofixed.unlink(missing_ok=True)

    def test_writes_the_evidence_json_to_out_for_the_audit_ledger(self, repo: Path) -> None:
        out = repo / "evidence.json"
        try:
            status, _payload = _run_runner(repo, ["--cmd", "echo gate-green", "--out", str(out)])
            assert status == 0
            written = json.loads(out.read_text())
            assert written["ok"] is True
            assert written["cert"]["certifiedTip"] == _current_tip(repo)
        finally:
            out.unlink(missing_ok=True)

    def test_errs_missing_evidence_when_cmd_is_absent(self, repo: Path) -> None:
        status, payload = _run_runner(repo, [])
        assert status == 1
        assert payload["ok"] is False
        assert payload["error"]["kind"] == "MISSING_EVIDENCE"

    def test_fail_fast_reports_omit_the_evidence_key(self, repo: Path) -> None:
        """Never ``evidence: null`` -- the key is entirely absent, so
        object-typed schemas on the relaying side accept the report."""
        _status, payload = _run_runner(repo, ["--cmd", "echo x", "--expect-tip", _OTHER_TIP])
        assert payload["ok"] is False
        assert "evidence" not in payload

    def test_non_git_cwd_emits_structured_runner_error_no_stack_trace(self, tmp_path: Path) -> None:
        bare = tmp_path / "no-repo"
        bare.mkdir()
        status, payload = _run_runner(bare, ["--cmd", "echo x"])
        assert status == 1
        assert payload["ok"] is False
        assert payload["error"]["kind"] == "RUNNER_ERROR"

    def test_unknown_cli_flag_emits_structured_runner_error(self, repo: Path) -> None:
        status, payload = _run_runner(repo, ["--cmd", "echo x", "--bogus-flag"])
        assert status == 1
        assert payload["ok"] is False
        assert payload["error"]["kind"] == "RUNNER_ERROR"


# ─── linked-worktree + hook-env isolation (the core.worktree poisoning class) ─
# Incident 2026-07-13: prek hands hooks GIT_DIR + GIT_WORK_TREE; a git init
# that inherits them re-initializes the HOST repo and writes core.worktree
# into the SHARED .git/config (linked-worktree config writes land in the
# common config). The runner must resolve the repo from its INVOKING
# DIRECTORY, immune to a polluted caller env, and must never let the gate
# child inherit GIT_* either.


@pytest.fixture(scope="module")
def worktrees(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("exit-gate-wt-")
    main_repo = root / "main"
    linked_wt = root / "wt"
    decoy_repo = root / "decoy"

    # Round-3 review finding (BLOCKING): distinct commit messages per repo, not
    # a shared "init" -- two `git commit --allow-empty` calls under the same
    # identity in the same second, same empty tree, same parent (none), and
    # the SAME message previously produced byte-identical commit objects (and
    # therefore the SAME SHA) for main and decoy. That collision made
    # test_resolves_repo_from_cwd_not_inherited_git_dir_or_work_tree vacuous:
    # `payload["evidence"]["testedTip"] == decoy_tip` held whether the runner
    # read the decoy repo (correct) or the poisoned GIT_DIR target (the
    # incident), because both were the same hash. Distinct messages make the
    # two repos' initial commits genuinely different objects.
    for r, message in ((main_repo, "init-main"), (decoy_repo, "init-decoy")):
        subprocess.run(
            ["git", "init", "-q", str(r)],
            check=True,
            capture_output=True,
            text=True,
            env=_git_safe_env(root),
        )
        _git(
            r,
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-m",
            message,
            "-q",
            home=root,
        )

    _git(main_repo, "worktree", "add", "-q", "--detach", str(linked_wt), home=root)
    # advance main past the worktree checkout so main HEAD != worktree HEAD
    _git(
        main_repo,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "--allow-empty",
        "-m",
        "ahead",
        "-q",
        home=root,
    )

    # Anti-vacuity guard (round-3 review, BLOCKING): assert the two tips a
    # poisoned-vs-correct runner would return actually DIFFER, so a future
    # fixture edit that re-collides them fails LOUDLY here instead of
    # silently hollowing out test_resolves_repo_from_cwd_not_inherited_git_dir_or_work_tree
    # again with no signal. `wt_tip` is what a GIT_DIR-poisoned read of the
    # linked worktree returns (see _hostile_env); `decoy_tip` is what the
    # test asserts the runner actually certified.
    wt_tip = _git(linked_wt, "rev-parse", "HEAD", home=root)
    decoy_tip = _git(decoy_repo, "rev-parse", "HEAD", home=root)
    assert wt_tip != decoy_tip, (
        "anti-vacuity guard tripped: the linked worktree's HEAD and the decoy "
        "repo's HEAD collided again, which would silently make "
        "test_resolves_repo_from_cwd_not_inherited_git_dir_or_work_tree unable "
        "to fail -- give the two repos' commits a further distinguishing "
        "input (message/author-date) until this assertion holds"
    )

    return {"root": root, "main": main_repo, "wt": linked_wt, "decoy": decoy_repo}


def _hostile_env(root: Path, main_repo: Path, linked_wt: Path) -> dict[str, str]:
    """Exactly what prek hands hooks during a commit in the linked worktree."""
    env = _git_safe_env(root)
    env["GIT_DIR"] = str(main_repo / ".git" / "worktrees" / "wt")
    env["GIT_WORK_TREE"] = str(linked_wt)
    env["GIT_INDEX_FILE"] = str(main_repo / ".git" / "worktrees" / "wt" / "index")
    return env


class TestRunExitGateIsolation:
    """Mirrors exit-gate.test.ts's describe("run-exit-gate isolation:
    linked worktrees + hostile hook env")."""

    def test_resolves_repo_from_cwd_not_inherited_git_dir_or_work_tree(
        self, worktrees: dict[str, Path]
    ) -> None:
        root, main_repo, linked_wt, decoy = (
            worktrees["root"],
            worktrees["main"],
            worktrees["wt"],
            worktrees["decoy"],
        )
        decoy_tip = _git(decoy, "rev-parse", "HEAD", home=root)
        _status, payload = _run_runner(
            decoy,
            ["--cmd", "echo gate-green"],
            env=_hostile_env(
                root, main_repo, linked_wt
            ),  # points at mainRepo's worktree -- must be ignored
        )
        assert payload["evidence"]["testedTip"] == decoy_tip

    def test_leaves_shared_git_config_byte_identical_after_gate_child_git_init(
        self, worktrees: dict[str, Path]
    ) -> None:
        root, main_repo, linked_wt = worktrees["root"], worktrees["main"], worktrees["wt"]
        shared_config = main_repo / ".git" / "config"
        before = shared_config.read_text()
        scratch = root / "scratch-init"
        try:
            _status, payload = _run_runner(
                linked_wt,
                # the incident shape: a gate-child `git init` in a scratch dir
                # under hook env
                ["--cmd", f"mkdir -p {scratch} && cd {scratch} && git init -q && echo init-done"],
                env=_hostile_env(root, main_repo, linked_wt),
            )
            after = shared_config.read_text()
            assert after == before
            assert "worktree =" not in after
            assert "init-done" in payload["evidence"]["tail"]
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_certifies_the_linked_worktrees_own_head_under_hostile_env(
        self, worktrees: dict[str, Path]
    ) -> None:
        root, main_repo, linked_wt = worktrees["root"], worktrees["main"], worktrees["wt"]
        wt_tip = _git(linked_wt, "rev-parse", "HEAD", home=root)
        status, payload = _run_runner(
            linked_wt, ["--cmd", "echo gate-green"], env=_hostile_env(root, main_repo, linked_wt)
        )
        assert status == 0
        assert payload["cert"]["certifiedTip"] == wt_tip
