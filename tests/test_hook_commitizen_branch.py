"""Pins for the ``commitizen-branch`` pre-push hook wrapper (FOR-1541).

MEASURED 2026-09-22 and 2026-10-05: ``just ship-it`` runs
``gt submit --stack ... --always``. When the bottom branch of the stack is
already up to date on the remote, ``--always`` still pushes it, prek hands the
pre-push hooks ``PRE_COMMIT_FROM_REF == PRE_COMMIT_TO_REF`` for that ref, the
upstream hook runs ``cz check --rev-range X..X`` and cz exits 3 with
``No commit found with range: 'X..X'`` -- aborting the whole atomic push.

``scripts/hooks/cz_check_branch.py`` is the local replacement that treats an
empty range as "nothing to lint" and otherwise delegates verbatim. These tests
pin the wrapper contract and the config wiring. The real ``cz`` is never
called: a fake executable records its argv (selected through ``CZ_CHECK_BIN``,
the test-only override documented in the wrapper).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WRAPPER = _REPO_ROOT / "scripts" / "hooks" / "cz_check_branch.py"
_CONFIG = _REPO_ROOT / ".pre-commit-config.yaml"
_COMMITIZEN_REPO = "https://github.com/commitizen-tools/commitizen"


def _config() -> dict[str, Any]:
    loaded = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _repo_block(url: str) -> dict[str, Any]:
    blocks = [b for b in _config()["repos"] if b["repo"] == url]
    assert len(blocks) == 1, f"expected exactly one {url} block, found {len(blocks)}"
    return blocks[0]


def _local_hook(hook_id: str) -> dict[str, Any]:
    hooks = [
        h
        for b in _config()["repos"]
        if b["repo"] == "local"
        for h in b["hooks"]
        if h["id"] == hook_id
    ]
    assert len(hooks) == 1, f"expected exactly one local hook {hook_id!r}, found {len(hooks)}"
    return hooks[0]


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull},
    )
    return done.stdout.strip()


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name, encoding="utf-8")
    _git(repo, "add", name)
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--no-verify",
        "-m",
        f"chore: add {name}",
    )
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def two_commits(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    older = _commit(repo, "a.txt")
    newer = _commit(repo, "b.txt")
    return repo, older, newer


def _fake_cz(tmp_path: Path) -> tuple[Path, Path]:
    """A fake ``cz``: records argv one-per-line, exits with ``$FAKE_CZ_EXIT``."""
    record = tmp_path / "cz-argv.txt"
    exe = tmp_path / "fake-cz"
    exe.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" >> "$FAKE_CZ_RECORD"\nexit "${FAKE_CZ_EXIT:-0}"\n',
        encoding="utf-8",
    )
    exe.chmod(0o755)
    return exe, record


def _run(
    cwd: Path,
    from_ref: str | None,
    to_ref: str | None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PRE_COMMIT_FROM_REF", "PRE_COMMIT_TO_REF", "CZ_CHECK_BIN"}
    }
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    if from_ref is not None:
        env["PRE_COMMIT_FROM_REF"] = from_ref
    if to_ref is not None:
        env["PRE_COMMIT_TO_REF"] = to_ref
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, str(_WRAPPER)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


# --- (a) config pin -----------------------------------------------------------


def test_config_local_hook_replaces_upstream_commitizen_branch() -> None:
    hook = _local_hook("commitizen-branch")
    assert hook["entry"] == "python scripts/hooks/cz_check_branch.py"
    assert hook["language"] == "python"
    assert hook["stages"] == ["pre-push"]
    assert hook["always_run"] is True
    assert hook["pass_filenames"] is False


def test_config_local_hook_commitizen_version_tracks_the_repo_rev() -> None:
    rev = _repo_block(_COMMITIZEN_REPO)["rev"]
    deps = _local_hook("commitizen-branch")["additional_dependencies"]
    assert deps == [f"commitizen=={rev.removeprefix('v')}"]


def test_config_upstream_block_keeps_only_the_commit_msg_hook() -> None:
    ids = [h["id"] for h in _repo_block(_COMMITIZEN_REPO)["hooks"]]
    assert ids == ["commitizen"]


# --- (b) empty range ----------------------------------------------------------


def test_empty_range_from_equals_to_exits_zero_with_message(
    two_commits: tuple[Path, str, str],
) -> None:
    repo, _older, newer = two_commits
    done = _run(repo, newer, newer)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == (
        f"commitizen-branch: empty range {newer}..{newer} (ref already up to date), nothing to lint"
    )


def test_empty_range_never_invokes_cz(two_commits: tuple[Path, str, str], tmp_path: Path) -> None:
    repo, _older, newer = two_commits
    fake, record = _fake_cz(tmp_path)
    done = _run(
        repo,
        newer,
        newer,
        {"CZ_CHECK_BIN": str(fake), "FAKE_CZ_RECORD": str(record), "FAKE_CZ_EXIT": "3"},
    )
    assert done.returncode == 0, done.stderr
    assert not record.exists(), "cz must not run on an empty range"


def test_empty_range_from_newer_to_older_exits_zero(
    two_commits: tuple[Path, str, str], tmp_path: Path
) -> None:
    repo, older, newer = two_commits
    fake, record = _fake_cz(tmp_path)
    done = _run(
        repo,
        newer,
        older,
        {"CZ_CHECK_BIN": str(fake), "FAKE_CZ_RECORD": str(record), "FAKE_CZ_EXIT": "3"},
    )
    assert done.returncode == 0, done.stderr
    assert "empty range" in done.stdout
    assert not record.exists()


# --- (c) fail loud on unexpected invocation -------------------------------------


@pytest.mark.parametrize(
    ("from_ref", "to_ref"),
    [(None, None), ("abc", None), (None, "abc"), ("", "abc"), ("abc", "")],
)
def test_missing_or_empty_env_exits_two(
    two_commits: tuple[Path, str, str], from_ref: str | None, to_ref: str | None
) -> None:
    repo, _older, _newer = two_commits
    done = _run(repo, from_ref, to_ref)
    assert done.returncode == 2
    assert "PRE_COMMIT_FROM_REF" in done.stderr
    assert "PRE_COMMIT_TO_REF" in done.stderr


# --- (d) delegation -------------------------------------------------------------


@pytest.mark.parametrize("code", [0, 3])
def test_non_empty_range_delegates_verbatim_and_propagates_exit(
    two_commits: tuple[Path, str, str], tmp_path: Path, code: int
) -> None:
    repo, older, newer = two_commits
    fake, record = _fake_cz(tmp_path)
    done = _run(
        repo,
        older,
        newer,
        {"CZ_CHECK_BIN": str(fake), "FAKE_CZ_RECORD": str(record), "FAKE_CZ_EXIT": str(code)},
    )
    assert done.returncode == code, done.stderr
    assert record.read_text(encoding="utf-8").splitlines() == [
        "check",
        "--rev-range",
        f"{older}..{newer}",
    ]


def test_unresolvable_range_is_delegated_not_swallowed(
    two_commits: tuple[Path, str, str], tmp_path: Path
) -> None:
    """If git cannot count the range, the wrapper cannot call it empty: cz decides."""
    repo, _older, newer = two_commits
    fake, record = _fake_cz(tmp_path)
    bogus = "0" * 40
    done = _run(
        repo,
        bogus,
        newer,
        {"CZ_CHECK_BIN": str(fake), "FAKE_CZ_RECORD": str(record), "FAKE_CZ_EXIT": "3"},
    )
    assert done.returncode == 3, done.stderr
    assert record.read_text(encoding="utf-8").splitlines() == [
        "check",
        "--rev-range",
        f"{bogus}..{newer}",
    ]


def test_cz_resolved_from_path_when_no_override(
    two_commits: tuple[Path, str, str], tmp_path: Path
) -> None:
    repo, older, newer = two_commits
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake, record = _fake_cz(tmp_path)
    (bindir / "cz").symlink_to(fake)
    path = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
    done = _run(
        repo, older, newer, {"PATH": path, "FAKE_CZ_RECORD": str(record), "FAKE_CZ_EXIT": "0"}
    )
    assert done.returncode == 0, done.stderr
    assert record.read_text(encoding="utf-8").splitlines() == [
        "check",
        "--rev-range",
        f"{older}..{newer}",
    ]
