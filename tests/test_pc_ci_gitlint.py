"""FOR-1536: the exit gate lints every commit message in the branch range.

MEASURED 2026-09-21 (p3b, wf_8a5c8008-50e): ``prek run --from-ref/--to-ref``
fires the pre-commit and pre-push stages only; gitlint is registered for the
commit-msg stage, so a commit made with hooks skipped carried an 84-char title
all the way to ``gt submit``. Two arms: the recipe text names the range lint
(static; the justfile is the gate's only source), and gitlint itself, invoked
the way the recipe invokes it, goes RED on an over-length title and GREEN on a
conforming one (behavioural; proves the tool and its arguments, not the prose).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


def test_pc_ci_recipe_lints_the_commit_range_with_gitlint() -> None:
    text = (REPO / "justfile").read_text()
    recipe = text[text.index("pc-ci base=") :]
    recipe = recipe[: recipe.index("\n\n")]
    # 2026-09-22: lints the UNPUBLISHED commits only (a published ancestor with a bad
    # message is past the point this gate exists for and may never be rewritten).
    assert "git rev-list HEAD --not --remotes" in recipe, recipe
    assert re.search(r'uv run gitlint --commit "\$\{sha\}"', recipe), recipe


def _git(cwd: Path, *args: str, env: dict[str, str]) -> None:
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)


@pytest.mark.parametrize(
    ("title", "expect_ok"),
    [
        ("fix(gate): a conforming title", True),
        ("fix(gate): " + "x" * 70, False),
    ],
)
def test_gitlint_over_a_range_judges_titles(tmp_path: Path, title: str, expect_ok: bool) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "HOME": str(tmp_path),
    }
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", env=env)
    (repo / "a").write_text("1\n")
    _git(repo, "add", "a", env=env)
    _git(repo, "commit", "-q", "-m", "chore: base", env=env)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()
    (repo / "a").write_text("2\n")
    _git(repo, "add", "a", env=env)
    _git(repo, "commit", "-q", "-m", title + "\n\nbody line under eighty characters.", env=env)
    result = subprocess.run(
        [str(Path(sys.executable).parent / "gitlint"), "--commits", f"{base}..HEAD"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is expect_ok, result.stderr + result.stdout
    if not expect_ok:
        assert "T1 Title exceeds max length" in result.stderr + result.stdout
