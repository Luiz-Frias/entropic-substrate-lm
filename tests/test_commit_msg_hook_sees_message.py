"""The commit-msg stage must receive the message file git hands it.

MEASURED 2026-10-08 (FOR-1633). The global ``exclude`` in
``.pre-commit-config.yaml`` carried ``\\.git/`` as its first alternative. prek
applies that pattern to every stage's file list, and the commit-msg stage's
only file is the path git passes to the hook: ``.git/COMMIT_EDITMSG`` when
``git commit`` starts at the repository root (gitdir relative), an absolute
path when it starts in a subdirectory or a linked worktree (gitdir absolute).
The anchored pattern matched the relative form only, so from the root both
commitizen and gitlint reported ``(no files to check) Skipped`` and four
over-length messages landed this sprint, while the same hook blocked the same
message from a subdirectory, from a worktree, and from a conductor control that
named the file by another path.

Git never stages anything under ``.git/``, so the entry did nothing for the
pre-commit and pre-push stages; it only ever matched hook-stage filenames.
Guard teeth: re-adding ``\\.git/|`` to the global exclude turns the first test
RED; removing the ``commit-msg`` stage from either hook turns the second RED.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _REPO_ROOT / ".pre-commit-config.yaml"

# The three message files git can hand a commit-msg hook, as git names them
# when the gitdir is relative (the repository-root case that was fail-open).
_MESSAGE_FILES = (".git/COMMIT_EDITMSG", ".git/MERGE_MSG", ".git/SQUASH_MSG")


def _config() -> dict[str, object]:
    loaded = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _hook(config: dict[str, object], hook_id: str) -> dict[str, object]:
    repos = config["repos"]
    assert isinstance(repos, list)
    for repo in repos:
        for hook in repo["hooks"]:
            if hook["id"] == hook_id:
                return hook
    raise AssertionError(f"hook {hook_id!r} not found in {_CONFIG.name}")


@pytest.mark.parametrize("message_file", _MESSAGE_FILES)
def test_global_exclude_keeps_the_commit_message_file(message_file: str) -> None:
    exclude = _config()["exclude"]
    assert isinstance(exclude, str)
    assert re.search(exclude, message_file) is None, (
        f"the global exclude filters {message_file!r} out of the commit-msg "
        "stage, so commitizen and gitlint never see the message (FOR-1633)"
    )


@pytest.mark.parametrize("hook_id", ("gitlint", "commitizen"))
def test_message_hooks_run_at_the_commit_msg_stage(hook_id: str) -> None:
    assert _hook(_config(), hook_id).get("stages") == ["commit-msg"]
