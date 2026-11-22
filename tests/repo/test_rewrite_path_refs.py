"""Teeth for ``scripts/rewrite_path_refs.py``, the bulk path-reference rewriter.

The tool exists because a file move leaves prose behind: after the tests/
tree was split into folders, 159 references under ``docs/`` still named the
old flat paths. Every case below pins one way a bulk rewrite goes wrong:
touching a longer token that merely contains a moved path, writing in
dry-run, rewriting an excluded file, or reporting success over an empty map.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import rewrite_path_refs as tool

pytestmark = pytest.mark.unit

MAPPING = {
    "tests/test_a.py": "tests/unit/core/test_a.py",
    "tests/test_ab.py": "tests/repo/test_ab.py",
}


def test_path_references_are_rewritten_and_counted() -> None:
    text, count = tool.rewrite_text(
        "see tests/test_a.py and `tests/test_ab.py`.", tool.compile_rules(MAPPING)
    )
    assert text == "see tests/unit/core/test_a.py and `tests/repo/test_ab.py`."
    assert count == 2


@pytest.mark.parametrize(
    "untouched",
    [
        "xtests/test_a.py",  # a longer path that ends with a moved one
        "src/tests/test_a.py",
        "tests/test_a.pyc",  # a longer token that starts with a moved one
        "tests/test_a.py.bak",
        "tests/test_abc.py",  # a different file sharing a prefix
        "tests/unit/core/test_a.py",  # already rewritten
    ],
)
def test_tokens_that_only_contain_a_moved_path_are_left_alone(untouched: str) -> None:
    text, count = tool.rewrite_text(f"({untouched})", tool.compile_rules(MAPPING))
    assert (text, count) == (f"({untouched})", 0)


def test_a_node_id_suffix_survives_the_rewrite() -> None:
    text, _ = tool.rewrite_text("tests/test_a.py::test_x[case]", tool.compile_rules(MAPPING))
    assert text == "tests/unit/core/test_a.py::test_x[case]"


def test_rewrite_is_idempotent() -> None:
    rules = tool.compile_rules(MAPPING)
    once, _ = tool.rewrite_text("tests/test_a.py", rules)
    assert tool.rewrite_text(once, rules) == (once, 0)


def test_dotted_module_references_are_rewritten_only_on_request() -> None:
    source = "from tests.test_a import helper  # tests.test_abx stays"
    assert tool.rewrite_text(source, tool.compile_rules(MAPPING))[1] == 0
    text, count = tool.rewrite_text(source, tool.compile_rules(MAPPING, dotted=True))
    assert text == "from tests.unit.core.test_a import helper  # tests.test_abx stays"
    assert count == 1


def test_an_empty_mapping_is_refused_not_reported_clean() -> None:
    with pytest.raises(tool.EmptyMappingError):
        tool.compile_rules({})


def _tree(root: Path) -> None:
    (root / "docs" / "verdicts").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text("run tests/test_a.py\n")
    (root / "docs" / "verdicts" / "filed.md").write_text("cited tests/test_a.py\n")
    (root / "docs" / "clean.md").write_text("nothing here\n")
    (root / "docs" / "blob.bin").write_bytes(b"\xff\xfe tests/test_a.py")


def test_dry_run_reports_without_writing(tmp_path: Path) -> None:
    _tree(tmp_path)
    report = tool.run(
        tmp_path, [Path("docs")], tool.compile_rules(MAPPING), excludes=(), apply=False
    )
    assert report.total == 2
    assert (tmp_path / "docs" / "guide.md").read_text() == "run tests/test_a.py\n"


def test_apply_writes_and_honours_excludes_and_skips_binary(tmp_path: Path) -> None:
    _tree(tmp_path)
    report = tool.run(
        tmp_path,
        [Path("docs")],
        tool.compile_rules(MAPPING),
        excludes=("docs/verdicts/*",),
        apply=True,
    )
    assert report.rewritten == {"docs/guide.md": 1}
    assert report.excluded == {"docs/verdicts/filed.md": 1}
    assert (tmp_path / "docs" / "guide.md").read_text() == "run tests/unit/core/test_a.py\n"
    assert (tmp_path / "docs" / "verdicts" / "filed.md").read_text() == "cited tests/test_a.py\n"
    assert (tmp_path / "docs" / "blob.bin").read_bytes() == b"\xff\xfe tests/test_a.py"


def _git(repo: Path, *args: str) -> str:
    env = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    }
    identity = [
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "commit.gpgsign=false",
    ]
    done = subprocess.run(
        ["git", *identity, *args], cwd=repo, env=env, check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def test_mapping_is_derived_from_git_renames_between_two_revisions(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n" * 5)
    (tmp_path / "keep.txt").write_text("unmoved\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "tests" / "unit").mkdir()
    _git(tmp_path, "mv", "tests/test_a.py", "tests/unit/test_a.py")
    _git(tmp_path, "commit", "-q", "-m", "move")
    assert tool.mapping_from_git(tmp_path, base, "HEAD") == {
        "tests/test_a.py": "tests/unit/test_a.py"
    }


def test_cli_check_mode_exits_nonzero_while_stale_references_remain(tmp_path: Path) -> None:
    _tree(tmp_path)
    mapping = tmp_path / "map.json"
    mapping.write_text('{"tests/test_a.py": "tests/unit/core/test_a.py"}')
    common = [
        "--root",
        str(tmp_path),
        "--mapping-json",
        str(mapping),
        "--exclude",
        "docs/verdicts/*",
        "docs",
    ]
    assert tool.main(["--check", *common]) == 1
    assert tool.main(["--apply", *common]) == 0
    assert tool.main(["--check", *common]) == 0


@pytest.mark.parametrize("revision", ["--output=/tmp/x", "-M", ""])
def test_a_revision_that_git_would_read_as_an_option_is_refused(
    tmp_path: Path, revision: str
) -> None:
    with pytest.raises(ValueError, match="revision"):
        tool.mapping_from_git(tmp_path, revision, "HEAD")
