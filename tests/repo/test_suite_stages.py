"""Teeth for ``scripts/suite_stages.py``: which tests each stage runs, and the
record that lets a non-blocking stage still gate a push.

The layout rule is: the folder says what KIND a test is, a marker says what it
COSTS. Stages select by folder path and filter by marker. The hazards pinned
here are a folder no stage ever runs, a post-commit result nobody reads, and a
record store that grows without bound.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import suite_stages as stages

import repo_paths

pytestmark = pytest.mark.unit

TREE_A = "a" * 40
TREE_B = "b" * 40
TREE_C = "c" * 40


def _folders_with_tests() -> set[str]:
    return {
        path.relative_to(repo_paths.TESTS_ROOT).parts[0]
        for path in repo_paths.TESTS_ROOT.rglob("test_*.py")
        if len(path.relative_to(repo_paths.TESTS_ROOT).parts) > 1
    }


def test_no_test_file_sits_directly_under_tests() -> None:
    assert sorted(p.name for p in repo_paths.TESTS_ROOT.glob("test_*.py")) == []


def test_every_test_folder_is_run_in_full_by_some_stage() -> None:
    full = {
        Path(path).name
        for stage in stages.STAGES.values()
        if stage.marker is None or "slow" not in stage.marker
        for path in stage.paths
    }
    assert _folders_with_tests() - full - set(stages.FULL_TIER_ONLY) == set()


def test_folders_left_to_the_full_tier_are_named_and_exist() -> None:
    assert set(stages.FULL_TIER_ONLY) == {"live"}
    assert set(stages.FULL_TIER_ONLY) <= _folders_with_tests()


def test_every_stage_path_exists() -> None:
    missing = [
        path
        for stage in stages.STAGES.values()
        for path in stage.paths
        if not (repo_paths.REPO_ROOT / path).is_dir()
    ]
    assert missing == []


def test_the_commit_stage_is_the_fast_unit_folder_only() -> None:
    stage = stages.STAGES["pre-commit"]
    assert stage.paths == ("tests/unit",)
    assert stage.marker == "not slow"


def test_the_push_stage_runs_unit_and_e2e_including_slow() -> None:
    stage = stages.STAGES["pre-push"]
    assert set(stage.paths) == {"tests/unit", "tests/e2e"}
    assert stage.marker is None


def test_what_post_commit_runs_fast_post_ship_runs_in_full_or_push_does() -> None:
    fast = set(stages.STAGES["post-commit"].paths)
    later = set(stages.STAGES["post-ship"].paths) | set(stages.STAGES["pre-push"].paths)
    assert fast <= later
    assert stages.STAGES["post-commit"].marker == "not slow"
    assert stages.STAGES["post-ship"].marker is None


def test_pytest_argv_keeps_the_parallel_flags_the_suite_is_measured_under() -> None:
    argv = stages.pytest_argv(stages.STAGES["pre-commit"])
    assert argv[:1] == ["tests/unit"]
    assert argv[argv.index("-m") + 1] == "not slow"
    assert argv[argv.index("-n") + 1] == "auto"
    assert argv[argv.index("--dist") + 1] == "loadfile"
    assert "-m" not in stages.pytest_argv(stages.STAGES["pre-push"])


def test_a_record_round_trips_and_is_keyed_by_tree(tmp_path: Path) -> None:
    stages.write_record(
        tmp_path, stages.Record(tree=TREE_A, stage="post-commit", passed=True, summary="3 passed")
    )
    assert stages.read_record(tmp_path, TREE_A) == stages.Record(
        TREE_A, "post-commit", True, "3 passed"
    )
    assert stages.read_record(tmp_path, TREE_B) is None


@pytest.mark.parametrize(
    "body",
    [
        "",
        "{",
        "[]",
        '{"tree": "x"}',
        json.dumps({"tree": TREE_B, "stage": "s", "passed": "yes", "summary": ""}),
    ],
)
def test_an_unreadable_or_mismatched_record_counts_as_absent_never_as_a_pass(
    tmp_path: Path, body: str
) -> None:
    (tmp_path / f"{TREE_A}.json").write_text(body)
    assert stages.read_record(tmp_path, TREE_A) is None


def test_verify_accepts_a_passing_record_without_rerunning(tmp_path: Path) -> None:
    stages.write_record(tmp_path, stages.Record(TREE_A, "post-commit", True, "ok"))
    calls: list[str] = []
    assert stages.verify(tmp_path, TREE_A, rerun=lambda: calls.append("ran") or True) is True
    assert calls == []


def test_verify_refuses_a_failing_record_without_rerunning(tmp_path: Path) -> None:
    stages.write_record(tmp_path, stages.Record(TREE_A, "post-commit", False, "1 failed"))
    calls: list[str] = []
    assert stages.verify(tmp_path, TREE_A, rerun=lambda: calls.append("ran") or True) is False
    assert calls == []


@pytest.mark.parametrize("outcome", [True, False])
def test_verify_runs_the_stage_itself_when_no_record_exists(tmp_path: Path, outcome: bool) -> None:
    calls: list[str] = []

    def rerun() -> bool:
        calls.append("ran")
        return outcome

    assert stages.verify(tmp_path, TREE_A, rerun=rerun) is outcome
    assert calls == ["ran"]


def test_prune_keeps_only_records_for_live_trees_and_caps_the_rest(tmp_path: Path) -> None:
    for tree in (TREE_A, TREE_B, TREE_C):
        stages.write_record(tmp_path, stages.Record(tree, "post-commit", True, "ok"))
    (tmp_path / "stray.txt").write_text("not a record")
    removed = stages.prune(tmp_path, keep={TREE_B})
    assert removed == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{TREE_B}.json", "stray.txt"]


def test_prune_on_a_missing_store_is_a_no_op(tmp_path: Path) -> None:
    assert stages.prune(tmp_path / "absent", keep=set()) == 0


def _pointer(root: Path, rel: str, body: str) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(body)


def test_export_links_dvc_data_only_where_the_pointer_matches_the_commit(tmp_path: Path) -> None:
    live, export = tmp_path / "live", tmp_path / "export"
    for root in (live, export):
        _pointer(root, "results/same.jsonl.dvc", "md5: 111\n")
        _pointer(root, "results/absent.jsonl.dvc", "md5: 333\n")
    _pointer(live, "results/drifted.jsonl.dvc", "md5: 222-live\n")
    _pointer(export, "results/drifted.jsonl.dvc", "md5: 222-commit\n")
    (live / "results" / "same.jsonl").write_text("payload")
    (live / "results" / "drifted.jsonl").write_text("other payload")

    linked = stages.link_dvc_data(live, export)

    assert linked == ["results/same.jsonl"]
    assert (export / "results" / "same.jsonl").read_text() == "payload"
    assert (export / "results" / "same.jsonl").is_symlink()
    assert not (export / "results" / "drifted.jsonl").exists(), (
        "data for a different pointer must not be linked"
    )
    assert not (export / "results" / "absent.jsonl").exists()


# --- the commit stage is scoped to the folders a commit touches -------------
#
# tests/unit mirrors src/entropic_substrate, so a staged path names the unit
# folder that covers it. Everything the scope leaves out still runs, in full,
# at pre-push.


def _scope(*changed: str) -> tuple[str, ...]:
    return stages.unit_scope(changed, root=repo_paths.REPO_ROOT)


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ("src/entropic_substrate/core/sentence/x.py", ("tests/unit/core/sentence",)),
        ("src/entropic_substrate/ladder/run/worker/x.py", ("tests/unit/ladder/run",)),
        ("tests/unit/ladder/capture/reader/test_x.py", ("tests/unit/ladder/capture",)),
        # a module directly under a top package covers that whole package
        ("src/entropic_substrate/kernel/lifecycle.py", ("tests/unit/kernel",)),
        ("tests/unit/kernel/test_x.py", ("tests/unit/kernel",)),
    ],
)
def test_a_mirrored_path_scopes_to_its_unit_folder(changed: str, expected: tuple[str, ...]) -> None:
    assert _scope(changed) == expected


@pytest.mark.parametrize(
    "changed",
    [
        "src/entropic_substrate/__init__.py",
        "tests/conftest.py",
        "tests/support/repo_paths.py",
        "tests/fixtures/anything.json",
        "pyproject.toml",
        "uv.lock",
        "some/path/nobody/classified.py",
    ],
)
def test_a_shared_or_unclassified_path_widens_to_the_whole_unit_folder(changed: str) -> None:
    assert _scope("src/entropic_substrate/core/sentence/x.py", changed) == stages.FULL_UNIT


@pytest.mark.parametrize(
    "changed",
    ["docs/architecture.md", "README.md", "scripts/x.py", "tests/repo/test_x.py", "justfile"],
)
def test_a_path_named_as_deferred_adds_nothing_to_the_commit_stage(changed: str) -> None:
    assert _scope(changed) == ()
    assert _scope(changed, "src/entropic_substrate/core/geometry/x.py") == (
        "tests/unit/core/geometry",
    )


def test_a_package_with_no_mirror_folder_widens_rather_than_running_nothing() -> None:
    assert _scope("src/entropic_substrate/core/brand_new/x.py") == ("tests/unit/core",)
    assert _scope("src/entropic_substrate/brand_new/sub/x.py") == stages.FULL_UNIT


def test_scopes_are_sorted_and_never_nested() -> None:
    assert _scope(
        "src/entropic_substrate/kernel/runtime/x.py",
        "src/entropic_substrate/kernel/lifecycle.py",
        "src/entropic_substrate/core/latent/x.py",
    ) == ("tests/unit/core/latent", "tests/unit/kernel")


def test_every_deferred_folder_is_one_a_later_stage_runs_or_holds_no_tests() -> None:
    later = {path for name in ("post-commit", "pre-push") for path in stages.STAGES[name].paths}
    unowned = [
        prefix
        for prefix in stages.DEFERRED_PREFIXES
        if prefix.startswith("tests/")
        and prefix.rstrip("/") not in later
        and prefix.rstrip("/").removeprefix("tests/") not in stages.FULL_TIER_ONLY
    ]
    assert unowned == []


def test_the_scoped_stage_keeps_the_fast_marker_and_parallel_flags() -> None:
    stage = stages.scoped_commit_stage(("tests/unit/core/latent",))
    assert stage == stages.Stage(("tests/unit/core/latent",), "not slow")
    assert stages.scoped_commit_stage(()) is None


# --- review round 1 (PR 439) ------------------------------------------------


def _g(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    done = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, env=env
    )
    return done.stdout.strip()


def _tmp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _g(repo, "init", "-q", "-b", "main")
    _g(repo, "config", "user.email", "t@example.invalid")
    _g(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-q", "-m", "base")
    return repo


def _branch(repo: Path, name: str, onto: str) -> str:
    _g(repo, "checkout", "-q", "-b", name, onto)
    (repo / f"{name}.txt").write_text(name, encoding="utf-8")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-q", "-m", name)
    return _g(repo, "rev-parse", "HEAD")


def test_staged_paths_come_from_the_index_git_hands_the_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `git commit -a` and `git commit --only` give the hook a TEMPORARY index
    # through GIT_INDEX_FILE; the ordinary index shows nothing staged.
    repo = _tmp_repo(tmp_path)
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    alt = tmp_path / "alt-index"
    shutil.copy(repo / ".git" / "index", alt)
    _g(repo, "add", "a.txt", env={**os.environ, "GIT_INDEX_FILE": str(alt)})
    assert _g(repo, "diff", "--cached", "--name-only") == ""  # control: default index is clean
    monkeypatch.setenv("GIT_INDEX_FILE", str(alt))
    assert stages._staged(repo) == ["a.txt"]


def test_pushed_commits_are_read_from_the_hook_input_not_from_head(tmp_path: Path) -> None:
    repo = _tmp_repo(tmp_path)
    one = _branch(repo, "one", "main")
    other = _branch(repo, "other", "main")
    zero = "0" * 40
    lines = [f"refs/heads/one {one} refs/heads/one {zero}", f"(delete) {zero} refs/heads/x {one}"]
    assert _g(repo, "rev-parse", "HEAD") == other
    _g(repo, "update-ref", "refs/remotes/origin/main", "main")
    assert stages.pushed_commits(repo, stdin_lines=lines, to_ref=None) == [one]
    assert stages.pushed_commits(repo, stdin_lines=[], to_ref=one) == [one]
    assert stages.pushed_commits(repo, stdin_lines=[], to_ref=None) == [other]


def test_a_stack_push_also_names_every_unpushed_branch_on_the_same_line(tmp_path: Path) -> None:
    # The hook runner hands over ONE ref; `gt submit --stack` pushes the line.
    repo = _tmp_repo(tmp_path)
    bottom = _branch(repo, "bottom", "main")
    top = _branch(repo, "top", "bottom")
    _branch(repo, "sibling", "main")
    _g(repo, "update-ref", "refs/remotes/origin/main", "main")
    assert stages.pushed_commits(repo, stdin_lines=[], to_ref=bottom) == [bottom, top]
    _g(repo, "update-ref", "refs/remotes/origin/top", top)  # the whole line is published
    assert stages.pushed_commits(repo, stdin_lines=[], to_ref=bottom) == []


def test_the_push_gate_refuses_when_any_pushed_tree_has_a_failed_record(tmp_path: Path) -> None:
    store = tmp_path / "store"
    stages.write_record(store, stages.Record("a" * 40, "post-commit", True, "ok"))
    stages.write_record(store, stages.Record("b" * 40, "post-commit", False, "1 failed"))
    reran: list[str] = []

    def rerun(tree: str) -> bool:
        reran.append(tree)
        return True

    assert stages.verify_all(store, ["a" * 40, "b" * 40], rerun=rerun) == ["b" * 40]
    assert stages.verify_all(store, ["a" * 40, "c" * 40], rerun=rerun) == []
    assert reran == ["c" * 40]


def test_an_export_records_the_tree_of_the_commit_it_was_given(tmp_path: Path) -> None:
    repo = _tmp_repo(tmp_path)
    one = _branch(repo, "one", "main")
    _branch(repo, "other", "main")
    seen: list[Path] = []

    def fake(stage: stages.Stage, *, cwd: Path, src: Path) -> tuple[bool, str]:
        seen.append(cwd)
        assert (cwd / "one.txt").is_file() and not (cwd / "other.txt").exists()
        return True, "ok"

    record = stages.run_in_export(repo, "post-commit", commit=one, run=fake)
    assert record.tree == _g(repo, "rev-parse", f"{one}^{{tree}}")
    assert seen and not seen[0].exists()  # the export is gone afterwards


# --- found by the first real push through the new gate (2026-11-21) ---------
#
# The push listed four branches; the gate exported the BOTTOM one, a commit
# already on the remote whose layout predates tests/unit, and pytest found
# nothing to run. Two rules follow.


def test_a_commit_the_remote_already_has_is_not_verified_again(tmp_path: Path) -> None:
    repo = _tmp_repo(tmp_path)
    bottom = _branch(repo, "bottom", "main")
    top = _branch(repo, "top", "bottom")
    _g(repo, "update-ref", "refs/remotes/origin/main", "main")
    _g(repo, "update-ref", "refs/remotes/origin/bottom", bottom)
    zero = "0" * 40
    lines = [
        f"refs/heads/bottom {bottom} refs/heads/bottom {bottom}",
        f"refs/heads/top {top} refs/heads/top {zero}",
    ]
    assert stages.pushed_commits(repo, stdin_lines=lines, to_ref=None) == [top]
    assert stages.pushed_commits(repo, stdin_lines=[], to_ref=bottom) == [top]
    _g(repo, "update-ref", "refs/remotes/origin/top", top)
    assert stages.pushed_commits(repo, stdin_lines=lines, to_ref=None) == []


def test_a_commit_older_than_the_stage_table_is_named_not_judged_by_it(tmp_path: Path) -> None:
    repo = _tmp_repo(tmp_path)
    before = _g(repo, "rev-parse", "HEAD")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "suite_stages.py").write_text("# table\n", encoding="utf-8")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-q", "-m", "table arrives")
    after = _g(repo, "rev-parse", "HEAD")
    assert stages.split_by_gate(repo, [before, after]) == ([after], [before])


def test_a_non_ascii_staged_path_is_read_unquoted(tmp_path: Path) -> None:
    repo = _tmp_repo(tmp_path)
    (repo / "é.txt").write_text("x\n", encoding="utf-8")
    _g(repo, "add", "-A")
    assert stages._staged(repo) == ["é.txt"]
