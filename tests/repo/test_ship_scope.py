"""Teeth for ``scripts/ship_scope.py``: slim ``just ship-it`` must refuse a
change that touches the live stack and point at ``just ship-it-full``.

The slim recipe no longer runs the container gate. The hazard is the one
house rule 1 records: dependency commit 81bd4dd turned the live MLflow gate
red under a green CI because nothing ran it. "Run the full gate when infra
changes" only holds if a machine decides what counts as infra.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import ship_scope

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "path",
    [
        "compose.yaml",
        "Containerfile",
        "Containerfile.caddy",
        "Caddyfile",
        ".dockerignore",
        ".containerignore",
        "pyproject.toml",
        "uv.lock",
        "scripts/run_container_test.sh",
        "scripts/derive_compose_project.sh",
        "src/entropic_substrate/ladder/capture/adapters/mlflow.py",
        "src/entropic_substrate/ladder/capture/adapters/mlflow_strict.py",
        "src/entropic_substrate/ladder/capture/reader/sinks.py",
        "tests/live/test_mlflow_live.py",
        "src/entropic_substrate/kernel/runtime/logging.py",
        "src/entropic_substrate/ladder/run/worker_log_events.py",
        "src/entropic_substrate/ladder/run/worker/h_probe_log_events.py",
        "src/entropic_substrate/ladder/run/worker/h_probe_host_log.py",
        "scripts/logger_smoke.py",
    ],
)
def test_live_stack_paths_require_the_full_gate(path: str) -> None:
    assert ship_scope.requires_full([path]) == [path]


@pytest.mark.parametrize(
    "path",
    [
        "src/entropic_substrate/core/sentence/model.py",
        "tests/unit/core/sentence/test_sentence_model.py",
        "docs/architecture.md",
        "justfile",
        "scripts/suite_stages.py",
        "src/entropic_substrate/ladder/capture/reader/telemetry.py",
        "notes/compose.yaml.md",
    ],
)
def test_other_paths_do_not(path: str) -> None:
    assert ship_scope.requires_full([path]) == []


def test_one_infra_file_among_many_is_enough_and_is_named() -> None:
    changed = [
        "docs/a.md",
        "uv.lock",
        "src/entropic_substrate/core/sentence/model.py",
        "compose.yaml",
    ]
    assert ship_scope.requires_full(changed) == ["compose.yaml", "uv.lock"]


def test_an_empty_change_set_is_refused_not_passed() -> None:
    assert ship_scope.decide([]) == (2, "no changed files against the base: nothing examined")


def test_decide_names_the_files_and_the_recipe_to_run() -> None:
    code, message = ship_scope.decide(["uv.lock", "docs/a.md"])
    assert code == ship_scope.EXIT_NEEDS_FULL
    assert "uv.lock" in message
    assert "just ship-it-full" in message
    assert ship_scope.decide(["docs/a.md"])[0] == 0


def test_every_infra_pattern_matches_something_in_this_repository() -> None:
    import repo_paths

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=repo_paths.REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.split()
    dead = [
        pattern
        for pattern in ship_scope.INFRA_PATTERNS
        if not ship_scope.requires_full(tracked, patterns=(pattern,))
    ]
    assert dead == [], "a pattern that matches no tracked file guards nothing"


# --- review round 1 (PR 439): the gate's own entry point, and renames -------


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _repo(tmp_path: Path, justfile: str = "container-test:\n    bash gate.sh\n") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "justfile").write_text(justfile + "\nother:\n    echo hi\n", encoding="utf-8")
    (repo / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    (repo / "notes.md").write_text("n\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "work")
    return repo


def _commit(repo: Path) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")


def test_a_change_to_the_container_test_recipe_requires_the_full_gate(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    text = (repo / "justfile").read_text(encoding="utf-8")
    (repo / "justfile").write_text(text.replace("bash gate.sh", "true"), encoding="utf-8")
    _commit(repo)
    code, message = ship_scope.decide(ship_scope.changed_files(repo, "main"))
    assert code == ship_scope.EXIT_NEEDS_FULL
    assert "container-test" in message


def test_a_change_to_another_recipe_in_the_justfile_stays_slim(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    text = (repo / "justfile").read_text(encoding="utf-8")
    (repo / "justfile").write_text(text.replace("echo hi", "echo bye"), encoding="utf-8")
    _commit(repo)
    code, _ = ship_scope.decide(ship_scope.changed_files(repo, "main"))
    assert code == 0


def test_renaming_a_live_stack_file_away_is_still_seen(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "notes").mkdir()
    _git(repo, "mv", "compose.yaml", "notes/compose.yaml.md")
    _commit(repo)
    changed = ship_scope.changed_files(repo, "main")
    assert "compose.yaml" in changed
    assert ship_scope.decide(changed)[0] == ship_scope.EXIT_NEEDS_FULL


def test_the_recipe_reader_takes_the_header_and_indented_body_only() -> None:
    text = "a:\n    one\n\n# note\ncontainer-test dep: x\n    two\n    three\n\nb:\n    four\n"
    assert ship_scope.recipe_text(text, "container-test") == (
        "container-test dep: x\n    two\n    three"
    )
    assert ship_scope.recipe_text(text, "missing") == ""


def test_the_guarded_recipe_exists_in_this_repository() -> None:
    import repo_paths

    text = (repo_paths.REPO_ROOT / "justfile").read_text(encoding="utf-8")
    assert "run_container_test.sh" in ship_scope.recipe_text(text, ship_scope.GATE_RECIPE)


def test_a_non_ascii_live_path_is_matched_not_quoted_away(tmp_path: Path) -> None:
    # git quotes such a path ("tests/live/test_\303\251.py") unless asked for
    # NUL-separated output, and the quoted form matches no pattern.
    repo = _repo(tmp_path)
    (repo / "tests" / "live").mkdir(parents=True)
    (repo / "tests" / "live" / "test_é.py").write_text("x = 1\n", encoding="utf-8")
    _commit(repo)
    changed = ship_scope.changed_files(repo, "main")
    assert "tests/live/test_é.py" in changed
    assert ship_scope.decide(changed)[0] == ship_scope.EXIT_NEEDS_FULL


def test_a_changed_path_keeps_its_leading_whitespace(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / " odd.txt").write_text("x\n", encoding="utf-8")
    _commit(repo)
    assert " odd.txt" in ship_scope.changed_files(repo, "main")
