"""Teeth for ``scripts/ship_scope.py``: slim ``just ship-it`` must refuse a
change that touches the live stack and point at ``just ship-it-full``.

The slim recipe no longer runs the container gate. The hazard is the one
house rule 1 records: dependency commit 81bd4dd turned the live MLflow gate
red under a green CI because nothing ran it. "Run the full gate when infra
changes" only holds if a machine decides what counts as infra.
"""

from __future__ import annotations

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
    import subprocess

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
