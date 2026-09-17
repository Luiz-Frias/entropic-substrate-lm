"""Explicit repository operations must not inherit another repository's context."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from entropic_substrate.git_environment import isolated_git_environment

pytestmark = pytest.mark.unit


def test_child_environment_is_a_fresh_copy_without_git_context() -> None:
    original = {
        "PATH": "/usr/bin",
        "KEEP": "sentinel",
        "GIT_DIR": "foreign",
        "GIT_INDEX_FILE": "foreign-index",
        "GIT_CONFIG_COUNT": "1",
        "GIT_OBJECT_DIRECTORY": "foreign-objects",
        "GIT_FUTURE_OPTION": "unknown",
    }
    actual = isolated_git_environment(original)
    assert actual == {"PATH": "/usr/bin", "KEEP": "sentinel"}
    actual["KEEP"] = "changed"
    assert original["KEEP"] == "sentinel" and original["GIT_DIR"] == "foreign"
    assert isolated_git_environment({}) == {}


def test_nested_pytest_does_not_consume_or_modify_the_callers_index(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.index"
    sentinel = b"not this test repository's index"
    foreign.write_bytes(sentinel)
    root = Path(__file__).resolve().parents[1]
    environment = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": str(root / "src"),
        "GIT_INDEX_FILE": str(foreign),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_d1_predecessor_loader.py::test_bundle_proves_actual_blob_and_ancestry_without_checkout",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert foreign.read_bytes() == sentinel
