"""``repo_paths.REPO_ROOT`` is the one definition of the repository root for tests.

Before the tests/ tree was split into folders, 71 test modules derived the
root from their own depth (``Path(__file__).resolve().parents[1]``). That
encodes the directory layout in every file, so moving a file silently
re-points it. The root is now found by marker, not by depth.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import repo_paths

pytestmark = pytest.mark.unit


def test_root_is_the_directory_holding_the_project_markers() -> None:
    root = repo_paths.REPO_ROOT
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "entropic_substrate").is_dir()
    assert (root / "tests" / "conftest.py").is_file()


def test_root_is_independent_of_the_calling_file_depth() -> None:
    here = Path(__file__).resolve()
    assert repo_paths.REPO_ROOT in here.parents
    assert here.parents[1] != repo_paths.REPO_ROOT, (
        "this file sits two levels down; depth-1 must not be the root"
    )


def test_find_root_refuses_a_tree_without_the_marker(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"pyproject\.toml"):
        repo_paths.find_root(tmp_path / "a" / "b")


def test_no_test_module_derives_the_root_from_its_own_depth() -> None:
    own_file = re.compile(r"(?<![.\w])__file__")
    offenders = []
    for path in sorted(repo_paths.TESTS_ROOT.rglob("*.py")):
        if path.name in {"repo_paths.py", "conftest.py", Path(__file__).name}:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Subscript | ast.Attribute):
                continue
            text = ast.unparse(node)
            if not own_file.search(text):
                continue
            if (isinstance(node, ast.Subscript) and ".parents[" in text) or text.endswith(
                ".parent.parent"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
