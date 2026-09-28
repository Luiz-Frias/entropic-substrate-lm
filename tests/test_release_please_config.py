"""release-please is the only publisher of stable ``vX.Y.Z`` tags; this pins its inputs.

Guard teeth for the manifest-driven release-please setup (``release-please-config.json``,
``.release-please-manifest.json``, ``.github/workflows/release-please.yml``). Every file
that carries the project version must agree with the manifest AND be owned by a
release-please updater, otherwise a release PR bumps one copy and leaves another stale:

- ``pyproject.toml`` ``[project].version``: the python strategy's own updater.
- ``src/entropic_substrate/__init__.py`` ``__version__``: NOT found by the python
  strategy (it probes ``src/<project name>/__init__.py`` for ``entropic-substrate-lm`` and
  ``entropic_substrate_lm`` only, release-please v17.11.2 ``src/strategies/python.ts``), so
  it is a ``generic`` extra-file and its line carries the ``x-release-please-version`` mark.
- ``uv.lock``: records the editable project's own version. MEASURED 2026-09-28: bumping
  ``pyproject.toml`` to 0.1.1 alone makes ``uv lock --check`` fail ("needs to be updated");
  bumping the one ``uv.lock`` line with it passes. So ``uv.lock`` is a ``toml`` extra-file.
  The jsonpath reads ``@.name.value`` because release-please's TOML parser tags every
  scalar as ``{value, start, end}`` (``src/util/toml-edit.ts``).

The bootstrap SHA must be a full 40-hex commit on this history: release-please stops its
first commit walk there, and a SHA it never meets means walking the whole history.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "release-please-config.json"
MANIFEST_PATH = REPO / ".release-please-manifest.json"
WORKFLOW_PATH = REPO / ".github" / "workflows" / "release-please.yml"
INIT_PY = "src/entropic_substrate/__init__.py"
VERSION_MARK = "x-release-please-version"


def _config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _manifest_version() -> str:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert list(manifest) == ["."], f"expected a single root package, got {manifest!r}"
    return manifest["."]


def _pyproject() -> dict[str, Any]:
    with (REPO / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def _root_package() -> dict[str, Any]:
    packages = _config()["packages"]
    assert list(packages) == ["."], f"expected exactly the root package, got {list(packages)}"
    return packages["."]


def _extra_files() -> list[dict[str, Any]]:
    extra = _root_package().get("extra-files", [])
    assert all(isinstance(e, dict) for e in extra), f"extra-files must be typed objects: {extra}"
    return extra


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def test_manifest_version_equals_pyproject_version() -> None:
    assert _manifest_version() == _pyproject()["project"]["version"]


def test_config_is_python_root_package_with_bare_v_tags() -> None:
    config = _config()
    assert config["release-type"] == "python"
    assert config["include-component-in-tag"] is False
    assert _root_package().get("include-component-in-tag", False) is False


def test_package_dunder_version_is_owned_and_matches_manifest() -> None:
    source = (REPO / INIT_PY).read_text(encoding="utf-8")
    values = [
        node.value.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert values == [_manifest_version()], values
    line = next(ln for ln in source.splitlines() if ln.startswith("__version__"))
    assert VERSION_MARK in line, f"generic updater needs the inline mark on: {line!r}"
    assert {"type": "generic", "path": INIT_PY} in _extra_files()


def test_uv_lock_project_version_is_owned_and_matches_manifest() -> None:
    name = _pyproject()["project"]["name"]
    with (REPO / "uv.lock").open("rb") as f:
        lock = tomllib.load(f)
    versions = [p.get("version") for p in lock["package"] if p["name"] == name]
    assert versions == [_manifest_version()], versions
    expected = {
        "type": "toml",
        "path": "uv.lock",
        "jsonpath": f"$.package[?(@.name.value=='{name}')].version",
    }
    assert expected in _extra_files()


def test_bootstrap_sha_is_a_full_sha() -> None:
    sha = _config()["bootstrap-sha"]
    assert re.fullmatch(r"[0-9a-f]{40}", sha), f"bootstrap-sha must be a full 40-hex SHA: {sha!r}"


def test_bootstrap_sha_is_an_ancestor_of_head() -> None:
    shallow = _git("rev-parse", "--is-shallow-repository").stdout.strip()
    if shallow != "false":
        pytest.skip(f"shallow or unreadable clone (is-shallow={shallow!r}): history absent")
    sha = _config()["bootstrap-sha"]
    exists = _git("cat-file", "-e", f"{sha}^{{commit}}")
    assert exists.returncode == 0, f"bootstrap-sha {sha} is not a commit here: {exists.stderr}"
    ancestor = _git("merge-base", "--is-ancestor", sha, "HEAD")
    assert ancestor.returncode == 0, f"bootstrap-sha {sha} is not an ancestor of HEAD"


def test_changelog_sections_use_only_commit_types_the_schema_admits() -> None:
    pattern = _pyproject()["tool"]["commitizen"]["customize"]["schema_pattern"]
    admitted = set(re.search(r"\(\?:\(\?:([a-z|]+)\)", pattern).group(1).split("|"))  # type: ignore[union-attr]
    sections = _config()["changelog-sections"]
    types = [s["type"] for s in sections]
    assert set(types) <= admitted, set(types) - admitted
    assert len(types) == len(set(types)), types


def test_workflow_runs_the_action_on_main_with_both_files() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    triggers = workflow[True] if True in workflow else workflow["on"]  # YAML 1.1 reads `on` as True
    assert triggers["push"]["branches"] == ["main"]
    steps = [s for job in workflow["jobs"].values() for s in job["steps"]]
    action = [
        s for s in steps if str(s.get("uses", "")).startswith("googleapis/release-please-action@")
    ]
    assert len(action) == 1, action
    # The action is pinned to a full commit SHA, never a mutable tag (review on
    # PR #306, 2026-09-28): `v5.0.0` is a lightweight tag on this commit, and
    # a moved tag would otherwise change what runs with `contents: write`.
    ref = str(action[0]["uses"]).split("@", 1)[1]
    assert re.fullmatch(r"[0-9a-f]{40}", ref), f"action must be SHA-pinned, got {ref!r}"
    inputs = action[0]["with"]
    assert inputs["config-file"] == CONFIG_PATH.name
    assert inputs["manifest-file"] == MANIFEST_PATH.name
