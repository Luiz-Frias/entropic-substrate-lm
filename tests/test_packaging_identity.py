"""p0 packaging door (frozen_after: p13): pins that ``[tool.setuptools.packages.find]``
auto-discovers the five ruled empty layer packages with zero edits to its own
config (DN-17 -- 'Packaging collapses to a pin: `where = ["src"]` already
auto-discovers nested packages. Assert, do not edit.'), and that the
basedpyright / pre-commit include regexes for src+scripts stay aligned so a
future divergence between them is caught here rather than silently in CI.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from setuptools import find_packages

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"

_FROZEN_NEW_LAYER_PACKAGES = (
    "entropic_substrate.composition",
    "entropic_substrate.core",
    "entropic_substrate.historical",
    "entropic_substrate.kernel",
    "entropic_substrate.ladder",
)

_FROZEN_PACKAGES = (
    "entropic_substrate",
    "entropic_substrate.analysis",
    "entropic_substrate.composition",
    "entropic_substrate.core",
    "entropic_substrate.data",
    "entropic_substrate.evidence",
    "entropic_substrate.experiments",
    "entropic_substrate.geometry",
    "entropic_substrate.historical",
    "entropic_substrate.init",
    "entropic_substrate.kernel",
    "entropic_substrate.ladder",
    "entropic_substrate.layers",
    "entropic_substrate.models",
    "entropic_substrate.observation",
    "entropic_substrate.predictive_state",
    "entropic_substrate.sentence_prediction",
)


def _pyproject() -> dict:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


class TestPackagesFindDiscoversTheFiveLayerMarkers:
    def test_packages_find_config_is_unedited_where_src(self) -> None:
        config = _pyproject()
        find_cfg = config["tool"]["setuptools"]["packages"]["find"]
        assert find_cfg == {"where": ["src"]}, (
            "packages.find config changed -- DN-17 rules this collapses to a "
            "pin, never an edit; a real change here is a DECISION"
        )

    def test_discovery_over_the_live_tree_includes_all_five_new_layer_packages(self) -> None:
        discovered = set(find_packages(where=str(_SRC)))
        missing = set(_FROZEN_NEW_LAYER_PACKAGES) - discovered
        assert not missing, f"layer marker packages not discovered: {sorted(missing)}"

    def test_discovery_over_the_live_tree_matches_the_frozen_package_list(self) -> None:
        discovered = tuple(sorted(find_packages(where=str(_SRC))))
        assert discovered == _FROZEN_PACKAGES


class TestBasedpyrightAndPrecommitIncludeRegexesStayAligned:
    """basedpyright's [tool.basedpyright].include and the pre-commit mirror
    hook's `files:` regex both claim to cover src/entropic_substrate + scripts.
    This asserts they still agree, without editing .pre-commit-config.yaml
    (operator ask-first)."""

    def test_pyproject_basedpyright_include_is_frozen(self) -> None:
        config = _pyproject()
        include = config["tool"]["basedpyright"]["include"]
        assert include == ["src", "scripts"]

    def test_precommit_basedpyright_files_regex_matches_the_same_tree(self) -> None:
        precommit_text = (_REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        # Locate the basedpyright hook block and its `files:` regex without a
        # YAML parse dependency shift -- a plain line scan anchored on the
        # hook id, matching how this file's own comment cites the pin.
        match = re.search(r"id:\s*basedpyright.*?files:\s*(\S+)", precommit_text, re.DOTALL)
        assert match is not None, "basedpyright hook not found in .pre-commit-config.yaml"
        files_regex = match.group(1)
        pattern = re.compile(files_regex)
        assert pattern.match("src/entropic_substrate/checkpoint.py")
        assert pattern.match("scripts/run_exit_gate.py")
        assert not pattern.match("tests/test_packaging_identity.py")

    def test_precommit_basedpyright_files_regex_covers_the_five_new_layer_markers(self) -> None:
        precommit_text = (_REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        match = re.search(r"id:\s*basedpyright.*?files:\s*(\S+)", precommit_text, re.DOTALL)
        assert match is not None
        pattern = re.compile(match.group(1))
        for pkg in ("kernel", "core", "ladder", "historical", "composition"):
            rel = f"src/entropic_substrate/{pkg}/__init__.py"
            assert pattern.match(rel), f"{rel} not covered by the basedpyright files regex"
