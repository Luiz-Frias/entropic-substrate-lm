"""Contracts for automated dependency update configuration."""

import pytest
import yaml

import repo_paths

pytestmark = pytest.mark.unit

REPO_ROOT = repo_paths.REPO_ROOT


def test_dependabot_checks_uv_and_actions_quarterly() -> None:
    config = yaml.safe_load((REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    updates = config["updates"]

    assert config["version"] == 2
    assert {entry["package-ecosystem"] for entry in updates} == {"uv", "github-actions"}
    assert all(entry["directory"] == "/" for entry in updates)
    assert all(entry["schedule"]["interval"] == "quarterly" for entry in updates)
