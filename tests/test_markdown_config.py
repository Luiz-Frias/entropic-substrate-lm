"""Formatter policy and literal budget notation survive the Markdown gates."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def test_dprint_emphasis_matches_the_existing_markdownlint_rule() -> None:
    config = json.loads((ROOT / "dprint.json").read_text())
    assert config["markdown"].get("emphasisKind") == "asterisks"


@pytest.mark.parametrize("literal", ["2*4*d^2 = 4.72M", "e1-*"])
def test_budget_notation_is_protected_from_emphasis_parsing(literal: str) -> None:
    document = (ROOT / "configs" / "_shared.md").read_text()
    assert f"`{literal}`" in document
