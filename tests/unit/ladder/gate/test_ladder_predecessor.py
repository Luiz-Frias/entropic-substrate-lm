"""A later-rung gate cannot be rehydrated from an outcome assertion."""

from pathlib import Path

import pytest
from tests.unit.ladder.gate.test_ladder_campaign import artifact

from entropic_substrate.ladder.gate.ladder_predecessor import load_ladder_predecessor

pytestmark = pytest.mark.unit


def test_summary_assertion_is_not_a_filed_gate(tmp_path: Path) -> None:
    summary = artifact(tmp_path, "summary.json", {"rung": "d2", "authorized": True})
    with pytest.raises(ValueError, match="fields"):
        load_ladder_predecessor(tmp_path, summary)
