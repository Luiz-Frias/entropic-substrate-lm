"""Scientific execution never obtains authority from a truthy flag."""

from pathlib import Path

import pytest

from entropic_substrate.composition.wiring.d_ladder_cli import tiny_rung_spec
from entropic_substrate.historical.d_ladder.d_ladder_request import ScientificRequest

pytestmark = pytest.mark.unit


def test_scientific_request_refuses_boolean_authority(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="RunAuthorization"):
        ScientificRequest(tiny_rung_spec("d2-s", seed=1), True, tmp_path, "enwik8")
