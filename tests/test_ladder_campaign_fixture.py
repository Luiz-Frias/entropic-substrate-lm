"""Real D1 filing graph supplies local test-only D2 authority."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_scientific_fixture_uses_real_d1_filing_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.d_ladder_cli import tiny_rung_spec

    from .ladder_campaign_fixtures import authorization

    spec = tiny_rung_spec("d2-s", seed=7)
    auth = authorization(
        tmp_path,
        spec,
        "a" * 64,
        {
            "train_start": 0,
            "train_end": 100,
            "eval_start": 100,
            "eval_end": 200,
            "source_total_bytes": 200,
        },
        monkeypatch,
    )
    assert auth.spec_digest == spec.digest and auth.rung == "d2"
