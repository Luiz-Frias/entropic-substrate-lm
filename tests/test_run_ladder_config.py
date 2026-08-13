"""Contracts for the RunPod experiment-ladder shell boundary."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runpod_ladder_restricts_the_dvc_site_cache() -> None:
    """Mitigate DiskCache local tampering until DVC ships a fixed dependency."""
    ladder = (REPO_ROOT / "scripts" / "run_ladder.sh").read_text(encoding="utf-8")

    assert "umask 077" in ladder
    assert "DVC_SITE_CACHE_DIR=${DVC_SITE_CACHE_DIR:-.cache/dvc-site-cache}" in ladder
    assert 'install -d -m 0700 "$DVC_SITE_CACHE_DIR"' in ladder
    assert 'chmod 0700 "$DVC_SITE_CACHE_DIR"' in ladder
