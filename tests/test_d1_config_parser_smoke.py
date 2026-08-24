"""Independent smoke contract for the strict D1 profile parser."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from entropic_substrate.experiments.d1_config_profile import load_d1_profile

pytestmark = pytest.mark.unit
_PROFILE = Path(__file__).parents[1] / "configs/d1-none.yaml"


def test_profile_binds_execution_and_returns_fresh_legacy_adapters() -> None:
    profile = load_d1_profile(_PROFILE)
    assert profile.arm.value == "none"
    assert profile.draft.execution_protocol == profile.family.drafts[0].execution_protocol
    data, train = profile.data, profile.train
    data.source, train.max_steps = "synthetic", 1
    assert profile.data.source == "enwik8"
    assert profile.train.max_steps == 20_000
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.name = "d1-sg"  # type: ignore[misc]
