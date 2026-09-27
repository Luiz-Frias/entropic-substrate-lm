"""Explicit remote GPU tiny profiles preserve all four stabilization axes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from entropic_substrate.ladder.register.d1_config import (
    d1_configs_dir,
    load_d1_profile,
    load_d1_profile_catalog,
)
from entropic_substrate.ladder.register.d1_plan import parse_d1_confirmatory_plan

pytestmark = pytest.mark.unit


def test_gpu_tiny_keeps_cpu_tiny_budget_and_scientific_axes() -> None:
    cpu = load_d1_profile_catalog(tiny=True)
    gpu = load_d1_profile_catalog(tiny=True, tiny_target="remote_gpu")
    assert len(cpu) == len(gpu) == 4
    assert len({profile.family.family_digest for profile in gpu}) == 1
    for local, remote in zip(cpu, gpu, strict=True):
        assert remote.arm is local.arm
        assert remote.scale.value == "gpu_tiny"
        assert remote.draft.encoder_pair == local.draft.encoder_pair
        assert remote.draft.target == local.draft.target
        assert remote.draft.future_law == local.draft.future_law
        assert remote.draft.geometry == local.draft.geometry
        assert remote.draft.target_update is local.draft.target_update
        assert remote.draft.collapse_strategy is local.draft.collapse_strategy
        assert remote.draft.ema == local.draft.ema
        assert remote.draft.collapse == local.draft.collapse
        assert remote.data == local.data
        assert remote.train.max_steps == local.train.max_steps
        assert remote.train.device == "cuda"
        assert remote.train.amp_dtype == "bf16"
        assert local.train.device == "cpu"
        assert remote.family.family_digest != local.family.family_digest
        with pytest.raises(ValueError, match="full launch profile"):
            parse_d1_confirmatory_plan(remote, {})


def test_gpu_target_requires_tiny_mode() -> None:
    with pytest.raises(ValueError, match="requires tiny"):
        load_d1_profile_catalog(tiny_target="remote_gpu")


def test_execution_catalog_has_one_ordered_full_cpu_gpu_roster() -> None:
    from entropic_substrate.ladder.register.d1_config import load_d1_execution_catalog

    catalog = load_d1_execution_catalog()
    assert [profile.scale.value for profile in catalog] == (
        ["full"] * 4 + ["tiny"] * 4 + ["gpu_tiny"] * 4
    )
    assert len({profile.draft.construction_digest for profile in catalog}) == 12


def test_unknown_tiny_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="tiny_target"):
        load_d1_profile_catalog(tiny=True, tiny_target="automatic")


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("future_law", "future_loss_weight", 0.0, "future_loss_weight"),
        ("train", "device", "cpu", "CUDA with bf16"),
        ("train", "amp_dtype", "off", "CUDA with bf16"),
    ],
)
def test_gpu_overlay_rejects_axis_mutation_or_silent_cpu_fallback(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    payload = yaml.safe_load((d1_configs_dir() / "d1-gpu-tiny.yaml").read_text())
    payload[section][key] = value
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match=message):
        load_d1_profile(d1_configs_dir() / "d1-both.yaml", tiny_overlay_path=overlay)
