"""Four-arm catalog and D1-only overlay tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from entropic_substrate.experiments.d1_config import (
    D1LaunchScale,
    d1_configs_dir,
    load_d1_profile,
    load_d1_profile_catalog,
)
from entropic_substrate.experiments.d1_primitives import D1Arm

pytestmark = pytest.mark.unit


def _payload(name: str) -> dict[str, object]:
    value = yaml.safe_load((d1_configs_dir() / f"{name}.yaml").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_full_and_tiny_catalogs_are_exact_matched_families() -> None:
    full, tiny = load_d1_profile_catalog(), load_d1_profile_catalog(tiny=True)
    assert tuple(profile.arm for profile in full) == tuple(D1Arm)
    assert {profile.scale for profile in full} == {D1LaunchScale.FULL}
    assert {profile.scale for profile in tiny} == {D1LaunchScale.TINY}
    assert len({profile.family.family_digest for profile in full}) == 1
    assert len({profile.family.family_digest for profile in tiny}) == 1
    assert full[0].family.family_digest != tiny[0].family.family_digest
    full_protocol = full[0].draft.execution_protocol.protocol_digest
    tiny_protocol = tiny[0].draft.execution_protocol.protocol_digest
    assert full_protocol != tiny_protocol
    assert all(profile.data.source == "synthetic" for profile in tiny)
    assert all(profile.train.device == "cpu" for profile in tiny)


def test_catalog_refuses_cross_profile_family_drift(tmp_path: Path) -> None:
    paths: list[Path] = []
    for name in ("d1-none", "d1-sg", "d1-ac", "d1-both"):
        payload = _payload(name)
        if name == "d1-sg":
            encoder = payload["encoder"]
            assert isinstance(encoder, dict)
            encoder["n_layers"] = 10
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        paths.append(path)
    with pytest.raises(ValueError, match="one family digest"):
        load_d1_profile_catalog(paths=paths)


def test_tiny_overlay_cannot_select_a_scientific_arm(tmp_path: Path) -> None:
    overlay = yaml.safe_load((d1_configs_dir() / "d1-tiny.yaml").read_text(encoding="utf-8"))
    assert isinstance(overlay, dict)
    overlay["arm"] = "both"
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_d1_profile(d1_configs_dir() / "d1-none.yaml", tiny_overlay_path=path)
