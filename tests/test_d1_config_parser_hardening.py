"""Mutation teeth for strict D1 YAML parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from entropic_substrate.ladder.register.d1_config import d1_configs_dir, load_d1_profile

pytestmark = pytest.mark.unit


def _payload() -> dict[str, object]:
    value = yaml.safe_load((d1_configs_dir() / "d1-none.yaml").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _mutate(payload: dict[str, object], path: tuple[str, ...], value: object) -> None:
    target = payload
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("arm",), "sg", "requires arm"),
        (("model",), {"field_state": "hidden"}, "unknown keys"),
        (("state", "field_state"), "hidden", "unknown keys"),
        (("codec", "kind"), "gpt2", "identity byte codec"),
        (("encoder", "initialization", "name"), "fisher", "initialization identity"),
        (("ema", "decay"), 1.0, "EMA decay"),
        (("collapse", "std_floor"), 0.0, "strictly positive"),
        (("future_law", "mean_baseline_calibration_batches"), 31, "32 batches"),
        (("data", "batch_size"), True, "integer"),
        (("data", "val_fraction"), 1, "finite float"),
        (("train", "max_steps"), "20000", "integer"),
        (("train", "lr"), -0.1, "finite float"),
    ],
)
def test_profile_refuses_schema_type_range_and_construction_drift(
    tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    payload = _payload()
    _mutate(payload, path, value)
    with pytest.raises(ValueError, match=message):
        load_d1_profile(_write(tmp_path, payload))


def test_profile_refuses_missing_required_section(tmp_path: Path) -> None:
    payload = _payload()
    payload.pop("encoder")
    with pytest.raises(ValueError, match="missing keys"):
        load_d1_profile(_write(tmp_path, payload))
