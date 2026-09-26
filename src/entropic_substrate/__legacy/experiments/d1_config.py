"""Public D1 profile loader and matched four-arm catalog."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final

from entropic_substrate.config import configs_dir, merge_dicts
from entropic_substrate.experiments.d1_config_profile import (
    D1LaunchProfile,
    D1LaunchScale,
    parse_d1_profile_payload,
    read_d1_profile_mapping,
    require_d1_exact_keys,
)
from entropic_substrate.experiments.d1_config_profile import (
    load_d1_profile as _load_full_profile,
)
from entropic_substrate.experiments.d1_primitives import D1Arm

__all__ = [
    "D1LaunchProfile",
    "D1LaunchScale",
    "d1_configs_dir",
    "load_d1_execution_catalog",
    "load_d1_profile",
    "load_d1_profile_catalog",
]

_PROFILE_ORDER: Final = ("d1-none", "d1-sg", "d1-ac", "d1-both")
_OVERLAY_KEYS: Final = ("kind", "encoder", "future_law", "data", "train")


def d1_configs_dir() -> Path:
    """Locate D1 YAMLs through the repository's shared config locator."""
    return configs_dir()


def load_d1_profile(
    path: str | Path,
    *,
    tiny_overlay_path: str | Path | None = None,
) -> D1LaunchProfile:
    """Load one full profile or apply the strictly budget-only tiny overlay."""
    if tiny_overlay_path is None:
        return _load_full_profile(path)
    payload = read_d1_profile_mapping(Path(path))
    overlay = read_d1_profile_mapping(Path(tiny_overlay_path))
    require_d1_exact_keys(overlay, _OVERLAY_KEYS, label="D1 tiny overlay")
    scales = {"d1-tiny-overlay": D1LaunchScale.TINY, "d1-gpu-tiny-overlay": D1LaunchScale.GPU_TINY}
    kind = overlay["kind"]
    if type(kind) is not str or kind not in scales:
        msg = "D1 tiny overlay requires kind='d1-tiny-overlay' or 'd1-gpu-tiny-overlay'"
        raise ValueError(msg)
    merged = merge_dicts(
        payload,
        {key: overlay[key] for key in _OVERLAY_KEYS if key != "kind"},
    )
    return parse_d1_profile_payload(merged, scale=scales[kind])


def load_d1_profile_catalog(
    *,
    paths: Iterable[str | Path] | None = None,
    tiny: bool = False,
    tiny_target: str = "host_cpu",
) -> tuple[D1LaunchProfile, ...]:
    """Load exactly one matched profile for every registered D1 arm."""
    if tiny_target not in {"host_cpu", "remote_gpu"}:
        raise ValueError("D1 tiny_target must be host_cpu or remote_gpu")
    if tiny_target == "remote_gpu" and not tiny:
        raise ValueError("D1 remote_gpu target requires tiny=True")
    selected = (
        tuple(Path(path) for path in paths)
        if paths is not None
        else tuple(d1_configs_dir() / f"{name}.yaml" for name in _PROFILE_ORDER)
    )
    overlay_name = "d1-tiny.yaml" if tiny_target == "host_cpu" else "d1-gpu-tiny.yaml"
    overlay = d1_configs_dir() / overlay_name if tiny else None
    profiles = tuple(load_d1_profile(path, tiny_overlay_path=overlay) for path in selected)
    by_name = {profile.name: profile for profile in profiles}
    if len(profiles) != len(D1Arm) or set(by_name) != set(_PROFILE_ORDER):
        msg = "D1 launch catalog requires each exact registered profile once"
        raise ValueError(msg)
    ordered = tuple(by_name[name] for name in _PROFILE_ORDER)
    if len({profile.family.family_digest for profile in ordered}) != 1:
        msg = "D1 launch profiles must resolve to one family digest"
        raise ValueError(msg)
    if len({profile.data_manifest for profile in ordered}) != 1:
        msg = "D1 launch profiles must share one DataConfig identity"
        raise ValueError(msg)
    if len({profile.train_manifest for profile in ordered}) != 1:
        msg = "D1 launch profiles must share one TrainConfig identity"
        raise ValueError(msg)
    return ordered


def load_d1_execution_catalog() -> tuple[D1LaunchProfile, ...]:
    """One ordered full/CPU-tiny/GPU-tiny roster for every execution entrypoint."""
    return (
        *load_d1_profile_catalog(),
        *load_d1_profile_catalog(tiny=True),
        *load_d1_profile_catalog(tiny=True, tiny_target="remote_gpu"),
    )
