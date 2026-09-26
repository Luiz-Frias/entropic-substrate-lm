"""One complete numerical profile; it describes a run but cannot authorize science."""

from dataclasses import replace

import pytest

from entropic_substrate.core.latent.dynamics import SolverConfig
from entropic_substrate.core.sentence.codec import CodecSpec
from entropic_substrate.core.sentence.law import LawSpec
from entropic_substrate.core.sentence.profile import HProfile, OptimizationSpec, RunSchedule
from entropic_substrate.core.sentence.records import content_identity
from entropic_substrate.core.sentence.spine import SpineSpec

pytestmark = pytest.mark.unit


def profile() -> HProfile:
    return HProfile(
        name="h0-fisher-tiny",
        stage="H0",
        run_class="local_tiny",
        source_digest=content_identity(("source-fixture",)),
        seed=42,
        source_seed=42,
        device="cpu",
        precision="fp32",
        codec=CodecSpec(8, 1, 8, 1, 4, 64),
        spine=SpineSpec(8, 2, 4, 4, "magnitude"),
        law=LawSpec(8, "magnitude"),
        consumer="fisher",
        solver=SolverConfig(),
        optimization=OptimizationSpec(0.001, 0.01, (0.9, 0.95), 1e-8, 1.0, 0.3),
        schedule=RunSchedule(2, 4, 2, 2, 2, 1, (2, 4), 1, 2),
        permutation=tuple(range(256)),
    )


def test_complete_profile_roundtrip_binds_every_mechanism_and_budget() -> None:
    value = profile()
    restored = HProfile.from_manifest(value.manifest)
    assert restored == value and restored.identity == value.identity
    for changed in (
        replace(value, consumer="isotropic"),
        replace(value, law=replace(value.law, mode="sphere")),
        replace(value, spine=replace(value.spine, carrier="sphere")),
        replace(value, permutation=tuple(reversed(range(256)))),
        replace(value, schedule=replace(value.schedule, steps=6000)),
    ):
        assert changed.identity != value.identity


@pytest.mark.parametrize("section", ["codec", "spine", "law", "solver", "optimization", "schedule"])
def test_missing_nested_fields_never_use_constructor_defaults(section: str) -> None:
    payload = profile().manifest
    payload[section].pop(next(iter(payload[section])))
    with pytest.raises(ValueError, match="fields"):
        HProfile.from_manifest(payload)


def test_remote_tiny_requires_cuda_bf16_and_retains_six_thousand_steps() -> None:
    local = profile()
    with pytest.raises(ValueError, match="CUDA"):
        replace(local, run_class="remote_tiny")
    remote = replace(
        local,
        run_class="remote_tiny",
        device="cuda",
        precision="bf16",
        schedule=replace(local.schedule, steps=6000),
    )
    assert remote.schedule.steps == 6000 and remote.consumer == "fisher"
    assert HProfile.from_manifest(remote.manifest) == remote


def test_scientific_profiles_cannot_impersonate_tiny_or_bundle_h1_mechanisms() -> None:
    with pytest.raises(ValueError, match="tiny"):
        replace(profile(), run_class="confirmatory", device="cuda", precision="bf16")
    with pytest.raises(ValueError, match="H1"):
        replace(profile(), stage="H1", name="h1-fisher-tiny")
    with pytest.raises(ValueError, match="H1"):
        replace(profile(), stage="H1", name="h1-field-tiny", consumer="isotropic")


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": True},
        {"stage": "D1"},
        {"precision": "half"},
        {"name": "../bad"},
        {"permutation": (0,) * 256},
    ],
)
def test_invalid_profile_fields_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(profile(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 0},
        {"checkpoint_interval": 0},
        {"importance_samples": (4, 2)},
        {"importance_chunk": 0},
        {"generation_sentences": True},
    ],
)
def test_invalid_execution_schedule_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(profile().schedule, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"learning_rate": 0},
        {"weight_decay": -1},
        {"future_weight": float("nan")},
        {"betas": (True, 0.95)},
        {"epsilon": 0},
    ],
)
def test_invalid_optimization_policy_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(profile().optimization, **changes)


def test_boolean_solver_constants_are_not_numeric_registration_values() -> None:
    with pytest.raises(ValueError, match="booleans"):
        replace(profile(), solver=SolverConfig(damping=True))
