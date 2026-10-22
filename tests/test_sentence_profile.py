"""One complete numerical profile; it describes a run but cannot authorize science."""

from dataclasses import asdict, replace
from inspect import Parameter, signature

import pytest

import entropic_substrate.core.sentence.profile as profile_module
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
        schedule=RunSchedule(2, 4, 2, 2, 2, 1, (2, 4), 1, 2, diagnostic_samples=8),
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
        {"seed": 2**32},
        {"seed": 2**40},
        {"stage": "D1"},
        {"precision": "half"},
        {"name": "../bad"},
        {"permutation": (0,) * 256},
    ],
)
def test_invalid_profile_fields_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(profile(), **changes)


def test_seed_boundary_is_the_largest_step_keyed_seed() -> None:
    assert replace(profile(), seed=2**32 - 1).seed == 2**32 - 1

    def mixed(seed: int, step: int) -> int:  # the evaluation seed (execution.py)
        return (seed * 0x9E3779B1 + step) % 2**32

    # Why 2**32 is the bound: from there seed aliases seed - 2**32 at every step, while
    # the largest admitted seed stays a distinct in-range 32-bit key.
    assert mixed(2**32, 7) == mixed(0, 7)
    assert 0 <= mixed(2**32 - 1, 2**32 - 1) < 2**32
    assert mixed(2**32 - 1, 7) != mixed(0, 7)


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


def development(run_class: str = "pilot") -> HProfile:
    """The registered development profile: a non-smoke class on the full curve (FOR-1664)."""
    local = profile()
    return replace(
        local,
        name="h0-fisher",
        run_class=run_class,
        device="cuda",
        precision="bf16",
        schedule=replace(
            local.schedule,
            importance_samples=(2, 4, 8, 16, 32, 64),
            diagnostic_samples=256,
        ),
    )


def test_development_setting_is_the_registered_curve_plus_one_diagnostic() -> None:
    """Pins the operator ruling (2026-10-20): a curve ending in 64, 256 carried apart."""
    assert profile_module.DEVELOPMENT_IMPORTANCE_SAMPLES == (2, 4, 8, 16, 32, 64)
    assert profile_module.DEVELOPMENT_DIAGNOSTIC_SAMPLES == 256


@pytest.mark.parametrize("run_class", ["pilot", "confirmatory"])
def test_development_classes_require_the_registered_curve_and_diagnostic(run_class: str) -> None:
    accepted = development(run_class)  # control: the refusals below are about the schedule
    assert accepted.run_class == run_class
    assert HProfile.from_manifest(accepted.manifest) == accepted
    for changes in (
        {"importance_samples": (2, 4, 8, 16, 32)},  # stops short of the 64 headline
        {"importance_samples": (2, 4)},  # the plumbing smoke setting
        {"importance_samples": (4, 8, 16, 32, 64)},  # drops the first point of the curve
        {"importance_samples": (2, 4, 8, 16, 32, 64, 128)},  # exact equality, not a floor
        {"diagnostic_samples": 128},
        {"diagnostic_samples": 512},
    ):
        with pytest.raises(ValueError, match="development") as refusal:
            replace(accepted, schedule=replace(accepted.schedule, **changes))
        # Other tests pin the tiny/authority guards by these substrings; this message
        # must never satisfy them, or their fixtures could pass without reaching them.
        for word in ("tiny", "scientific", "authority"):
            assert word not in str(refusal.value)


def test_smoke_run_classes_are_the_only_ones_below_the_development_envelope() -> None:
    assert profile_module.SMOKE_RUN_CLASSES == ("local_tiny", "remote_tiny")
    local = profile()
    assert local.run_class in profile_module.SMOKE_RUN_CLASSES
    assert local.schedule.importance_samples == (2, 4)
    remote = replace(local, run_class="remote_tiny", device="cuda", precision="bf16")
    assert remote.schedule.importance_samples == (2, 4)
    for run_class in ("pilot", "confirmatory"):
        assert run_class not in profile_module.SMOKE_RUN_CLASSES
        with pytest.raises(ValueError, match="development"):
            replace(local, run_class=run_class, name="h0-fisher", device="cuda", precision="bf16")


@pytest.mark.parametrize("value", [4, 3, 0, -8, True, 8.0, "8", None])
def test_diagnostic_samples_must_be_an_int_beyond_the_curve(value: object) -> None:
    with pytest.raises(ValueError, match="diagnostic"):
        replace(profile().schedule, diagnostic_samples=value)


def test_diagnostic_samples_is_required_last_and_never_defaulted() -> None:
    parameters = signature(RunSchedule).parameters
    assert list(parameters)[-1] == "diagnostic_samples"
    assert parameters["diagnostic_samples"].default is Parameter.empty
    with pytest.raises(TypeError):
        RunSchedule(2, 4, 2, 2, 2, 1, (2, 4), 1, 2)  # type: ignore[call-arg]
    payload = profile().manifest
    del payload["schedule"]["diagnostic_samples"]
    with pytest.raises(ValueError, match="fields"):
        HProfile.from_manifest(payload)


def test_manifest_schema_is_the_v2_successor_and_v1_is_refused() -> None:
    value = profile()
    assert value.manifest["schema"] == "h-run-profile-v2"
    assert value.manifest["schedule"]["diagnostic_samples"] == value.schedule.diagnostic_samples
    assert HProfile.from_manifest(value.manifest) == value
    with pytest.raises(ValueError, match="unknown H profile version"):
        HProfile.from_manifest(value.manifest | {"schema": "h-run-profile-v1"})
    changed = replace(value, schedule=replace(value.schedule, diagnostic_samples=16))
    assert changed.identity != value.identity


def test_evaluated_samples_is_one_derived_tuple_outside_the_manifest() -> None:
    schedule = profile().schedule
    assert schedule.evaluated_samples == (*schedule.importance_samples, schedule.diagnostic_samples)
    assert schedule.evaluated_samples == (2, 4, 8)
    assert "evaluated_samples" not in asdict(schedule)
    assert "evaluated_samples" not in profile().manifest["schedule"]
    assert "evaluated_samples" not in signature(RunSchedule).parameters
