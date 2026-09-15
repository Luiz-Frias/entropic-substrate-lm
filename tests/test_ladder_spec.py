"""Rung mechanisms vary only on their declared scientific axis."""

from dataclasses import replace

import pytest

from d1_runtime_fixtures import runtime_family
from entropic_substrate.experiments.d1 import D1Arm
from entropic_substrate.experiments.ladder import CoarseSpec, Rung, RungSpec, Treatment
from entropic_substrate.observation import ByteObservationCodec

pytestmark = pytest.mark.unit


def d2(treatment: Treatment = Treatment.S) -> RungSpec:
    return RungSpec(runtime_family().draft_for(D1Arm.BOTH), Rung.D2, treatment, seed=7)


@pytest.mark.parametrize("horizons", [(True,), (), (2,), (4, 1), (1, 1)])
def test_horizon_roster_is_strict(horizons: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="horizons"):
        replace(d2(), horizons=horizons)


def test_language_only_predecessor_cannot_construct_later_predictive_rung() -> None:
    language = d2(Treatment.L)
    with pytest.raises(ValueError, match="predictive"):
        replace(language, rung=Rung.D3, predecessor=language)


def test_codec_rung_requires_byte_boundary_for_identity_too() -> None:
    first = d2()
    second = replace(first, rung=Rung.D3, predecessor=first)
    third = replace(second, rung=Rung.D4, coarse=CoarseSpec("fixed"), predecessor=second)
    with pytest.raises(ValueError, match="byte"):
        replace(third, rung=Rung.D5, predecessor=third)


def test_predecessor_construction_cannot_drift() -> None:
    first = d2()
    with pytest.raises(ValueError, match="predecessor"):
        replace(first, rung=Rung.D3, treatment=Treatment.G, predecessor=first)
    second = replace(first, rung=Rung.D3, horizons=(4,), predecessor=first)
    with pytest.raises(ValueError, match="horizons"):
        replace(second, rung=Rung.D4, horizons=(1,), coarse=CoarseSpec("fixed"), predecessor=second)


def test_seed_and_codec_construction_are_digest_bound() -> None:
    first = replace(d2(), base=replace(d2().base, codec=ByteObservationCodec().identity))
    second = replace(first, rung=Rung.D3, predecessor=first)
    third = replace(second, rung=Rung.D4, coarse=CoarseSpec("fixed"), predecessor=second)
    fourth = replace(third, rung=Rung.D5, predecessor=third)
    assert fourth.digest != replace(fourth, seed=8).digest
    assert fourth.digest != replace(fourth, permutation=tuple(reversed(range(256)))).digest
    assert fourth.scientific_digest == replace(fourth, seed=8).scientific_digest
    assert (
        fourth.scientific_digest
        != replace(fourth, permutation=tuple(reversed(range(256)))).scientific_digest
    )


def test_codec_control_can_follow_a_filed_short_horizon_without_chunking() -> None:
    first = replace(d2(), base=replace(d2().base, codec=ByteObservationCodec().identity))
    short_horizon = replace(first, rung=Rung.D3, predecessor=first)
    codec_control = replace(short_horizon, rung=Rung.D5, predecessor=short_horizon)
    assert codec_control.coarse is None
    assert codec_control.horizons == (1,)
    with pytest.raises(ValueError, match="predecessor"):
        replace(first, rung=Rung.D5, predecessor=first)


@pytest.mark.parametrize(
    "kwargs",
    [{"smoothing": float("nan")}, {"rate_weight": 0.0}, {"window": True}, {"random_seed": -1}],
)
def test_coarse_configuration_rejects_invalid_numerics(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CoarseSpec("learned", **kwargs)
