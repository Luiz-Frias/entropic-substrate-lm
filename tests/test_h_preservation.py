"""H1 profiles isolate two normalization axes without conferring run authority."""

import json
from dataclasses import replace

import pytest
from tests.test_sentence_profile import profile

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.ladder.register.h_preservation import H1PreservationBlock

pytestmark = pytest.mark.unit


def profiles() -> tuple[HProfile, ...]:
    base = profile()
    base = replace(
        base,
        stage="H1",
        name="h1-base-tiny",
        consumer="isotropic",
        optimization=replace(base.optimization, future_weight=0),
    )
    return tuple(
        replace(
            base,
            name=f"h1-{sentence}-{spine}-tiny",
            law=replace(base.law, mode=sentence),
            spine=replace(base.spine, carrier=spine),
        )
        for sentence, spine in (
            ("magnitude", "magnitude"),
            ("sphere", "magnitude"),
            ("magnitude", "sphere"),
            ("sphere", "sphere"),
        )
    )


def test_complete_block_is_canonical_and_has_no_scientific_authority() -> None:
    values = profiles()
    block = H1PreservationBlock(tuple(reversed(values)))
    assert block.profiles == values
    assert block.identity == H1PreservationBlock(values).identity
    assert H1PreservationBlock.from_manifest(json.loads(json.dumps(block.manifest))) == block
    assert block.manifest["scientific_authority"] == "none"
    assert block.codec_sharing_groups == (
        (values[0].name, values[2].name),
        (values[1].name, values[3].name),
    )


def test_factorial_terms_are_main_effects_and_difference_of_differences() -> None:
    block = H1PreservationBlock(profiles())
    measurements = dict(zip((p.name for p in block.profiles), (0, 1, 10, 12), strict=True))
    results = {
        name: sum(measurements[arm] * weight for arm, weight in terms)
        for name, terms in block.factorial_terms.items()
    }
    assert results == {"sentence_normalization": 1.5, "spine_normalization": 10.5, "interaction": 1}
    assert all(sum(weight for _, weight in terms) == 0 for terms in block.factorial_terms.values())


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "seed",
        "source_seed",
        "source",
        "schedule",
        "optimizer",
        "dispersion",
        "context",
        "codec",
        "stage",
        "duplicate_name",
    ],
)
def test_incomplete_or_confounded_block_is_rejected(mutation: str) -> None:
    values = list(profiles())
    first = values[0]
    if mutation == "missing":
        values.pop()
    elif mutation == "duplicate":
        values[3] = replace(first, name=values[3].name)
    elif mutation == "seed":
        values[0] = replace(first, seed=43)
    elif mutation == "source_seed":
        values[0] = replace(first, source_seed=43)
    elif mutation == "source":
        values[0] = replace(first, source_digest="1" * 64)
    elif mutation == "schedule":
        values[0] = replace(first, schedule=replace(first.schedule, warmup_steps=3))
    elif mutation == "optimizer":
        values[0] = replace(first, optimization=replace(first.optimization, learning_rate=0.002))
    elif mutation == "dispersion":
        values[0] = replace(first, law=replace(first.law, mean_resultant=0.8))
    elif mutation == "context":
        values[0] = replace(first, spine=replace(first.spine, context_sentences=5))
    elif mutation == "codec":
        values[0] = replace(first, codec=replace(first.codec, encoder_layers=2))
    elif mutation == "stage":
        values[0] = replace(first, stage="H2", name="h2-other-tiny")
    else:
        values[0] = replace(first, name=values[1].name)
    with pytest.raises(ValueError, match="H1"):
        H1PreservationBlock(tuple(values))


@pytest.mark.parametrize(
    "mutation", ["authority", "missing", "foreign_schema", "extra", "non_list", "short_list"]
)
def test_manifest_does_not_invent_policy_or_accept_authority_claims(mutation: str) -> None:
    value = H1PreservationBlock(profiles()).manifest
    if mutation == "authority":
        value["scientific_authority"] = "authorized"
    elif mutation == "missing":
        del value["scientific_authority"]
    elif mutation == "foreign_schema":
        value["schema"] = "d-ladder-statistical-design-v1"
    elif mutation == "non_list":
        value["profiles"] = None
    elif mutation == "short_list":
        value["profiles"] = []
    else:
        value["epsilon"] = 0.01
    with pytest.raises(ValueError):
        H1PreservationBlock.from_manifest(value)


@pytest.mark.parametrize("mechanism", ["fisher", "future_law"])
def test_forged_profiles_cannot_bundle_other_mechanisms(mechanism: str) -> None:
    values = profiles()
    if mechanism == "fisher":
        object.__setattr__(values[0], "consumer", "fisher")
    else:
        object.__setattr__(values[0].optimization, "future_weight", 0.3)
    with pytest.raises(ValueError, match="H1"):
        H1PreservationBlock(values)


def test_block_owns_revalidated_profiles_not_caller_aliases() -> None:
    values = profiles()
    block = H1PreservationBlock(values)
    original = block.identity
    object.__setattr__(values[0], "seed", 43)
    assert block.identity == original and block.profiles[0].seed == 42


@pytest.mark.parametrize("value", [None, (), []])
def test_empty_or_untyped_block_cannot_become_a_design(value: object) -> None:
    with pytest.raises(ValueError, match="H1"):
        H1PreservationBlock(value)


def test_block_rejects_non_profile_members_even_with_four_cells() -> None:
    with pytest.raises(TypeError, match="profiles"):
        H1PreservationBlock((None, None, None, None))
