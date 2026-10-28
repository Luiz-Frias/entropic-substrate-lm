"""H1 profiles isolate two normalization axes without conferring run authority."""

import json
from dataclasses import replace

import pytest
from tests.test_sentence_profile import profile

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.ladder.register.h_preservation import (
    H1ExclusionReason,
    H1PreservationBlock,
    h1_construction_digest,
    h1_exposure_bytes,
)

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


# Golden: profiles()[0] = h1-magnitude-magnitude-tiny (stage H1, run_class local_tiny,
# law.mode magnitude, spine.carrier magnitude, schema h-run-profile-v3) with the top-level
# keys name, seed and source_seed removed. Not comparable to any pilot profile.
_GOLDEN_CONSTRUCTION_DIGEST = (
    "52f2a51ebd8142e3aacc47c3c57c13a57e4835e61507207e6cb0a7fa56a180cb"  # pragma: allowlist secret
)
# Cross-check that the block identity bytes did not move with this phase.
_GOLDEN_BLOCK_IDENTITY = (
    "bc90190a8b15acde8bf99a7735e0db256f4e6e14fefc4e3803e4c657ba5edf4a"  # pragma: allowlist secret
)


def test_block_identity_bytes_are_unmoved() -> None:
    assert H1PreservationBlock(profiles()).identity == _GOLDEN_BLOCK_IDENTITY


def test_construction_digest_golden_is_full_lowercase_hex() -> None:
    digest = h1_construction_digest(profiles()[0])
    assert digest == _GOLDEN_CONSTRUCTION_DIGEST
    assert len(digest) == 64 and digest == digest.lower()


def test_construction_digest_ignores_name_seed_and_source_seed() -> None:
    first = profiles()[0]
    other = replace(first, name="h1-other-arm-tiny", seed=1001, source_seed=2002)
    assert h1_construction_digest(other) == h1_construction_digest(first)
    for changed in (
        replace(first, name="h1-renamed-tiny"),
        replace(first, seed=7),
        replace(first, source_seed=7),
    ):
        assert h1_construction_digest(changed) == h1_construction_digest(first)


def test_construction_digest_does_not_mutate_the_profile() -> None:
    first = profiles()[0]
    before = first.identity
    h1_construction_digest(first)
    assert first.identity == before and "name" in first.manifest


def test_construction_digest_separates_the_four_cells() -> None:
    digests = {h1_construction_digest(p) for p in profiles()}
    assert len(digests) == 4


@pytest.mark.parametrize(
    "mutation", ["carrier", "law_mode", "spine_width", "source_digest", "schedule"]
)
def test_construction_digest_moves_with_every_construction_field(mutation: str) -> None:
    first = profiles()[0]
    if mutation == "carrier":
        changed = replace(first, spine=replace(first.spine, carrier="sphere"))
    elif mutation == "law_mode":
        changed = replace(first, law=replace(first.law, mode="sphere"))
    elif mutation == "spine_width":
        changed = replace(first, spine=replace(first.spine, width=16))
    elif mutation == "source_digest":
        changed = replace(first, source_digest="1" * 64)
    else:
        changed = replace(first, schedule=replace(first.schedule, warmup_steps=3))
    assert h1_construction_digest(changed) != h1_construction_digest(first)


def test_construction_digest_is_not_the_profile_identity() -> None:
    first = profiles()[0]
    assert h1_construction_digest(first) != first.identity


@pytest.mark.parametrize("value", [None, "profile", {"name": "x"}, profiles()])
def test_construction_digest_refuses_non_profiles(value: object) -> None:
    with pytest.raises(TypeError, match="profile"):
        h1_construction_digest(value)  # type: ignore[arg-type]


def test_exclusion_reason_is_closed_to_exactly_five_members() -> None:
    assert {member.name: member.value for member in H1ExclusionReason} == {
        "NONFINITE_ENDPOINT": "nonfinite_endpoint",
        "MISSING_ENDPOINT": "missing_endpoint",
        "READER_REFUSAL": "reader_refusal",
        "INADMISSIBLE_ESS": "inadmissible_ess",
        "INDEPENDENT_WARMUP": "independent_warmup",
    }
    assert len(H1ExclusionReason) == 5


@pytest.mark.parametrize("value", ["anything_else", "", "NONFINITE_ENDPOINT", None, 3])
def test_exclusion_reason_unknown_value_raises(value: object) -> None:
    with pytest.raises(ValueError):
        H1ExclusionReason(value)


def test_exclusion_reason_lookup_round_trips_every_value() -> None:
    for member in H1ExclusionReason:
        assert H1ExclusionReason(member.value) is member


def _row(
    phase: str = "prediction",
    status: str = "committed",
    source_bytes: object = ...,
) -> dict[str, object]:
    row: dict[str, object] = {"schema": "h-training-step-v2", "phase": phase, "status": status}
    if source_bytes is not ...:
        row["source_bytes"] = source_bytes
    return row


def test_exposure_sums_committed_prediction_rows_only() -> None:
    rows = [
        _row("codec_warmup", "committed", 1000),
        _row("prediction", "committed", 10),
        _row("prediction", "committed", 32),
        _row("prediction", "started", 500),
        _row("codec_warmup", "started", 700),
    ]
    assert h1_exposure_bytes(rows) == 42


def test_exposure_of_empty_input_is_zero() -> None:
    assert h1_exposure_bytes([]) == 0
    assert h1_exposure_bytes(iter(())) == 0


def test_exposure_consumes_a_one_shot_iterator() -> None:
    assert h1_exposure_bytes(_row(source_bytes=n) for n in (1, 2, 3)) == 6


def test_exposure_warmup_only_is_zero_even_with_bytes() -> None:
    assert h1_exposure_bytes([_row("codec_warmup", "committed", 99)]) == 0


def test_exposure_does_not_dedupe_or_reorder() -> None:
    row = _row(source_bytes=5)
    assert h1_exposure_bytes([row, row]) == 10


def test_exposure_zero_bytes_row_is_admissible() -> None:
    assert h1_exposure_bytes([_row(source_bytes=0)]) == 0


@pytest.mark.parametrize("status", ["committed", "started", "scientific_gap", "execution_defect"])
def test_exposure_prediction_row_without_source_bytes_raises(status: str) -> None:
    with pytest.raises(ValueError, match="source_bytes"):
        h1_exposure_bytes([_row("prediction", status)])


@pytest.mark.parametrize("bad", [None, -1, 1.5, True, "7"])
def test_exposure_prediction_row_with_bad_source_bytes_raises(bad: object) -> None:
    with pytest.raises(ValueError, match="source_bytes"):
        h1_exposure_bytes([_row(source_bytes=bad)])


@pytest.mark.parametrize(
    "row",
    [
        {"status": "committed", "source_bytes": 1},
        {"phase": "evaluation", "status": "committed", "source_bytes": 1},
        {"phase": "prediction", "source_bytes": 1},
        {"phase": "prediction", "status": "commited", "source_bytes": 1},
        {"phase": "prediction", "status": None, "source_bytes": 1},
    ],
)
def test_exposure_unrecognized_rows_raise_never_skip(row: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="row"):
        h1_exposure_bytes([row])


def test_exposure_non_mapping_row_raises() -> None:
    with pytest.raises(TypeError, match="row"):
        h1_exposure_bytes([("prediction", "committed")])  # type: ignore[list-item]
