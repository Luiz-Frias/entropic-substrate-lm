"""The sealed H run authority: strict construction, canonical bytes, a digest that binds them.

Rung B (operator ruling 2026-10-27, "option B"): a ``pilot`` or ``confirmatory`` H run
is admissible only under a sealed ``RunAuthority``. These tests pin the artifact on its
own; the admission port and the gate sites that read it are pinned in
``tests/test_h_run_admission.py`` and ``tests/test_h_run_admission_sites.py``.
"""

import json
from dataclasses import replace
from typing import Any

import pytest

from entropic_substrate.ladder.run.run_authority import (
    RUN_AUTHORITY_SCHEMA,
    AuthorityRuling,
    RunAuthority,
    SpendCap,
    parse_run_authority,
    seal_run_authority,
)

pytestmark = pytest.mark.unit

REGISTRATION = "a" * 40


def ruling(**changes: object) -> AuthorityRuling:
    fields: dict[str, Any] = {
        "date": "2026-10-27",
        "text": "Run the H1 pilot: 8 paired 2x2 blocks at the development profile.",
        "operator": "operator",
    }
    fields.update(changes)
    return AuthorityRuling(**fields)


def authority(**changes: object) -> RunAuthority:
    fields: dict[str, Any] = {
        "stage": "H1",
        "run_class": "pilot",
        "ruling": ruling(),
        "spend_cap": SpendCap(usd=40.0, hours=6.0),
        "gpu_class": "NVIDIA A100 80GB PCIe",
        "registration_sha": REGISTRATION,
    }
    fields.update(changes)
    return seal_run_authority(**fields)


def test_seal_roundtrips_through_canonical_bytes() -> None:
    value = authority()
    assert value.verify()
    assert parse_run_authority(json.loads(value.payload)) == value
    assert value.manifest["schema"] == RUN_AUTHORITY_SCHEMA
    assert value.manifest["digest"] == value.digest
    assert (
        value.payload
        == json.dumps(
            value.manifest, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    )


def test_the_digest_binds_every_field() -> None:
    base = authority()
    variants = [
        authority(stage="H2"),
        authority(run_class="confirmatory"),
        authority(ruling=ruling(date="2026-10-28")),
        authority(ruling=ruling(text="another ruling")),
        authority(ruling=ruling(operator="someone-else")),
        authority(spend_cap=SpendCap(usd=41.0, hours=6.0)),
        authority(spend_cap=SpendCap(usd=40.0, hours=7.0)),
        authority(gpu_class="NVIDIA H100 80GB HBM3"),
        authority(registration_sha="b" * 40),
    ]
    assert len({v.digest for v in (base, *variants)}) == len(variants) + 1


def test_an_unknown_gpu_class_is_admitted_and_recorded() -> None:
    value = authority(gpu_class="Some Future GPU 9000")
    assert value.manifest["gpu_class"] == "Some Future GPU 9000"


# ------------------------------------------------------------------ refusals
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"stage": "H9"}, "run authority stage is not an H stage"),
        ({"stage": "h1"}, "run authority stage is not an H stage"),
        ({"run_class": "remote_tiny"}, "run authority names a non-scientific run class"),
        ({"run_class": "local_tiny"}, "run authority names a non-scientific run class"),
        ({"gpu_class": ""}, "run authority GPU class must be a safe explicit identifier"),
        ({"gpu_class": " A100"}, "run authority GPU class must be a safe explicit identifier"),
        ({"gpu_class": "A100;rm"}, "run authority GPU class must be a safe explicit identifier"),
        ({"gpu_class": 100}, "run authority GPU class must be a safe explicit identifier"),
        ({"registration_sha": "a" * 39}, "registration SHA must be a full 40-hex git SHA"),
        ({"registration_sha": "a" * 41}, "registration SHA must be a full 40-hex git SHA"),
        ({"registration_sha": "A" * 40}, "registration SHA must be a full 40-hex git SHA"),
        ({"registration_sha": "a" * 64}, "registration SHA must be a full 40-hex git SHA"),
        ({"registration_sha": None}, "registration SHA must be a full 40-hex git SHA"),
        ({"ruling": {"date": "2026-10-27"}}, "run authority ruling must be an AuthorityRuling"),
        ({"spend_cap": (40.0, 6.0)}, "run authority spend cap must be a SpendCap"),
    ],
)
def test_construction_refuses_by_name(changes: dict[str, object], message: str) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        authority(**changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"date": "2026-13-01"}, "ruling date must be an ISO calendar date"),
        ({"date": "20261027"}, "ruling date must be an ISO calendar date"),
        ({"date": "2026-10-27T00:00"}, "ruling date must be an ISO calendar date"),
        ({"date": ""}, "ruling date must be an ISO calendar date"),
        ({"date": 20261027}, "ruling date must be an ISO calendar date"),
        ({"text": ""}, "ruling text must be non-empty text"),
        ({"text": "   "}, "ruling text must be non-empty text"),
        ({"text": None}, "ruling text must be non-empty text"),
        ({"operator": ""}, "ruling operator must be non-empty text"),
        ({"operator": " op"}, "ruling operator must be non-empty text"),
    ],
)
def test_ruling_refuses_by_name(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ruling(**changes)


@pytest.mark.parametrize(
    ("usd", "hours"),
    [
        (0, 6.0),
        (0.0, 6.0),
        (-1.0, 6.0),
        (40.0, 0),
        (40.0, -6.0),
        (float("inf"), 6.0),
        (float("nan"), 6.0),
        (40.0, float("inf")),
        (True, 6.0),
        ("40", 6.0),
        (None, 6.0),
    ],
)
def test_spend_cap_must_be_finite_and_positive(usd: object, hours: object) -> None:
    with pytest.raises(ValueError, match="spend cap must be finite positive numbers"):
        SpendCap(usd=usd, hours=hours)  # type: ignore[arg-type]


def test_integer_caps_are_admitted() -> None:
    assert authority(spend_cap=SpendCap(usd=40, hours=6)).verify()


# ------------------------------------------------------------ digest and tamper
def test_a_forged_digest_is_refused_at_construction() -> None:
    sealed = authority()
    with pytest.raises(ValueError, match="run authority digest does not match its content"):
        replace(sealed, digest="0" * 64)


def test_an_edited_field_with_the_old_digest_is_refused() -> None:
    sealed = authority()
    with pytest.raises(ValueError, match="run authority digest does not match its content"):
        replace(sealed, stage="H2")


@pytest.mark.parametrize("digest", ["0" * 63, "G" * 64, None, ""])
def test_a_malformed_digest_is_refused(digest: object) -> None:
    sealed = authority()
    with pytest.raises(ValueError, match="run authority digest must be 64 lowercase hex"):
        replace(sealed, digest=digest)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("stage", "H2"),
        ("run_class", "confirmatory"),
        ("gpu_class", "NVIDIA H100 80GB HBM3"),
        ("registration_sha", "b" * 40),
        ("digest", "0" * 64),
    ],
)
def test_tamper_after_sealing_fails_verification(name: str, value: object) -> None:
    sealed = authority()
    object.__setattr__(sealed, name, value)
    assert not sealed.verify()


def test_tamper_inside_a_nested_part_fails_verification() -> None:
    sealed = authority()
    object.__setattr__(sealed.spend_cap, "usd", 4000.0)
    assert not sealed.verify()
    sealed = authority()
    object.__setattr__(sealed.ruling, "date", "not-a-date")
    assert not sealed.verify()


def test_tamper_to_an_invalid_value_fails_verification_rather_than_raising() -> None:
    sealed = authority()
    object.__setattr__(sealed, "stage", object())
    assert not sealed.verify()


# --------------------------------------------------------------------- parse
def test_parse_refuses_another_schema_and_extra_or_missing_keys() -> None:
    manifest = authority().manifest
    with pytest.raises(ValueError, match="run authority schema is not admitted"):
        parse_run_authority({**manifest, "schema": "h-run-authority-v0"})
    with pytest.raises(ValueError, match="fields differ"):
        parse_run_authority({**manifest, "extra": 1})
    missing = dict(manifest)
    del missing["gpu_class"]
    with pytest.raises(ValueError, match="fields differ"):
        parse_run_authority(missing)
    with pytest.raises(ValueError, match="fields differ"):
        parse_run_authority([manifest])


def test_parse_refuses_a_tampered_payload() -> None:
    manifest = json.loads(authority().payload)
    manifest["spend_cap"]["usd"] = 4000.0
    with pytest.raises(ValueError, match="run authority digest does not match its content"):
        parse_run_authority(manifest)


def test_parse_keeps_the_numeric_type_of_the_cap() -> None:
    sealed = authority(spend_cap=SpendCap(usd=40, hours=6))
    parsed = parse_run_authority(json.loads(sealed.payload))
    assert parsed.payload == sealed.payload
    assert parsed == sealed
