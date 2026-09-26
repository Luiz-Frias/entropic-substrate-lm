"""Strict external-worker control over the salted D1 alias commitment and reveal."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import replace

import pytest

from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.analyze.d1_machine_analysis import D1ArmAliasMap
from entropic_substrate.ladder.review.d1_blind_aliases import (
    D1InMemoryAliasCustodyFake,
    prepare_d1_blind_aliases,
)
from entropic_substrate.ladder.review.d1_external_analysis_control import (
    D1AliasApplicationControl,
    D1ExternalAnalysisControlError,
    decode_d1_alias_application_control,
    parse_d1_alias_commitment_control,
    parse_d1_alias_reveal_control,
)

from .test_d1_blind_alias_custody import _DeterministicCrypto

pytestmark = pytest.mark.unit


def _case():
    custody = D1InMemoryAliasCustodyFake()
    prepared = prepare_d1_blind_aliases(
        randomness=_DeterministicCrypto(),
        custody=custody,
    )
    return prepared, custody.public_bytes, custody.private_bytes


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _other_control() -> D1AliasApplicationControl:
    _, _, reveal_bytes = _case()
    reveal = json.loads(reveal_bytes)
    reveal["nonce"] = "f" * 64
    changed_reveal = _canonical(reveal)
    changed_commitment = _canonical(
        {
            "reveal_sha256": hashlib.sha256(changed_reveal).hexdigest(),
            "schema_version": "d1-alias-commitment-v1",
        }
    )
    return decode_d1_alias_application_control(changed_commitment, changed_reveal)


def _verified_control() -> D1AliasApplicationControl:
    _, commitment_bytes, reveal_bytes = _case()
    return decode_d1_alias_application_control(commitment_bytes, reveal_bytes)


def _clone_fields(source: D1AliasApplicationControl) -> D1AliasApplicationControl:
    clone = object.__new__(D1AliasApplicationControl)
    for name in ("commitment", "reveal", "aliases", "commitment_sha256", "reveal_sha256"):
        object.__setattr__(clone, name, getattr(source, name))
    return clone


def _mismatched_aliases(source: D1AliasApplicationControl) -> D1ArmAliasMap:
    aliases = dict(source.aliases.aliases)
    aliases[D1Arm.SG], aliases[D1Arm.AC] = aliases[D1Arm.AC], aliases[D1Arm.SG]
    return D1ArmAliasMap(aliases, sealed_commitment=source.commitment_sha256)


def test_existing_custody_bytes_decode_to_exact_analysis_alias_map() -> None:
    prepared, commitment_bytes, reveal_bytes = _case()

    commitment = parse_d1_alias_commitment_control(commitment_bytes)
    reveal = parse_d1_alias_reveal_control(reveal_bytes)
    control = decode_d1_alias_application_control(commitment_bytes, reveal_bytes)

    assert commitment.canonical_bytes == commitment_bytes
    assert reveal.canonical_bytes == reveal_bytes
    assert reveal.nonce == bytes(range(32))
    assert set(reveal.mapping) == {"A", "B", "C", "D"}
    assert set(reveal.mapping.values()) == {arm.value for arm in D1Arm}
    assert control.aliases == prepared.aliases
    assert control.commitment_sha256 == hashlib.sha256(commitment_bytes).hexdigest()
    assert control.reveal_sha256 == hashlib.sha256(reveal_bytes).hexdigest()
    assert control.aliases.sealed_commitment == control.commitment_sha256


@pytest.mark.parametrize(
    ("which", "payload"),
    [
        (
            "commitment",
            b'{"reveal_sha256":"'
            + b"1" * 64
            + b'","reveal_sha256":"'
            + b"2" * 64
            + b'","schema_version":"d1-alias-commitment-v1"}',
        ),
        ("commitment", b'{"reveal_sha256":NaN,"schema_version":"d1-alias-commitment-v1"}'),
        (
            "commitment",
            b' {"reveal_sha256":"' + b"1" * 64 + b'","schema_version":"d1-alias-commitment-v1"}',
        ),
        (
            "reveal",
            b'{"mapping":{"A":"none","A":"sg"},"nonce":"'
            + b"0" * 64
            + b'","schema_version":"d1-alias-reveal-v1"}',
        ),
        ("reveal", b"[]"),
    ],
)
def test_control_decoders_reject_duplicate_nonfinite_noncanonical_or_nonobject_bytes(
    which: str, payload: bytes
) -> None:
    with pytest.raises(D1ExternalAnalysisControlError):
        if which == "commitment":
            parse_d1_alias_commitment_control(payload)
        else:
            parse_d1_alias_reveal_control(payload)


def test_commitment_must_bind_exact_reveal_bytes() -> None:
    _, commitment_bytes, reveal_bytes = _case()
    reveal = json.loads(reveal_bytes)
    reveal["nonce"] = "f" * 64
    changed_reveal = _canonical(reveal)

    with pytest.raises(D1ExternalAnalysisControlError, match="reveal_sha256"):
        decode_d1_alias_application_control(commitment_bytes, changed_reveal)


@pytest.mark.parametrize(
    "nonce",
    [
        "0" * 62,
        "0" * 66,
        "A" * 64,
        "g" * 64,
    ],
)
def test_reveal_nonce_is_exactly_32_lowercase_hex_bytes(nonce: str) -> None:
    _, _, reveal_bytes = _case()
    reveal = json.loads(reveal_bytes)
    reveal["nonce"] = nonce

    with pytest.raises(D1ExternalAnalysisControlError, match="nonce"):
        parse_d1_alias_reveal_control(_canonical(reveal))


@pytest.mark.parametrize(
    "mapping",
    [
        {"A": "none", "B": "sg", "C": "ac"},
        {"A": "none", "B": "sg", "C": "ac", "E": "both"},
        {"A": "none", "B": "sg", "C": "ac", "D": "ac"},
        {"A": "none", "B": "sg", "C": "ac", "D": "invented"},
    ],
)
def test_reveal_requires_exact_alias_and_arm_bijection(mapping: dict[str, str]) -> None:
    _, _, reveal_bytes = _case()
    reveal = json.loads(reveal_bytes)
    reveal["mapping"] = mapping

    with pytest.raises(D1ExternalAnalysisControlError, match="bijection"):
        parse_d1_alias_reveal_control(_canonical(reveal))


def test_control_requires_byte_payloads_and_lowercase_commitment_digest() -> None:
    _, commitment_bytes, reveal_bytes = _case()
    commitment = json.loads(commitment_bytes)
    commitment["reveal_sha256"] = str(commitment["reveal_sha256"]).upper()
    with pytest.raises(D1ExternalAnalysisControlError, match="lowercase"):
        parse_d1_alias_commitment_control(_canonical(commitment))
    with pytest.raises(TypeError, match="bytes"):
        decode_d1_alias_application_control("not-bytes", reveal_bytes)  # type: ignore[arg-type]


def test_verified_alias_application_is_factory_only() -> None:
    _, commitment_bytes, reveal_bytes = _case()
    verified = decode_d1_alias_application_control(commitment_bytes, reveal_bytes)

    with pytest.raises(TypeError, match="factory-only"):
        D1AliasApplicationControl(
            commitment=verified.commitment,
            reveal=verified.reveal,
            aliases=verified.aliases,
            commitment_sha256=verified.commitment_sha256,
            reveal_sha256=verified.reveal_sha256,
        )


def test_factory_attestation_cannot_be_replayed_by_dataclass_replacement() -> None:
    verified = _verified_control()
    other = _other_control()
    verified.require_factory_attested()

    with pytest.raises(TypeError, match="factory-only"):
        replace(verified)
    with pytest.raises(TypeError, match="factory-only"):
        replace(
            verified,
            commitment=other.commitment,
            reveal=other.reveal,
            aliases=other.aliases,
            commitment_sha256=other.commitment_sha256,
            reveal_sha256=other.reveal_sha256,
        )


@pytest.mark.parametrize("copier", (copy, deepcopy, pickle.dumps))
def test_factory_attested_control_cannot_be_copied(
    copier: Callable[[object], object],
) -> None:
    with pytest.raises(TypeError, match="cannot be copied"):
        copier(_verified_control())


def test_forged_attestation_markers_are_rejected_by_distinct_guards() -> None:
    verified = _verified_control()
    forged = _clone_fields(verified)
    object.__setattr__(
        forged,
        "_attestation",
        object.__getattribute__(verified, "_attestation"),
    )
    with pytest.raises(TypeError, match="not minted"):
        forged.require_factory_attested()

    unattested = _clone_fields(verified)
    with pytest.raises(TypeError, match="not minted"):
        unattested.require_factory_attested()

    self_owned = _clone_fields(verified)
    attestation_type = type(object.__getattribute__(verified, "_attestation"))
    object.__setattr__(self_owned, "_attestation", attestation_type(self_owned))
    with pytest.raises(TypeError, match="not minted"):
        self_owned.require_factory_attested()

    missing_marker = _verified_control()
    object.__delattr__(missing_marker, "_attestation")
    with pytest.raises(TypeError, match="marker is absent"):
        missing_marker.require_factory_attested()

    wrong_owner = _verified_control()
    object.__setattr__(
        wrong_owner,
        "_attestation",
        object.__getattribute__(_other_control(), "_attestation"),
    )
    with pytest.raises(TypeError, match="another instance"):
        wrong_owner.require_factory_attested()


def test_minted_control_revalidates_values_after_low_level_tampering() -> None:
    mutations: tuple[
        tuple[Callable[[D1AliasApplicationControl], None], str],
        ...,
    ] = (
        (
            lambda value: object.__setattr__(value, "commitment_sha256", "0" * 64),
            "commitment digest",
        ),
        (
            lambda value: object.__setattr__(value, "reveal_sha256", "0" * 64),
            "reveal digest",
        ),
        (
            lambda value: object.__setattr__(value, "aliases", _mismatched_aliases(value)),
            "mapping differs",
        ),
    )
    for mutation, message in mutations:
        verified = _verified_control()
        mutation(verified)
        with pytest.raises(ValueError, match=message):
            verified.require_factory_attested()
