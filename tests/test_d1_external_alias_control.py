"""Strict external-worker control over the salted D1 alias commitment and reveal."""

from __future__ import annotations

import hashlib
import json

import pytest

from entropic_substrate.analysis.d1_blind_aliases import (
    D1InMemoryAliasCustodyFake,
    prepare_d1_blind_aliases,
)
from entropic_substrate.analysis.d1_external_analysis_control import (
    D1AliasApplicationControl,
    D1ExternalAnalysisControlError,
    decode_d1_alias_application_control,
    parse_d1_alias_commitment_control,
    parse_d1_alias_reveal_control,
)
from entropic_substrate.experiments.d1_primitives import D1Arm

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
