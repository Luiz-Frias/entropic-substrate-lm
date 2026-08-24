"""D1 four-arm aliases are randomized, salted, and durably separated."""

from __future__ import annotations

import hashlib
import json

import pytest

from entropic_substrate.analysis.d1_blind_aliases import (
    D1AliasArtifactRole,
    D1AliasStorageScope,
    D1InMemoryAliasCustodyFake,
    prepare_d1_blind_aliases,
)
from entropic_substrate.experiments.d1_primitives import D1Arm

pytestmark = pytest.mark.unit


class _DeterministicCrypto:
    """Contract fake whose draws exercise every Fisher-Yates exchange."""

    def __init__(self) -> None:
        self._draws = iter((1, 0, 1))

    def randbelow(self, upper: int) -> int:
        value = next(self._draws)
        assert 0 <= value < upper
        return value

    def token_bytes(self, length: int) -> bytes:
        assert length == 32
        return bytes(range(length))


def test_preparation_randomizes_all_four_aliases_and_commits_salted_reveal() -> None:
    repository = D1InMemoryAliasCustodyFake()

    prepared = prepare_d1_blind_aliases(randomness=_DeterministicCrypto(), custody=repository)

    assert set(prepared.aliases.aliases) == set(D1Arm)
    assert set(prepared.aliases.aliases.values()) == {"A", "B", "C", "D"}
    assert prepared.aliases.aliases != {
        D1Arm.NONE: "A",
        D1Arm.SG: "B",
        D1Arm.AC: "C",
        D1Arm.BOTH: "D",
    }
    reveal = json.loads(repository.read_private(prepared.private_reveal))
    assert reveal == {
        "mapping": {alias: arm.value for arm, alias in prepared.aliases.aliases.items()},
        "nonce": bytes(range(32)).hex(),
        "schema_version": "d1-alias-reveal-v1",
    }
    commitment = json.loads(repository.public_bytes)
    assert commitment == {
        "reveal_sha256": hashlib.sha256(repository.private_bytes).hexdigest(),
        "schema_version": "d1-alias-commitment-v1",
    }
    assert prepared.aliases.sealed_commitment == hashlib.sha256(repository.public_bytes).hexdigest()


def test_commitment_is_durable_before_private_packet_excluded_reveal() -> None:
    repository = D1InMemoryAliasCustodyFake()

    prepared = prepare_d1_blind_aliases(randomness=_DeterministicCrypto(), custody=repository)

    assert repository.write_order == (
        D1AliasArtifactRole.COMMITMENT,
        D1AliasArtifactRole.PRIVATE_REVEAL,
    )
    assert prepared.commitment.durable is True
    assert prepared.private_reveal.durable is True
    assert prepared.commitment.sequence < prepared.private_reveal.sequence
    assert prepared.commitment.scope is D1AliasStorageScope.PUBLIC
    assert prepared.private_reveal.scope is D1AliasStorageScope.PACKET_EXCLUDED
    assert prepared.commitment.path != prepared.private_reveal.path


def test_preparation_refuses_non_durable_or_same_scope_custody_receipts() -> None:
    repository = D1InMemoryAliasCustodyFake(private_scope=D1AliasStorageScope.PUBLIC)

    with pytest.raises(ValueError, match="packet-excluded"):
        prepare_d1_blind_aliases(randomness=_DeterministicCrypto(), custody=repository)


def test_system_requires_exact_32_byte_salt() -> None:
    class _ShortSalt(_DeterministicCrypto):
        def token_bytes(self, length: int) -> bytes:
            return b"short"

    with pytest.raises(ValueError, match="32 bytes"):
        prepare_d1_blind_aliases(randomness=_ShortSalt(), custody=D1InMemoryAliasCustodyFake())
