"""Salted four-arm alias generation and packet-excluded custody for D1."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Protocol

from entropic_substrate.analysis.d1_machine_analysis import D1ArmAliasMap
from entropic_substrate.evidence.d1_types import D1_OPAQUE_ARM_ALIAS_IDS
from entropic_substrate.experiments.d1_primitives import D1Arm

__all__ = [
    "D1AliasArtifactReceipt",
    "D1AliasArtifactRole",
    "D1AliasCustodyPort",
    "D1AliasRandomPort",
    "D1AliasStorageScope",
    "D1InMemoryAliasCustodyFake",
    "D1PreparedBlindAliases",
    "D1SystemAliasRandom",
    "prepare_d1_blind_aliases",
]

_ALIASES = D1_OPAQUE_ARM_ALIAS_IDS
_SALT_BYTES = 32


class D1AliasArtifactRole(StrEnum):
    COMMITMENT = "alias_commitment"
    PRIVATE_REVEAL = "private_alias_reveal"


class D1AliasStorageScope(StrEnum):
    PUBLIC = "public"
    PACKET_EXCLUDED = "packet_excluded"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class D1AliasArtifactReceipt:
    role: D1AliasArtifactRole
    scope: D1AliasStorageScope
    path: str
    sha256: str
    durable: bool
    sequence: int

    def __post_init__(self) -> None:
        if type(self.role) is not D1AliasArtifactRole:
            raise TypeError("D1 alias artifact role must be nominal")
        if type(self.scope) is not D1AliasStorageScope:
            raise TypeError("D1 alias artifact scope must be nominal")
        parsed = PurePosixPath(self.path)
        if (
            type(self.path) is not str
            or not self.path
            or parsed.is_absolute()
            or ".." in parsed.parts
            or parsed.as_posix() != self.path
        ):
            raise ValueError("D1 alias artifact path must be normalized and relative")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or self.sha256 != self.sha256.lower()
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("D1 alias artifact sha256 must be lowercase SHA-256")
        if type(self.durable) is not bool:
            raise TypeError("D1 alias artifact durability must be bool")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("D1 alias artifact sequence must be nonnegative")


class D1AliasRandomPort(Protocol):
    """Cryptographic draws are injected so the shuffle contract is testable."""

    def randbelow(self, upper: int) -> int: ...

    def token_bytes(self, length: int) -> bytes: ...


class D1AliasCustodyPort(Protocol):
    """Separate public commitment storage from packet-excluded reveal custody."""

    def store_public_commitment(self, payload: bytes) -> D1AliasArtifactReceipt: ...

    def store_private_reveal(self, payload: bytes) -> D1AliasArtifactReceipt: ...

    def read_private(self, receipt: D1AliasArtifactReceipt) -> bytes: ...


class D1SystemAliasRandom:
    """Production cryptographic randomness adapter."""

    def randbelow(self, upper: int) -> int:
        if type(upper) is not int or upper < 1:
            raise ValueError("D1 alias random upper bound must be positive")
        return secrets.randbelow(upper)

    def token_bytes(self, length: int) -> bytes:
        if type(length) is not int or length < 1:
            raise ValueError("D1 alias random byte length must be positive")
        return secrets.token_bytes(length)


@dataclass(frozen=True, slots=True)
class D1PreparedBlindAliases:
    aliases: D1ArmAliasMap
    commitment: D1AliasArtifactReceipt
    private_reveal: D1AliasArtifactReceipt

    def __post_init__(self) -> None:
        if type(self.aliases) is not D1ArmAliasMap:
            raise TypeError("D1 prepared aliases require D1ArmAliasMap")
        if self.commitment.role is not D1AliasArtifactRole.COMMITMENT:
            raise ValueError("D1 prepared aliases require a commitment receipt")
        if self.private_reveal.role is not D1AliasArtifactRole.PRIVATE_REVEAL:
            raise ValueError("D1 prepared aliases require a private reveal receipt")
        if self.commitment.scope is not D1AliasStorageScope.PUBLIC:
            raise ValueError("D1 alias commitment must use public custody")
        if self.private_reveal.scope is not D1AliasStorageScope.PACKET_EXCLUDED:
            raise ValueError("D1 alias reveal must use packet-excluded custody")
        if not self.commitment.durable or not self.private_reveal.durable:
            raise ValueError("D1 alias custody receipts must be durable")
        if self.commitment.sequence >= self.private_reveal.sequence:
            raise ValueError("D1 alias commitment must be durable before private reveal")
        if self.commitment.path == self.private_reveal.path:
            raise ValueError("D1 public commitment and private reveal paths must differ")


def _shuffle_aliases(randomness: D1AliasRandomPort) -> dict[D1Arm, str]:
    labels = list(_ALIASES)
    for upper in range(len(labels), 1, -1):
        selected = randomness.randbelow(upper)
        if type(selected) is not int or not 0 <= selected < upper:
            raise ValueError("D1 cryptographic random port returned an out-of-range draw")
        labels[upper - 1], labels[selected] = labels[selected], labels[upper - 1]
    return dict(zip(D1Arm, labels, strict=True))


def prepare_d1_blind_aliases(
    *, randomness: D1AliasRandomPort, custody: D1AliasCustodyPort
) -> D1PreparedBlindAliases:
    """Create one salted mapping, persist commitment first, and seal the reveal."""
    aliases = _shuffle_aliases(randomness)
    nonce = randomness.token_bytes(_SALT_BYTES)
    if type(nonce) is not bytes or len(nonce) != _SALT_BYTES:
        raise ValueError("D1 alias nonce must be exactly 32 bytes")
    reveal = _canonical(
        {
            "mapping": {alias: arm.value for arm, alias in aliases.items()},
            "nonce": nonce.hex(),
            "schema_version": "d1-alias-reveal-v1",
        }
    )
    commitment = _canonical(
        {
            "reveal_sha256": _sha256(reveal),
            "schema_version": "d1-alias-commitment-v1",
        }
    )
    commitment_receipt = custody.store_public_commitment(commitment)
    if commitment_receipt.sha256 != _sha256(commitment):
        raise ValueError("D1 alias commitment custody receipt does not bind stored bytes")
    private_receipt = custody.store_private_reveal(reveal)
    if private_receipt.sha256 != _sha256(reveal):
        raise ValueError("D1 private alias custody receipt does not bind stored bytes")
    alias_map = D1ArmAliasMap(aliases=aliases, sealed_commitment=_sha256(commitment))
    return D1PreparedBlindAliases(alias_map, commitment_receipt, private_receipt)


class D1InMemoryAliasCustodyFake:
    """Deterministic contract fake; never a production secrecy boundary."""

    def __init__(
        self,
        *,
        private_scope: D1AliasStorageScope = D1AliasStorageScope.PACKET_EXCLUDED,
    ) -> None:
        self._private_scope = private_scope
        self._public = b""
        self._private = b""
        self._order: list[D1AliasArtifactRole] = []

    def _receipt(
        self, role: D1AliasArtifactRole, scope: D1AliasStorageScope, payload: bytes
    ) -> D1AliasArtifactReceipt:
        self._order.append(role)
        return D1AliasArtifactReceipt(
            role=role,
            scope=scope,
            path=f"d1/alias-custody/{role.value}.json",
            sha256=_sha256(payload),
            durable=True,
            sequence=len(self._order) - 1,
        )

    def store_public_commitment(self, payload: bytes) -> D1AliasArtifactReceipt:
        self._public = bytes(payload)
        return self._receipt(D1AliasArtifactRole.COMMITMENT, D1AliasStorageScope.PUBLIC, payload)

    def store_private_reveal(self, payload: bytes) -> D1AliasArtifactReceipt:
        self._private = bytes(payload)
        return self._receipt(D1AliasArtifactRole.PRIVATE_REVEAL, self._private_scope, payload)

    def read_private(self, receipt: D1AliasArtifactReceipt) -> bytes:
        if receipt.role is not D1AliasArtifactRole.PRIVATE_REVEAL:
            raise ValueError("D1 alias custody can only read a private reveal receipt")
        if receipt.sha256 != _sha256(self._private):
            raise ValueError("D1 alias private receipt differs from stored bytes")
        return self._private

    @property
    def public_bytes(self) -> bytes:
        return self._public

    @property
    def private_bytes(self) -> bytes:
        return self._private

    @property
    def write_order(self) -> tuple[D1AliasArtifactRole, ...]:
        return tuple(self._order)
