"""Cloud-review port, retry-bound receipt, and post-receipt alias release."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from entropic_substrate.analysis.d1_blind_aliases import (
    D1AliasCustodyPort,
    D1PreparedBlindAliases,
)
from entropic_substrate.analysis.d1_review_packet import (
    D1BlindReviewPacket,
    D1FactorialReviewerPrompt,
)
from entropic_substrate.evidence.d1_filing_receipts import (
    parse_alias_commitment,
    parse_alias_reveal,
)
from entropic_substrate.experiments.d1_primitives import canonical_digest

__all__ = [
    "D1CloudReviewRequest",
    "D1CloudReviewResponse",
    "D1CloudReviewerFake",
    "D1CloudReviewerPort",
    "D1ExecutedBlindReview",
    "D1InMemoryReviewReceiptStoreFake",
    "D1ReleasedAliasReveal",
    "D1ReviewExecutionReceipt",
    "D1ReviewReceiptStoragePort",
    "D1ReviewReceiptStorageReceipt",
    "execute_d1_blind_review",
    "release_d1_alias_reveal",
]

_SCHEMA = "d1-review-execution-receipt-v2"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text(value: str, label: str) -> None:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be nonempty and trimmed")


def _digest(value: str, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True, slots=True)
class D1CloudReviewRequest:
    packet: D1BlindReviewPacket
    prompt: D1FactorialReviewerPrompt
    provider: str
    model_id: str
    round: int
    retry_lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.packet) is not D1BlindReviewPacket:
            raise TypeError("D1 cloud review requires a blind packet")
        if type(self.prompt) is not D1FactorialReviewerPrompt:
            raise TypeError("D1 cloud review requires a factorial reviewer prompt")
        _text(self.provider, "D1 reviewer provider")
        _text(self.model_id, "D1 reviewer model_id")
        if type(self.round) is not int or self.round < 1:
            raise ValueError("D1 reviewer round must be positive")
        if type(self.retry_lineage) is not tuple:
            raise TypeError("D1 retry lineage must be a tuple")
        if len(self.retry_lineage) != self.round - 1:
            raise ValueError("D1 retry lineage length must equal round minus one")
        if len(set(self.retry_lineage)) != len(self.retry_lineage):
            raise ValueError("D1 retry lineage must not repeat a receipt")
        for digest in self.retry_lineage:
            _digest(digest, "D1 retry receipt")
        if self.packet.packet_digest not in self.prompt.text:
            raise ValueError("D1 reviewer prompt does not bind the packet digest")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "packet_sha256": self.packet.packet_digest,
            "prompt_sha256": self.prompt.prompt_digest,
            "provider": self.provider,
            "retry_lineage": list(self.retry_lineage),
            "round": self.round,
            "schema_version": "d1-cloud-review-request-v1",
        }

    @property
    def request_digest(self) -> str:
        return canonical_digest(self.manifest)


@dataclass(frozen=True, slots=True)
class D1CloudReviewResponse:
    request_digest: str
    provider: str
    model_id: str
    response: bytes
    report: bytes

    def __post_init__(self) -> None:
        _digest(self.request_digest, "D1 review response request_digest")
        _text(self.provider, "D1 review response provider")
        _text(self.model_id, "D1 review response model_id")
        if type(self.response) is not bytes or not self.response:
            raise ValueError("D1 review raw response must be nonempty bytes")
        if type(self.report) is not bytes or not self.report:
            raise ValueError("D1 review report must be nonempty bytes")


class D1CloudReviewerPort(Protocol):
    def review(self, request: D1CloudReviewRequest) -> D1CloudReviewResponse: ...


class D1CloudReviewerFake:
    """No-network reviewer adapter for contract tests."""

    def __init__(self, *, response: bytes, report: bytes) -> None:
        if type(response) is not bytes or type(report) is not bytes:
            raise TypeError("D1 cloud reviewer fake requires byte payloads")
        self._response = response
        self._report = report

    def review(self, request: D1CloudReviewRequest) -> D1CloudReviewResponse:
        if type(request) is not D1CloudReviewRequest:
            raise TypeError("D1 cloud reviewer fake requires D1CloudReviewRequest")
        return D1CloudReviewResponse(
            request_digest=request.request_digest,
            provider=request.provider,
            model_id=request.model_id,
            response=self._response,
            report=self._report,
        )


@dataclass(frozen=True, slots=True)
class D1ReviewExecutionReceipt:
    canonical_json: str

    @property
    def manifest(self) -> dict[str, object]:
        value = json.loads(self.canonical_json)
        if type(value) is not dict:
            raise AssertionError("D1 review receipt lost its object root")
        return value

    @property
    def receipt_digest(self) -> str:
        return _sha(self.canonical_json.encode())


@dataclass(frozen=True, slots=True)
class D1ReviewReceiptStorageReceipt:
    path: str
    sha256: str
    durable: bool

    def __post_init__(self) -> None:
        parsed = PurePosixPath(self.path)
        if (
            type(self.path) is not str
            or not self.path
            or parsed.is_absolute()
            or ".." in parsed.parts
            or parsed.as_posix() != self.path
        ):
            raise ValueError("D1 review receipt path must be normalized and relative")
        _digest(self.sha256, "D1 review storage SHA-256")
        if type(self.durable) is not bool:
            raise TypeError("D1 review storage durable must be bool")


class D1ReviewReceiptStoragePort(Protocol):
    def store(self, payload: bytes) -> D1ReviewReceiptStorageReceipt: ...


class D1InMemoryReviewReceiptStoreFake:
    """Durable-behavior fake, not production persistence evidence."""

    def __init__(self) -> None:
        self.payload = b""

    def store(self, payload: bytes) -> D1ReviewReceiptStorageReceipt:
        if type(payload) is not bytes or not payload:
            raise ValueError("D1 review receipt store requires nonempty bytes")
        self.payload = bytes(payload)
        return D1ReviewReceiptStorageReceipt(
            "d1/review/execution-receipt.json", _sha(payload), True
        )


@dataclass(frozen=True, slots=True)
class D1ExecutedBlindReview:
    response: bytes
    report: bytes
    receipt: D1ReviewExecutionReceipt
    storage: D1ReviewReceiptStorageReceipt

    def __post_init__(self) -> None:
        if self.storage.sha256 != self.receipt.receipt_digest:
            raise ValueError("D1 stored review receipt differs from issued receipt")
        if not self.storage.durable:
            raise ValueError("D1 review execution requires durable receipt storage")


def execute_d1_blind_review(
    *,
    request: D1CloudReviewRequest,
    reviewer: D1CloudReviewerPort,
    receipt_store: D1ReviewReceiptStoragePort,
) -> D1ExecutedBlindReview:
    """Execute through the cloud port and durably bind both returned payloads."""
    if type(request) is not D1CloudReviewRequest:
        raise TypeError("D1 blind review execution requires a typed request")
    response = reviewer.review(request)
    if type(response) is not D1CloudReviewResponse:
        raise TypeError("D1 cloud reviewer returned an unsupported response")
    if (
        response.request_digest != request.request_digest
        or response.provider != request.provider
        or response.model_id != request.model_id
    ):
        raise ValueError("D1 cloud reviewer response identity differs from request")
    manifest = {
        **request.manifest,
        "request_digest": request.request_digest,
        "report_sha256": _sha(response.report),
        "response_sha256": _sha(response.response),
        "schema_version": _SCHEMA,
    }
    receipt = D1ReviewExecutionReceipt(_canonical(manifest))
    storage = receipt_store.store(receipt.canonical_json.encode())
    return D1ExecutedBlindReview(response.response, response.report, receipt, storage)


@dataclass(frozen=True, slots=True)
class D1ReleasedAliasReveal:
    payload: bytes
    review_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes or not self.payload:
            raise ValueError("D1 released alias reveal must be nonempty bytes")
        _digest(self.review_receipt_sha256, "D1 reveal review receipt")


def release_d1_alias_reveal(
    *,
    execution: D1ExecutedBlindReview,
    aliases: D1PreparedBlindAliases,
    custody: D1AliasCustodyPort,
    public_commitment: bytes,
) -> D1ReleasedAliasReveal:
    """Open private custody only after the review receipt is durably bound."""
    if (
        not execution.storage.durable
        or execution.storage.sha256 != execution.receipt.receipt_digest
    ):
        raise ValueError("D1 alias reveal requires a durable review receipt")
    if (
        _sha(public_commitment) != aliases.commitment.sha256
        or _sha(public_commitment) != aliases.aliases.sealed_commitment
    ):
        raise ValueError("D1 alias reveal public commitment differs from custody")
    commitment = parse_alias_commitment(public_commitment)
    payload = custody.read_private(aliases.private_reveal)
    if _sha(payload) != aliases.private_reveal.sha256 or commitment.reveal_sha256 != _sha(payload):
        raise ValueError("D1 alias reveal does not open the public commitment")
    reveal = parse_alias_reveal(payload)
    expected = {alias: arm.value for arm, alias in aliases.aliases.aliases.items()}
    if dict(reveal.mapping) != expected:
        raise ValueError("D1 alias reveal mapping differs from the analysis aliases")
    return D1ReleasedAliasReveal(payload, execution.receipt.receipt_digest)
