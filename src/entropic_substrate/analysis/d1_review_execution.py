"""Cloud-review port, retry-bound receipt, and post-receipt alias release."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import NoReturn, Protocol, SupportsIndex

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
from entropic_substrate.evidence.d1_review_adjudication import (
    parse_d1_review_execution_receipt,
)
from entropic_substrate.experiments.d1_primitives import canonical_digest
from entropic_substrate.factory_attestation import FactoryAttestation

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
    prior_executions: tuple[D1ExecutedBlindReview, ...] = ()

    def __post_init__(self) -> None:
        if type(self.packet) is not D1BlindReviewPacket:
            raise TypeError("D1 cloud review requires a blind packet")
        if type(self.prompt) is not D1FactorialReviewerPrompt:
            raise TypeError("D1 cloud review requires a factorial reviewer prompt")
        _text(self.provider, "D1 reviewer provider")
        _text(self.model_id, "D1 reviewer model_id")
        if type(self.round) is not int or self.round < 1:
            raise ValueError("D1 reviewer round must be positive")
        if type(self.prior_executions) is not tuple:
            raise TypeError("D1 prior review executions must be a tuple")
        if len(self.prior_executions) != self.round - 1:
            raise ValueError("D1 retry lineage length must equal round minus one")
        _validated_retry_lineage(self)
        if self.packet.packet_digest not in self.prompt.text:
            raise ValueError("D1 reviewer prompt does not bind the packet digest")

    @property
    def retry_lineage(self) -> tuple[str, ...]:
        return _validated_retry_lineage(self)

    @property
    def manifest(self) -> dict[str, object]:
        return _request_manifest(
            self,
            round_number=self.round,
            retry_lineage=self.retry_lineage,
        )

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


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class D1ExecutedBlindReview:
    request: D1CloudReviewRequest
    response: bytes
    report: bytes
    receipt: D1ReviewExecutionReceipt
    storage: D1ReviewReceiptStorageReceipt

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("D1ExecutedBlindReview is factory-attested by review execution")

    def require_factory_attested(self) -> None:
        if not _EXECUTION_ATTESTATION.registered(self):
            raise TypeError("D1 executed blind review is not factory-attested")
        _validate_storage_binding(self.receipt, self.storage)
        if not _EXECUTION_ATTESTATION.contains(self):
            raise TypeError("D1 executed blind review evidence changed after minting")

    def __copy__(self) -> NoReturn:
        raise TypeError("D1ExecutedBlindReview cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        raise TypeError("D1ExecutedBlindReview cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("D1ExecutedBlindReview cannot be copied")


def _execution_fingerprint(execution: D1ExecutedBlindReview) -> str:
    return canonical_digest(
        {
            "receipt_digest": execution.receipt.receipt_digest,
            "report_sha256": _sha(execution.report),
            "request_digest": execution.request.request_digest,
            "response_sha256": _sha(execution.response),
            "storage": {
                "durable": execution.storage.durable,
                "path": execution.storage.path,
                "sha256": execution.storage.sha256,
            },
        }
    )


_EXECUTION_ATTESTATION = FactoryAttestation[
    D1ExecutedBlindReview,
    str,
](_execution_fingerprint)


def _is_attested_execution(execution: D1ExecutedBlindReview) -> bool:
    return _EXECUTION_ATTESTATION.contains(execution)


def _validate_storage_binding(
    receipt: D1ReviewExecutionReceipt,
    storage: D1ReviewReceiptStorageReceipt,
) -> None:
    if storage.sha256 != receipt.receipt_digest:
        raise ValueError("D1 stored review receipt differs from issued receipt")
    if not storage.durable:
        raise ValueError("D1 review execution requires durable receipt storage")


def _mint_executed_review(
    request: D1CloudReviewRequest,
    response: bytes,
    report: bytes,
    receipt: D1ReviewExecutionReceipt,
    storage: D1ReviewReceiptStorageReceipt,
) -> D1ExecutedBlindReview:
    _validate_storage_binding(receipt, storage)
    execution = object.__new__(D1ExecutedBlindReview)
    object.__setattr__(execution, "request", request)
    object.__setattr__(execution, "response", response)
    object.__setattr__(execution, "report", report)
    object.__setattr__(execution, "receipt", receipt)
    object.__setattr__(execution, "storage", storage)
    _EXECUTION_ATTESTATION.register(execution)
    return execution


def _request_manifest(
    request: D1CloudReviewRequest,
    *,
    round_number: int,
    retry_lineage: tuple[str, ...],
) -> dict[str, object]:
    return {
        "model_id": request.model_id,
        "packet_sha256": request.packet.packet_digest,
        "prompt_sha256": request.prompt.prompt_digest,
        "provider": request.provider,
        "retry_lineage": list(retry_lineage),
        "round": round_number,
        "schema_version": "d1-cloud-review-request-v1",
    }


def _validated_retry_lineage(request: D1CloudReviewRequest) -> tuple[str, ...]:
    lineage: list[str] = []
    for ordinal, execution in enumerate(request.prior_executions, start=1):
        if type(execution) is not D1ExecutedBlindReview or not _is_attested_execution(execution):
            raise TypeError("D1 retry requires a factory-attested prior execution")
        parsed = parse_d1_review_execution_receipt(execution.receipt.canonical_json.encode())
        expected_lineage = tuple(lineage)
        if parsed.round != ordinal or parsed.retry_lineage != expected_lineage:
            raise ValueError("D1 retry predecessor lineage is not contiguous")
        expected_request = _request_manifest(
            request,
            round_number=ordinal,
            retry_lineage=expected_lineage,
        )
        if (
            execution.request.request_digest != parsed.request_digest
            or execution.request.round != ordinal
            or execution.request.retry_lineage != expected_lineage
            or execution.request.packet.packet_digest != parsed.packet_sha256
            or execution.request.prompt.prompt_digest != parsed.prompt_sha256
            or execution.request.provider != parsed.provider
            or execution.request.model_id != parsed.model_id
            or parsed.packet_sha256 != request.packet.packet_digest
            or parsed.prompt_sha256 != request.prompt.prompt_digest
            or parsed.provider != request.provider
            or parsed.model_id != request.model_id
            or parsed.request_digest != canonical_digest(expected_request)
        ):
            raise ValueError("D1 retry predecessor identity differs")
        try:
            _validate_storage_binding(execution.receipt, execution.storage)
        except ValueError as error:
            raise ValueError("D1 retry predecessor output commitment differs") from error
        if parsed.response_sha256 != _sha(execution.response) or parsed.report_sha256 != _sha(
            execution.report
        ):
            raise ValueError("D1 retry predecessor output commitment differs")
        lineage.append(execution.receipt.receipt_digest)
    return tuple(lineage)


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
    return _mint_executed_review(request, response.response, response.report, receipt, storage)


@dataclass(frozen=True, slots=True, weakref_slot=True, init=False)
class D1ReleasedAliasReveal:
    payload: bytes
    review_receipt_sha256: str
    review_packet_sha256: str
    alias_commitment_sha256: str

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("D1ReleasedAliasReveal is factory-attested")

    def require_factory_attested(self) -> None:
        if not _REVEAL_ATTESTATION.registered(self):
            raise TypeError("D1 released alias reveal is not factory-attested")
        _validate_released_alias_reveal(self)
        if not _REVEAL_ATTESTATION.contains(self):
            raise TypeError("D1 released alias reveal evidence changed after minting")

    def __copy__(self) -> NoReturn:
        raise TypeError("D1ReleasedAliasReveal cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        raise TypeError("D1ReleasedAliasReveal cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("D1ReleasedAliasReveal cannot be copied")


def _reveal_fingerprint(reveal: D1ReleasedAliasReveal) -> str:
    return canonical_digest(
        {
            "alias_commitment_sha256": reveal.alias_commitment_sha256,
            "payload_sha256": _sha(reveal.payload),
            "review_packet_sha256": reveal.review_packet_sha256,
            "review_receipt_sha256": reveal.review_receipt_sha256,
        }
    )


_REVEAL_ATTESTATION = FactoryAttestation[
    D1ReleasedAliasReveal,
    str,
](_reveal_fingerprint)


def _validate_released_alias_reveal(reveal: D1ReleasedAliasReveal) -> None:
    if type(reveal.payload) is not bytes or not reveal.payload:
        raise ValueError("D1 released alias reveal must be nonempty bytes")
    _digest(reveal.review_receipt_sha256, "D1 reveal review receipt")
    _digest(reveal.review_packet_sha256, "D1 reveal review packet")
    _digest(reveal.alias_commitment_sha256, "D1 reveal alias commitment")


def _mint_released_alias_reveal(
    *,
    payload: bytes,
    review_receipt_sha256: str,
    review_packet_sha256: str,
    alias_commitment_sha256: str,
) -> D1ReleasedAliasReveal:
    reveal = object.__new__(D1ReleasedAliasReveal)
    object.__setattr__(reveal, "payload", payload)
    object.__setattr__(reveal, "review_receipt_sha256", review_receipt_sha256)
    object.__setattr__(reveal, "review_packet_sha256", review_packet_sha256)
    object.__setattr__(reveal, "alias_commitment_sha256", alias_commitment_sha256)
    _validate_released_alias_reveal(reveal)
    _REVEAL_ATTESTATION.register(reveal)
    return reveal


def release_d1_alias_reveal(
    *,
    execution: D1ExecutedBlindReview,
    aliases: D1PreparedBlindAliases,
    custody: D1AliasCustodyPort,
    public_commitment: bytes,
) -> D1ReleasedAliasReveal:
    """Open private custody only after the review receipt is durably bound."""
    if type(execution) is not D1ExecutedBlindReview or not _is_attested_execution(execution):
        raise TypeError("D1 alias reveal requires a factory-attested review execution")
    try:
        _validate_storage_binding(execution.receipt, execution.storage)
    except ValueError as error:
        raise ValueError("D1 alias reveal requires a durable review receipt") from error
    parsed_receipt = parse_d1_review_execution_receipt(execution.receipt.canonical_json.encode())
    packet = execution.request.packet
    packet_commitment = packet.manifest.get("alias_commitment")
    if (
        parsed_receipt.packet_sha256 != packet.packet_digest
        or packet_commitment != aliases.aliases.sealed_commitment
    ):
        raise ValueError("D1 reviewed packet does not bind the alias commitment")
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
    return _mint_released_alias_reveal(
        payload=payload,
        review_receipt_sha256=execution.receipt.receipt_digest,
        review_packet_sha256=packet.packet_digest,
        alias_commitment_sha256=aliases.aliases.sealed_commitment,
    )
