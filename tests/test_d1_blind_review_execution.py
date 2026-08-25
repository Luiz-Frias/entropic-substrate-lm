"""Cloud review execution is receipt-bound before aliases may be revealed."""

from __future__ import annotations

import copy
import hashlib
import pickle
from dataclasses import replace
from typing import cast

import pytest

from entropic_substrate.analysis.d1_blind_aliases import (
    D1InMemoryAliasCustodyFake,
    prepare_d1_blind_aliases,
)
from entropic_substrate.analysis.d1_review_execution import (
    D1CloudReviewerFake,
    D1CloudReviewRequest,
    D1CloudReviewResponse,
    D1ExecutedBlindReview,
    D1InMemoryReviewReceiptStoreFake,
    D1ReleasedAliasReveal,
    D1ReviewReceiptStorageReceipt,
    execute_d1_blind_review,
    release_d1_alias_reveal,
)
from entropic_substrate.analysis.d1_review_packet import (
    D1BlindReviewPacket,
    D1FactorialReviewerPrompt,
)
from entropic_substrate.evidence.d1_review_adjudication import (
    D1_NO_MAJOR_PROMPT_DIRECTIVE,
)

pytestmark = pytest.mark.unit


class _Crypto:
    def __init__(self) -> None:
        self._draws = iter((2, 0, 1))

    def randbelow(self, upper: int) -> int:
        return next(self._draws)

    def token_bytes(self, length: int) -> bytes:
        return b"s" * length


def _request(
    *,
    round_number: int = 1,
    prior_executions: tuple[D1ExecutedBlindReview, ...] = (),
    alias_commitment: str | None = None,
) -> D1CloudReviewRequest:
    packet_manifest = {"schema_version": "d1-blind-review-packet-v1", "evidence": "opaque"}
    if alias_commitment is not None:
        packet_manifest["alias_commitment"] = alias_commitment
    packet = D1BlindReviewPacket.from_manifest(packet_manifest)
    prompt = D1FactorialReviewerPrompt(
        f"Packet SHA-256: {packet.packet_digest}\nRecompute only from packet.\n"
        f"{D1_NO_MAJOR_PROMPT_DIRECTIVE}\n"
    )
    return D1CloudReviewRequest(
        packet=packet,
        prompt=prompt,
        provider="review-provider",
        model_id="review-model-v1",
        round=round_number,
        prior_executions=prior_executions,
    )


def test_cloud_fake_execution_binds_raw_response_report_and_storage_receipt() -> None:
    request = _request()
    store = D1InMemoryReviewReceiptStoreFake()
    execution = execute_d1_blind_review(
        request=request,
        reviewer=D1CloudReviewerFake(response=b'{"id":"response-1"}', report=b"No MAJOR.\n"),
        receipt_store=store,
    )
    manifest = execution.receipt.manifest

    assert manifest["packet_sha256"] == request.packet.packet_digest
    assert manifest["prompt_sha256"] == request.prompt.prompt_digest
    assert manifest["provider"] == request.provider
    assert manifest["model_id"] == request.model_id
    assert manifest["round"] == 1
    assert manifest["retry_lineage"] == []
    assert manifest["response_sha256"] != manifest["report_sha256"]
    assert execution.storage.durable is True
    assert execution.storage.sha256 == execution.receipt.receipt_digest
    assert store.payload == execution.receipt.canonical_json.encode()


def test_retry_requires_complete_ordered_lineage_and_changes_receipt_identity() -> None:
    first = execute_d1_blind_review(
        request=_request(),
        reviewer=D1CloudReviewerFake(response=b"raw-1", report=b"report-1"),
        receipt_store=D1InMemoryReviewReceiptStoreFake(),
    )
    retry = _request(round_number=2, prior_executions=(first,))
    second = execute_d1_blind_review(
        request=retry,
        reviewer=D1CloudReviewerFake(response=b"raw-2", report=b"report-2"),
        receipt_store=D1InMemoryReviewReceiptStoreFake(),
    )

    assert second.receipt.manifest["retry_lineage"] == [first.receipt.receipt_digest]
    assert second.receipt.receipt_digest != first.receipt.receipt_digest
    with pytest.raises(ValueError, match="round minus one"):
        _request(round_number=2)


def test_retry_rejects_forged_skipped_and_wrong_predecessor_executions() -> None:
    first_a = execute_d1_blind_review(
        request=_request(),
        reviewer=D1CloudReviewerFake(response=b"raw-a1", report=b"report-a1"),
        receipt_store=D1InMemoryReviewReceiptStoreFake(),
    )
    first_b = execute_d1_blind_review(
        request=_request(),
        reviewer=D1CloudReviewerFake(response=b"raw-b1", report=b"report-b1"),
        receipt_store=D1InMemoryReviewReceiptStoreFake(),
    )
    second_b = execute_d1_blind_review(
        request=_request(round_number=2, prior_executions=(first_b,)),
        reviewer=D1CloudReviewerFake(response=b"raw-b2", report=b"report-b2"),
        receipt_store=D1InMemoryReviewReceiptStoreFake(),
    )

    with pytest.raises(TypeError, match="factory-attested"):
        _request(round_number=2, prior_executions=(copy.copy(first_a),))
    with pytest.raises(TypeError, match="factory-attested"):
        D1ExecutedBlindReview(
            first_a.response,
            first_a.report,
            first_a.receipt,
            first_a.storage,
        )
    with pytest.raises(TypeError, match="factory-attested"):
        _request(round_number=2, prior_executions=(copy.deepcopy(first_a),))
    with pytest.raises(TypeError, match="factory-attested"):
        _request(
            round_number=2,
            prior_executions=cast(tuple[D1ExecutedBlindReview, ...], ("d" * 64,)),
        )
    with pytest.raises(ValueError, match="round minus one"):
        _request(round_number=3, prior_executions=(first_a,))
    with pytest.raises(ValueError, match="predecessor"):
        _request(round_number=3, prior_executions=(first_a, second_b))


def test_retry_revalidates_prior_output_commitments() -> None:
    first = execute_d1_blind_review(
        request=_request(),
        reviewer=D1CloudReviewerFake(response=b"raw-1", report=b"report-1"),
        receipt_store=D1InMemoryReviewReceiptStoreFake(),
    )
    object.__setattr__(first, "response", b"substituted")

    with pytest.raises(TypeError, match="factory-attested"):
        _request(round_number=2, prior_executions=(first,))

    object.__setattr__(first, "response", b"raw-1")
    object.__setattr__(first, "report", b"substituted-report")
    with pytest.raises(TypeError, match="factory-attested"):
        _request(round_number=2, prior_executions=(first,))


def test_response_identity_must_equal_request_identity() -> None:
    request = _request()

    class _DriftedReviewer:
        def review(self, request: D1CloudReviewRequest) -> D1CloudReviewResponse:
            return D1CloudReviewResponse(
                request_digest=request.request_digest,
                provider="different-provider",
                model_id=request.model_id,
                response=b"raw",
                report=b"report",
            )

    with pytest.raises(ValueError, match="identity differs"):
        execute_d1_blind_review(
            request=request,
            reviewer=_DriftedReviewer(),
            receipt_store=D1InMemoryReviewReceiptStoreFake(),
        )


def test_alias_reveal_requires_durable_review_receipt_and_opens_commitment() -> None:
    custody = D1InMemoryAliasCustodyFake()
    aliases = prepare_d1_blind_aliases(randomness=_Crypto(), custody=custody)
    execution = execute_d1_blind_review(
        request=_request(alias_commitment=aliases.aliases.sealed_commitment),
        reviewer=D1CloudReviewerFake(response=b"raw", report=b"report"),
        receipt_store=D1InMemoryReviewReceiptStoreFake(),
    )

    released = release_d1_alias_reveal(
        execution=execution,
        aliases=aliases,
        custody=custody,
        public_commitment=custody.public_bytes,
    )

    assert released.payload == custody.private_bytes
    assert released.review_receipt_sha256 == execution.receipt.receipt_digest
    assert released.review_packet_sha256 == execution.request.packet.packet_digest
    assert released.alias_commitment_sha256 == aliases.aliases.sealed_commitment
    released.require_factory_attested()
    with pytest.raises(TypeError, match="not factory-attested"):
        object.__new__(D1ReleasedAliasReveal).require_factory_attested()
    with pytest.raises(TypeError, match="factory-attested"):
        D1ReleasedAliasReveal(
            released.payload,
            released.review_receipt_sha256,
            released.review_packet_sha256,
            released.alias_commitment_sha256,
        )
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(released)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(released)
    with pytest.raises(TypeError, match="cannot be copied"):
        pickle.dumps(released)
    with pytest.raises(TypeError, match="factory-attested"):
        replace(released, review_receipt_sha256="0" * 64)
    object.__setattr__(released, "payload", b"substituted reveal")
    with pytest.raises(TypeError, match="evidence changed"):
        released.require_factory_attested()
    with pytest.raises(ValueError, match="commitment"):
        release_d1_alias_reveal(
            execution=execution,
            aliases=aliases,
            custody=custody,
            public_commitment=b"{}",
        )
    with pytest.raises(TypeError, match="factory-attested"):
        replace(execution, storage=replace(execution.storage, durable=False))
    with pytest.raises(TypeError, match="factory-attested"):
        release_d1_alias_reveal(
            execution=copy.copy(execution),
            aliases=aliases,
            custody=custody,
            public_commitment=custody.public_bytes,
        )
    wrong_packet_execution = execute_d1_blind_review(
        request=_request(alias_commitment="0" * 64),
        reviewer=D1CloudReviewerFake(response=b"raw", report=b"report"),
        receipt_store=D1InMemoryReviewReceiptStoreFake(),
    )
    with pytest.raises(ValueError, match="reviewed packet"):
        release_d1_alias_reveal(
            execution=wrong_packet_execution,
            aliases=aliases,
            custody=custody,
            public_commitment=custody.public_bytes,
        )
    object.__setattr__(execution, "storage", replace(execution.storage, durable=False))
    with pytest.raises(TypeError, match="factory-attested"):
        release_d1_alias_reveal(
            execution=execution,
            aliases=aliases,
            custody=custody,
            public_commitment=custody.public_bytes,
        )


def test_alias_reveal_rejects_receipt_to_request_packet_substitution() -> None:
    custody = D1InMemoryAliasCustodyFake()
    aliases = prepare_d1_blind_aliases(randomness=_Crypto(), custody=custody)
    execution = execute_d1_blind_review(
        request=_request(alias_commitment=aliases.aliases.sealed_commitment),
        reviewer=D1CloudReviewerFake(response=b"raw", report=b"report"),
        receipt_store=D1InMemoryReviewReceiptStoreFake(),
    )
    substituted_packet = D1BlindReviewPacket.from_manifest(
        {
            "alias_commitment": aliases.aliases.sealed_commitment,
            "evidence": "different-packet-bytes",
            "schema_version": "d1-blind-review-packet-v1",
        }
    )
    substituted_prompt = D1FactorialReviewerPrompt(
        f"Packet SHA-256: {substituted_packet.packet_digest}\n{D1_NO_MAJOR_PROMPT_DIRECTIVE}\n"
    )
    substituted_request = D1CloudReviewRequest(
        packet=substituted_packet,
        prompt=substituted_prompt,
        provider=execution.request.provider,
        model_id=execution.request.model_id,
        round=1,
    )
    object.__setattr__(execution, "request", substituted_request)

    with pytest.raises(TypeError, match="factory-attested"):
        release_d1_alias_reveal(
            execution=execution,
            aliases=aliases,
            custody=custody,
            public_commitment=custody.public_bytes,
        )


def test_storage_receipt_refuses_false_hash_or_nondurable_type() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        D1ReviewReceiptStorageReceipt("review/receipt.json", "bad", True)
    with pytest.raises(TypeError, match="durable"):
        D1ReviewReceiptStorageReceipt("review/receipt.json", "a" * 64, 1)  # type: ignore[arg-type]


def test_execution_rejects_receipt_store_hash_drift_and_nondurability() -> None:
    class _Store:
        def __init__(self, *, matching: bool, durable: bool) -> None:
            self.matching = matching
            self.durable = durable

        def store(self, payload: bytes) -> D1ReviewReceiptStorageReceipt:
            digest = hashlib.sha256(payload).hexdigest() if self.matching else "a" * 64
            return D1ReviewReceiptStorageReceipt("review/receipt.json", digest, self.durable)

    for store, message in (
        (_Store(matching=False, durable=True), "differs from issued"),
        (_Store(matching=True, durable=False), "requires durable"),
    ):
        with pytest.raises(ValueError, match=message):
            execute_d1_blind_review(
                request=_request(),
                reviewer=D1CloudReviewerFake(response=b"raw", report=b"report"),
                receipt_store=store,
            )
