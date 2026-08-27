"""Content-addressed production custody for D1 operator transitions."""

from __future__ import annotations

import hashlib
from pathlib import Path

from entropic_substrate.d1_billing_finalization_values import (
    D1BillingFinalization,
    d1_billing_finalization_to_mapping,
)
from entropic_substrate.d1_content_addressed_custody import D1ContentAddressedCustodyStore
from entropic_substrate.d1_finalized_cell_custody import D1FinalizedCellCustodyRecord
from entropic_substrate.d1_finalized_cell_custody_repository import (
    D1FinalizedCellCustodyRepository,
)
from entropic_substrate.d1_operator_ports import D1RunPodAuthReceipt
from entropic_substrate.d1_operator_values import D1OperatorLaunchIntent
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.evidence.resource_provenance import RunStartProvenance
from entropic_substrate.evidence.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.experiments.d1_launch_attempt_codec import (
    launch_attempt_ledger_to_mapping,
)
from entropic_substrate.experiments.d1_launch_attempts import D1LaunchAttemptLedger
from entropic_substrate.experiments.d1_primitives import (
    canonical_json,
    require_nonempty_string,
)

__all__ = ["D1OperatorCustodyAdapter"]

_SCHEMA = "d1-operator-custody-event-v1"


class D1OperatorCustodyAdapter:
    """Publish every controller fact before the next external transition."""

    def __init__(self, *, root: Path, finalized_cell_root: Path) -> None:
        self._store = D1ContentAddressedCustodyStore(root)
        self._finalized = D1FinalizedCellCustodyRepository(finalized_cell_root)

    def close(self) -> None:
        self._finalized.close()
        self._store.close()

    def _publish(self, kind: str, value: dict[str, object]) -> ArtifactIdentity:
        require_nonempty_string(kind, label="D1 operator custody kind")
        payload = canonical_json(
            {
                "kind": kind,
                "schema_version": _SCHEMA,
                "value": value,
            }
        ).encode("utf-8")
        return self._store.publish(
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def persist_precreate(
        self,
        intent: D1OperatorLaunchIntent,
        ledger: D1LaunchAttemptLedger,
        auth: D1RunPodAuthReceipt,
    ) -> ArtifactIdentity:
        if type(intent) is not D1OperatorLaunchIntent:
            raise TypeError("operator precreate custody requires launch intent")
        if type(ledger) is not D1LaunchAttemptLedger:
            raise TypeError("operator precreate custody requires attempt ledger")
        if type(auth) is not D1RunPodAuthReceipt:
            raise TypeError("operator precreate custody requires auth receipt")
        return self._publish(
            "precreate",
            {
                "account_id": auth.account_id,
                "intent": intent.manifest,
                "ledger": launch_attempt_ledger_to_mapping(ledger),
            },
        )

    def persist_attempt_ledger(self, ledger: D1LaunchAttemptLedger) -> ArtifactIdentity:
        if type(ledger) is not D1LaunchAttemptLedger:
            raise TypeError("operator custody requires attempt ledger")
        return self._publish("attempt_ledger", launch_attempt_ledger_to_mapping(ledger))

    def persist_run_start(self, start: RunStartProvenance) -> ArtifactIdentity:
        if type(start) is not RunStartProvenance:
            raise TypeError("operator custody requires run start")
        return self._publish("run_start", run_start_to_mapping(start))

    def persist_finalization(self, finalization: D1BillingFinalization) -> ArtifactIdentity:
        return self._publish(
            "billing_finalization",
            d1_billing_finalization_to_mapping(finalization),
        )

    def persist_finalized_cell(self, record: D1FinalizedCellCustodyRecord) -> ArtifactIdentity:
        if type(record) is not D1FinalizedCellCustodyRecord:
            raise TypeError("operator custody requires finalized-cell record")
        return self._finalized.save(record)

    def persist_problem(self, subject_id: str, detail: str) -> ArtifactIdentity:
        require_nonempty_string(subject_id, label="D1 problem subject")
        if type(detail) is not str or not detail:
            raise ValueError("D1 problem detail must be nonempty text")
        payload = detail.encode("utf-8")
        return self._publish(
            "problem",
            {
                "detail_byte_count": len(payload),
                "detail_sha256": hashlib.sha256(payload).hexdigest(),
                "subject_id": subject_id,
            },
        )

    def discover_finalized_cells(
        self,
        *,
        exclude: tuple[ArtifactIdentity, ...] = (),
    ) -> tuple[tuple[D1FinalizedCellCustodyRecord, ArtifactIdentity], ...]:
        return self._finalized.discover(exclude=exclude)
