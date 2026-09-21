"""Pure admission of captured runpodctl pod-billing bucket rows."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import NoReturn, cast

from entropic_substrate.d1_operator_values import require_d1_pod_id
from entropic_substrate.evidence.resource_billing import (
    ProviderBilledAmountObservation,
    ProviderBillingObservation,
    ProviderBillingPartial,
    ProviderBillingPending,
    canonical_decimal,
)
from entropic_substrate.evidence.resource_provenance import (
    EvidenceGap,
    RunStartProvenance,
)
from entropic_substrate.experiments.d1_primitives import EvidenceClass

__all__ = [
    "RunpodBillingSchemaError",
    "parse_runpodctl_billing_observation",
]

_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
_REQUIRED = frozenset({"amount", "podId", "time", "timeBilledMs"})
_ALLOWED = _REQUIRED | {"diskSpaceBilledGb"}


class RunpodBillingSchemaError(ValueError):
    """Provider billing stdout is malformed, mismatched, or out of domain."""


def _fail(detail: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise RunpodBillingSchemaError(detail)
    raise RunpodBillingSchemaError(detail) from error


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("billing JSON repeats a key")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail(f"billing JSON contains non-finite constant {value!r}")


def _provider_timestamp(value: object) -> str:
    if type(value) is not str or _TIMESTAMP.fullmatch(value) is None:
        _fail("billing row time must use YYYY-MM-DD HH:MM:SS")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError as error:
        _fail("billing row time is not a valid calendar timestamp", error)
    if parsed.strftime("%Y-%m-%d %H:%M:%S") != value:
        _fail("billing row time is noncanonical")
    return value


def _amount(value: object) -> Decimal:
    if type(value) is int:
        value = Decimal(value)
    if type(value) is not Decimal or not value.is_finite() or value < 0:
        _fail("billing row amount is invalid")
    return value


def _checked_at(value: datetime) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        _fail("billing checked_at must be an exact UTC datetime")
    return value


def parse_runpodctl_billing_observation(
    raw: str,
    *,
    pod_id: str,
    start: RunStartProvenance,
    checked_at: datetime,
) -> ProviderBillingObservation:
    """Translate one exact CLI response; settlement policy remains outside."""
    pod = require_d1_pod_id(pod_id)
    if type(start) is not RunStartProvenance:
        raise TypeError("RunPod billing parser requires RunStartProvenance")
    if type(raw) is not str:
        raise TypeError("RunPod billing parser requires text stdout")
    observed_at = _checked_at(checked_at)
    try:
        decoded: object = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_float=Decimal,
            parse_constant=_nonfinite,
        )
    except RunpodBillingSchemaError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail("billing response is not valid UTF-8 JSON", error)
    if type(decoded) is not list:
        _fail("billing response root must be an array")
    rows = cast(list[object], decoded)
    if not rows:
        return ProviderBillingPending(
            start.logical_run_id,
            start.execution_segment_index,
            start.execution_segment_id,
            observed_at,
            "provider billing query returned no settled rows",
        )
    amounts: list[Decimal] = []
    billed_ms = 0
    timestamps: list[str] = []
    for raw_row in rows:
        if type(raw_row) is not dict:
            _fail("billing row must be an object")
        row = cast(dict[str, object], raw_row)
        if not _REQUIRED <= set(row) or set(row) - _ALLOWED:
            _fail("billing row fields differ from the pinned runpodctl schema")
        if row["podId"] != pod:
            _fail("billing row pod identity differs")
        milliseconds = row["timeBilledMs"]
        if type(milliseconds) is not int or milliseconds < 0:
            _fail("billing row timeBilledMs is invalid")
        disk = row.get("diskSpaceBilledGb")
        if disk is not None and (type(disk) is not int or disk < 0):
            _fail("billing row diskSpaceBilledGb is invalid")
        amounts.append(_amount(row["amount"]))
        billed_ms += milliseconds
        timestamps.append(_provider_timestamp(row["time"]))
    if len(set(timestamps)) != len(timestamps):
        _fail("billing rows repeat a provider bucket timestamp")
    reason = "runpodctl billing rows omit this provider field"
    return ProviderBillingPartial(
        start.logical_run_id,
        start.execution_segment_index,
        start.execution_segment_id,
        canonical_decimal(Decimal(billed_ms) / Decimal(1000)),
        ProviderBilledAmountObservation(
            canonical_decimal(sum(amounts, Decimal(0))),
            EvidenceGap("provider_billed_currency", reason),
        ),
        EvidenceGap("provider_billing_unit", reason),
        EvidenceGap("provider_billing_record_id", reason),
        EvidenceGap("provider_billing_status", reason),
        max(timestamps),
        hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        observed_at,
        EvidenceClass.PROVIDER_BILLED,
    )
