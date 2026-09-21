"""Strict canonical mapping and JSON codec for one execution-segment start."""

from __future__ import annotations

import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import NoReturn, cast

from entropic_substrate.evidence.resource_provenance import (
    BillingRoundingPolicy,
    BillingUnit,
    Money,
    PowerIntegrationMethod,
    PowerSamplingRegistration,
    PurchaseOption,
    QuotedRate,
    RateScope,
    ResourceActivationMode,
    ResourceActivationRegistration,
    RoundingMode,
    RunStartProvenance,
    SegmentStartMode,
)
from entropic_substrate.experiments.d1_primitives import EvidenceClass

__all__ = [
    "ResourceProvenanceCodecGap",
    "parse_run_start_line",
    "quoted_rate_from_mapping",
    "quoted_rate_to_mapping",
    "resource_activation_from_mapping",
    "resource_activation_to_mapping",
    "run_start_from_mapping",
    "run_start_to_mapping",
    "serialize_run_start_line",
]

_EVENT = "resource.segment_start"
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_KEYS = {
    "accelerator_count",
    "accelerator_model",
    "compiler_enabled",
    "compiler_version",
    "cuda_version",
    "driver_version",
    "event",
    "execution_segment_id",
    "execution_segment_index",
    "image_digest",
    "logical_run_id",
    "market",
    "pod_created_at",
    "provider",
    "provider_sku",
    "purchase_option",
    "pytorch_version",
    "quoted_rate",
    "region",
    "resource_activation",
    "resume_checkpoint_sha256",
    "schema_version",
    "segment_start_mode",
    "trainer_started_at",
}
_ACTIVATION_KEYS = {
    "device_scope",
    "factory_registration_digest",
    "flop_model_identity",
    "meter_provider",
    "mode",
    "power_sampling",
    "schema_version",
}
_POWER_KEYS = {
    "cadence_seconds",
    "device_scope",
    "integration_method",
    "max_gap_seconds",
    "provider",
    "schema_version",
}
_RATE_KEYS = {
    "amount",
    "billing_unit",
    "evidence_class",
    "observed_at",
    "quote_id",
    "rate_scope",
    "rounding_policy",
    "source",
}


class ResourceProvenanceCodecGap(Exception):
    """A start-provenance wire value is malformed or noncanonical."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise ResourceProvenanceCodecGap(message)
    raise ResourceProvenanceCodecGap(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be a JSON object with string keys")
    return value


def _exact(mapping: dict[str, object], keys: set[str], *, label: str) -> None:
    if set(mapping) != keys:
        _fail(f"{label} keys must be exact")


def _timestamp(value: object, *, label: str) -> datetime:
    if type(value) is not str or _TIMESTAMP.fullmatch(value) is None:
        _fail(f"{label} must be a canonical UTC Z timestamp")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        _fail(f"{label} is invalid", error)


def _enum[T: StrEnum](enum_type: type[T], value: object, *, label: str) -> T:
    if type(value) is not str:
        _fail(f"{label} must be a canonical enum string")
    try:
        return enum_type(value)
    except ValueError as error:
        _fail(f"{label} is unsupported", error)


def _money(value: object) -> Money:
    mapping = _mapping(value, label="quoted_rate.amount")
    _exact(mapping, {"amount", "currency"}, label="quoted_rate.amount")
    if type(mapping["amount"]) is not str or type(mapping["currency"]) is not str:
        _fail("quoted_rate.amount values must be canonical strings")
    try:
        return Money(Decimal(mapping["amount"]), mapping["currency"])
    except (InvalidOperation, TypeError, ValueError) as error:
        _fail("quoted_rate.amount is invalid or noncanonical", error)


def _rounding(value: object) -> BillingRoundingPolicy:
    mapping = _mapping(value, label="quoted_rate.rounding_policy")
    _exact(
        mapping,
        {"increment", "minimum_units", "mode"},
        label="quoted_rate.rounding_policy",
    )
    try:
        return BillingRoundingPolicy(
            increment=cast(int, mapping["increment"]),
            minimum_units=cast(int, mapping["minimum_units"]),
            mode=_enum(RoundingMode, mapping["mode"], label="quoted_rate.rounding_policy.mode"),
        )
    except (TypeError, ValueError) as error:
        _fail("quoted_rate.rounding_policy is invalid", error)


def _rate(value: object) -> QuotedRate:
    mapping = _mapping(value, label="quoted_rate")
    _exact(mapping, _RATE_KEYS, label="quoted_rate")
    try:
        return QuotedRate(
            amount=_money(mapping["amount"]),
            billing_unit=_enum(BillingUnit, mapping["billing_unit"], label="billing_unit"),
            rounding_policy=_rounding(mapping["rounding_policy"]),
            rate_scope=_enum(RateScope, mapping["rate_scope"], label="rate_scope"),
            source=cast(str, mapping["source"]),
            quote_id=cast(str, mapping["quote_id"]),
            observed_at=_timestamp(mapping["observed_at"], label="quoted_rate.observed_at"),
            evidence_class=_enum(
                EvidenceClass,
                mapping["evidence_class"],
                label="quoted_rate.evidence_class",
            ),
        )
    except (TypeError, ValueError) as error:
        _fail(f"quoted_rate is invalid: {error}", error)


def _power_registration(value: object) -> PowerSamplingRegistration | None:
    if value is None:
        return None
    mapping = _mapping(value, label="resource_activation.power_sampling")
    _exact(mapping, _POWER_KEYS, label="resource_activation.power_sampling")
    try:
        return PowerSamplingRegistration(
            cadence_seconds=cast(float, mapping["cadence_seconds"]),
            max_gap_seconds=cast(float, mapping["max_gap_seconds"]),
            integration_method=_enum(
                PowerIntegrationMethod,
                mapping["integration_method"],
                label="resource_activation.power_sampling.integration_method",
            ),
            provider=cast(str, mapping["provider"]),
            device_scope=cast(str, mapping["device_scope"]),
            schema_version=cast(str, mapping["schema_version"]),
        )
    except (TypeError, ValueError) as error:
        _fail(f"resource_activation.power_sampling is invalid: {error}", error)


def _activation(value: object) -> ResourceActivationRegistration:
    mapping = _mapping(value, label="resource_activation")
    _exact(mapping, _ACTIVATION_KEYS, label="resource_activation")
    try:
        return ResourceActivationRegistration(
            factory_registration_digest=cast(str, mapping["factory_registration_digest"]),
            mode=_enum(ResourceActivationMode, mapping["mode"], label="resource_activation.mode"),
            meter_provider=cast(str, mapping["meter_provider"]),
            device_scope=cast(str, mapping["device_scope"]),
            flop_model_identity=cast(str | None, mapping["flop_model_identity"]),
            power_sampling=_power_registration(mapping["power_sampling"]),
            schema_version=cast(str, mapping["schema_version"]),
        )
    except (TypeError, ValueError) as error:
        _fail(f"resource_activation is invalid: {error}", error)


def _z(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def quoted_rate_to_mapping(rate: QuotedRate) -> dict[str, object]:
    if type(rate) is not QuotedRate:
        _fail("quoted-rate codec requires QuotedRate")
    mapping = dict(rate.manifest)
    mapping["observed_at"] = _z(rate.observed_at)
    return mapping


def quoted_rate_from_mapping(value: object) -> QuotedRate:
    return _rate(value)


def resource_activation_to_mapping(
    registration: ResourceActivationRegistration,
) -> dict[str, object]:
    if type(registration) is not ResourceActivationRegistration:
        _fail("resource-activation codec requires ResourceActivationRegistration")
    return dict(registration.manifest)


def resource_activation_from_mapping(value: object) -> ResourceActivationRegistration:
    return _activation(value)


def run_start_to_mapping(start: RunStartProvenance) -> dict[str, object]:
    if type(start) is not RunStartProvenance:
        _fail("run-start codec requires RunStartProvenance")
    mapping = dict(start.manifest)
    mapping["event"] = _EVENT
    mapping["pod_created_at"] = _z(start.pod_created_at)
    mapping["trainer_started_at"] = _z(start.trainer_started_at)
    mapping["quoted_rate"] = quoted_rate_to_mapping(start.quoted_rate)
    mapping["resource_activation"] = resource_activation_to_mapping(start.resource_activation)
    return mapping


def run_start_from_mapping(value: object) -> RunStartProvenance:
    mapping = _mapping(value, label="resource segment start")
    _exact(mapping, _KEYS, label="resource segment start")
    if mapping["event"] != _EVENT:
        _fail("resource segment start event is unsupported")
    try:
        return RunStartProvenance(
            schema_version=cast(str, mapping["schema_version"]),
            logical_run_id=cast(str, mapping["logical_run_id"]),
            execution_segment_index=cast(int, mapping["execution_segment_index"]),
            execution_segment_id=cast(str, mapping["execution_segment_id"]),
            segment_start_mode=_enum(
                SegmentStartMode, mapping["segment_start_mode"], label="segment_start_mode"
            ),
            resume_checkpoint_sha256=cast(str | None, mapping["resume_checkpoint_sha256"]),
            accelerator_model=cast(str, mapping["accelerator_model"]),
            accelerator_count=cast(int, mapping["accelerator_count"]),
            provider=cast(str, mapping["provider"]),
            provider_sku=cast(str, mapping["provider_sku"]),
            region=cast(str, mapping["region"]),
            market=cast(str, mapping["market"]),
            purchase_option=_enum(
                PurchaseOption, mapping["purchase_option"], label="purchase_option"
            ),
            image_digest=cast(str, mapping["image_digest"]),
            cuda_version=cast(str, mapping["cuda_version"]),
            driver_version=cast(str, mapping["driver_version"]),
            pytorch_version=cast(str, mapping["pytorch_version"]),
            compiler_enabled=cast(bool, mapping["compiler_enabled"]),
            compiler_version=cast(str, mapping["compiler_version"]),
            resource_activation=resource_activation_from_mapping(mapping["resource_activation"]),
            quoted_rate=quoted_rate_from_mapping(mapping["quoted_rate"]),
            pod_created_at=_timestamp(mapping["pod_created_at"], label="pod_created_at"),
            trainer_started_at=_timestamp(
                mapping["trainer_started_at"], label="trainer_started_at"
            ),
        )
    except (TypeError, ValueError) as error:
        _fail(f"resource segment start is invalid: {error}", error)


def serialize_run_start_line(start: RunStartProvenance) -> str:
    return json.dumps(
        run_start_to_mapping(start),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON constant {value!r} is forbidden")


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_run_start_line(line: str) -> RunStartProvenance:
    if type(line) is not str or not line.strip():
        _fail("resource segment start JSON line must be nonblank")
    try:
        value = json.loads(
            line,
            object_pairs_hook=_no_duplicates,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        _fail("resource segment start line is invalid JSON", error)
    return run_start_from_mapping(value)
