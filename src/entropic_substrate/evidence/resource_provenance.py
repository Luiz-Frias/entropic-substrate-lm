"""Immutable logical-run and execution-segment provenance for D1-D5 compute."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from entropic_substrate.experiments.d1_primitives import EvidenceClass, canonical_digest

__all__ = [
    "RESOURCE_TELEMETRY_SCHEMA_VERSION",
    "BillingRoundingPolicy",
    "BillingUnit",
    "EvidenceGap",
    "Money",
    "PowerIntegrationMethod",
    "PowerSamplingRegistration",
    "PurchaseOption",
    "QuotedRate",
    "RateScope",
    "ResourceActivationMode",
    "ResourceActivationRegistration",
    "RoundingMode",
    "RunStartProvenance",
    "SegmentStartMode",
    "execution_segment_id",
    "unavailable_resource_activation_registration",
    "validate_execution_segments",
]

RESOURCE_TELEMETRY_SCHEMA_VERSION = "resource-telemetry/v2"
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OCI_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class PurchaseOption(StrEnum):
    SPOT = "spot"
    ON_DEMAND = "on-demand"


class BillingUnit(StrEnum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"


class SegmentStartMode(StrEnum):
    FRESH = "fresh"
    RESUME = "resume"


class RateScope(StrEnum):
    PER_ACCELERATOR = "per-accelerator"
    PER_INSTANCE = "per-instance"


class RoundingMode(StrEnum):
    CEILING = "ceiling"
    FLOOR = "floor"
    NEAREST = "nearest"


class ResourceActivationMode(StrEnum):
    GAPS_ONLY = "gaps-only"
    PRODUCTION = "production"


class PowerIntegrationMethod(StrEnum):
    TRAPEZOIDAL_MONOTONIC_NS = "trapezoidal-monotonic-ns-v1"


def _nonempty(value: str, *, label: str) -> None:
    if type(value) is not str or not value.strip():
        msg = f"{label} must be a non-empty string"
        raise ValueError(msg)


def _utc_timestamp(value: datetime, *, label: str) -> None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        msg = f"{label} must be a UTC timestamp"
        raise ValueError(msg)


def _canonical_decimal(value: Decimal) -> str:
    if type(value) is not Decimal:
        msg = "money amount must be a Decimal"
        raise TypeError(msg)
    if not value.is_finite() or value < 0:
        msg = "money amount must be finite and nonnegative"
        raise ValueError(msg)
    normalized = value.normalize()
    canonical = "0" if value.is_zero() else format(normalized, "f")
    if value.as_tuple() != Decimal(canonical).as_tuple():
        msg = f"money amount must use canonical decimal spelling {canonical!r}"
        raise ValueError(msg)
    return canonical


@dataclass(frozen=True, slots=True)
class PowerSamplingRegistration:
    cadence_seconds: float
    max_gap_seconds: float
    integration_method: PowerIntegrationMethod
    provider: str
    device_scope: str
    schema_version: str = "resource-power-sampling/v1"

    def __post_init__(self) -> None:
        for field in ("cadence_seconds", "max_gap_seconds"):
            value = getattr(self, field)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be a finite positive float")
        if self.max_gap_seconds < self.cadence_seconds:
            raise ValueError("max_gap_seconds must be at least cadence_seconds")
        if type(self.integration_method) is not PowerIntegrationMethod:
            raise TypeError("integration_method must be PowerIntegrationMethod")
        _nonempty(self.provider, label="power sampling provider")
        _nonempty(self.device_scope, label="power sampling device_scope")
        if self.schema_version != "resource-power-sampling/v1":
            raise ValueError("unsupported resource power-sampling schema")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "cadence_seconds": self.cadence_seconds,
            "device_scope": self.device_scope,
            "integration_method": self.integration_method.value,
            "max_gap_seconds": self.max_gap_seconds,
            "provider": self.provider,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class ResourceActivationRegistration:
    factory_registration_digest: str
    mode: ResourceActivationMode
    meter_provider: str
    device_scope: str
    flop_model_identity: str | None
    power_sampling: PowerSamplingRegistration | None
    schema_version: str = "resource-activation-registration/v1"

    def __post_init__(self) -> None:
        if (
            type(self.factory_registration_digest) is not str
            or _SHA256.fullmatch(self.factory_registration_digest) is None
        ):
            raise ValueError("factory_registration_digest must be a lowercase SHA-256")
        if type(self.mode) is not ResourceActivationMode:
            raise TypeError("resource activation mode must be ResourceActivationMode")
        _nonempty(self.meter_provider, label="resource meter_provider")
        _nonempty(self.device_scope, label="resource device_scope")
        if self.flop_model_identity is not None and (
            type(self.flop_model_identity) is not str
            or _SHA256.fullmatch(self.flop_model_identity) is None
        ):
            raise ValueError("flop_model_identity must be a lowercase SHA-256 or None")
        if (
            self.power_sampling is not None
            and type(self.power_sampling) is not PowerSamplingRegistration
        ):
            raise TypeError("power_sampling must be PowerSamplingRegistration or None")
        if self.schema_version != "resource-activation-registration/v1":
            raise ValueError("unsupported resource activation registration schema")
        unavailable = self.meter_provider == "unavailable" or self.device_scope == "unavailable"
        if self.mode is ResourceActivationMode.GAPS_ONLY:
            if (
                self.meter_provider != "unavailable"
                or self.device_scope != "unavailable"
                or self.flop_model_identity is not None
                or self.power_sampling is not None
            ):
                raise ValueError(
                    "gaps-only registration must explicitly mark every meter unavailable"
                )
        elif unavailable:
            raise ValueError("production registration cannot use an unavailable provider or scope")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "device_scope": self.device_scope,
            "factory_registration_digest": self.factory_registration_digest,
            "flop_model_identity": self.flop_model_identity,
            "meter_provider": self.meter_provider,
            "mode": self.mode.value,
            "power_sampling": (
                None if self.power_sampling is None else self.power_sampling.manifest
            ),
            "schema_version": self.schema_version,
        }

    @property
    def identity_digest(self) -> str:
        return canonical_digest(self.manifest)


def unavailable_resource_activation_registration() -> ResourceActivationRegistration:
    digest = canonical_digest({"name": "unavailable-resource-activation", "schema_version": "v1"})
    return ResourceActivationRegistration(
        factory_registration_digest=digest,
        mode=ResourceActivationMode.GAPS_ONLY,
        meter_provider="unavailable",
        device_scope="unavailable",
        flop_model_identity=None,
        power_sampling=None,
    )


@dataclass(frozen=True, slots=True)
class EvidenceGap:
    """Compatibility value for absent runtime meters; never accepted by start provenance."""

    field: str
    reason: str

    def __post_init__(self) -> None:
        _nonempty(self.field, label="evidence gap field")
        _nonempty(self.reason, label="evidence gap reason")


@dataclass(frozen=True, slots=True)
class Money:
    """Exact, nonnegative money represented by a canonical :class:`Decimal`."""

    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        _canonical_decimal(self.amount)
        if type(self.currency) is not str or _CURRENCY.fullmatch(self.currency) is None:
            msg = f"money currency must be a three-letter uppercase code, got {self.currency!r}"
            raise ValueError(msg)

    @property
    def canonical_amount(self) -> str:
        return _canonical_decimal(self.amount)

    @property
    def manifest(self) -> dict[str, str]:
        return {"amount": self.canonical_amount, "currency": self.currency}


@dataclass(frozen=True, slots=True)
class BillingRoundingPolicy:
    """Provider billing mechanics, represented as data rather than a prose label."""

    increment: int
    mode: RoundingMode
    minimum_units: int

    def __post_init__(self) -> None:
        if type(self.increment) is not int or self.increment < 1:
            msg = "rounding increment must be a positive integer"
            raise ValueError(msg)
        if type(self.mode) is not RoundingMode:
            msg = "rounding mode must be a RoundingMode"
            raise TypeError(msg)
        if type(self.minimum_units) is not int or self.minimum_units < 1:
            msg = "rounding minimum_units must be a positive integer"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "increment": self.increment,
            "minimum_units": self.minimum_units,
            "mode": self.mode.value,
        }


@dataclass(frozen=True, slots=True)
class QuotedRate:
    """The measured observation of an immutable provider rate at launch time."""

    amount: Money
    billing_unit: BillingUnit
    rounding_policy: BillingRoundingPolicy
    rate_scope: RateScope
    source: str
    quote_id: str
    observed_at: datetime
    evidence_class: EvidenceClass

    def __post_init__(self) -> None:
        if type(self.amount) is not Money:
            msg = "quoted rate amount must be Money"
            raise TypeError(msg)
        if type(self.billing_unit) is not BillingUnit:
            msg = "quoted rate billing_unit must be a BillingUnit"
            raise TypeError(msg)
        if type(self.rounding_policy) is not BillingRoundingPolicy:
            msg = "quoted rate rounding_policy must be a BillingRoundingPolicy"
            raise TypeError(msg)
        if type(self.rate_scope) is not RateScope:
            msg = "quoted rate rate_scope must be a RateScope"
            raise TypeError(msg)
        _nonempty(self.source, label="quoted rate source")
        _nonempty(self.quote_id, label="quoted rate quote_id")
        _utc_timestamp(self.observed_at, label="quoted rate observed_at")
        if self.evidence_class is not EvidenceClass.MEASURED:
            msg = "quoted rate evidence_class must be EvidenceClass.MEASURED"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "amount": self.amount.manifest,
            "billing_unit": self.billing_unit.value,
            "evidence_class": self.evidence_class.value,
            "observed_at": self.observed_at.isoformat(),
            "quote_id": self.quote_id,
            "rate_scope": self.rate_scope.value,
            "rounding_policy": self.rounding_policy.manifest,
            "source": self.source,
        }


def execution_segment_id(logical_run_id: str, index: int) -> str:
    _nonempty(logical_run_id, label="logical_run_id")
    if type(index) is not int or index < 0:
        msg = "execution_segment_index must be a nonnegative integer"
        raise ValueError(msg)
    return f"{logical_run_id}:segment:{index}"


@dataclass(frozen=True, slots=True)
class RunStartProvenance:
    """Mandatory facts for one physical execution segment of one logical run."""

    schema_version: str
    logical_run_id: str
    execution_segment_index: int
    execution_segment_id: str
    segment_start_mode: SegmentStartMode
    resume_checkpoint_sha256: str | None
    accelerator_model: str
    accelerator_count: int
    provider: str
    provider_sku: str
    region: str
    market: str
    purchase_option: PurchaseOption
    image_digest: str
    cuda_version: str
    driver_version: str
    pytorch_version: str
    compiler_enabled: bool
    compiler_version: str
    quoted_rate: QuotedRate
    pod_created_at: datetime
    trainer_started_at: datetime

    def __post_init__(self) -> None:
        if self.schema_version != RESOURCE_TELEMETRY_SCHEMA_VERSION:
            msg = f"unsupported resource telemetry schema_version {self.schema_version!r}"
            raise ValueError(msg)
        for label, value in (
            ("logical_run_id", self.logical_run_id),
            ("accelerator_model", self.accelerator_model),
            ("provider", self.provider),
            ("provider_sku", self.provider_sku),
            ("region", self.region),
            ("market", self.market),
            ("cuda_version", self.cuda_version),
            ("driver_version", self.driver_version),
            ("pytorch_version", self.pytorch_version),
            ("compiler_version", self.compiler_version),
        ):
            _nonempty(value, label=label)
        if type(self.execution_segment_index) is not int or self.execution_segment_index < 0:
            msg = "execution_segment_index must be a nonnegative integer"
            raise ValueError(msg)
        expected_id = execution_segment_id(self.logical_run_id, self.execution_segment_index)
        if self.execution_segment_id != expected_id:
            msg = f"execution_segment_id must be exactly {expected_id!r}"
            raise ValueError(msg)
        if type(self.segment_start_mode) is not SegmentStartMode:
            msg = "segment_start_mode must be a SegmentStartMode"
            raise TypeError(msg)
        if self.execution_segment_index == 0:
            if self.segment_start_mode is not SegmentStartMode.FRESH:
                msg = "segment 0 must be a fresh segment"
                raise ValueError(msg)
            if self.resume_checkpoint_sha256 is not None:
                msg = "fresh segment must not carry resume_checkpoint_sha256"
                raise ValueError(msg)
        else:
            if self.segment_start_mode is not SegmentStartMode.RESUME:
                msg = "later segment must be a resume segment"
                raise ValueError(msg)
            if (
                type(self.resume_checkpoint_sha256) is not str
                or _SHA256.fullmatch(self.resume_checkpoint_sha256) is None
            ):
                msg = "resume segment requires a lowercase resume_checkpoint_sha256"
                raise ValueError(msg)
        if type(self.accelerator_count) is not int or self.accelerator_count < 1:
            msg = "accelerator_count must be a positive integer"
            raise ValueError(msg)
        if type(self.purchase_option) is not PurchaseOption:
            msg = "purchase_option must be a PurchaseOption"
            raise TypeError(msg)
        if type(self.image_digest) is not str or _OCI_DIGEST.fullmatch(self.image_digest) is None:
            msg = "image_digest must be a lowercase sha256:<64 hex> OCI digest"
            raise ValueError(msg)
        if type(self.compiler_enabled) is not bool:
            msg = "compiler_enabled must be a bool"
            raise TypeError(msg)
        if not self.compiler_enabled and self.compiler_version != "disabled":
            msg = "compiler_version must be 'disabled' when compiler_enabled is false"
            raise ValueError(msg)
        if type(self.quoted_rate) is not QuotedRate:
            msg = "quoted_rate must be QuotedRate"
            raise TypeError(msg)
        _utc_timestamp(self.pod_created_at, label="pod_created_at")
        _utc_timestamp(self.trainer_started_at, label="trainer_started_at")
        if self.trainer_started_at < self.pod_created_at:
            msg = "pod-created/trainer-start timestamp ordering is inverted"
            raise ValueError(msg)
        if self.quoted_rate.observed_at > self.trainer_started_at:
            msg = "quote observed_at must be no later than trainer_start"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "accelerator_count": self.accelerator_count,
            "accelerator_model": self.accelerator_model,
            "compiler_enabled": self.compiler_enabled,
            "compiler_version": self.compiler_version,
            "cuda_version": self.cuda_version,
            "driver_version": self.driver_version,
            "execution_segment_id": self.execution_segment_id,
            "execution_segment_index": self.execution_segment_index,
            "image_digest": self.image_digest,
            "logical_run_id": self.logical_run_id,
            "market": self.market,
            "pod_created_at": self.pod_created_at.isoformat(),
            "provider": self.provider,
            "provider_sku": self.provider_sku,
            "purchase_option": self.purchase_option.value,
            "pytorch_version": self.pytorch_version,
            "quoted_rate": self.quoted_rate.manifest,
            "region": self.region,
            "resume_checkpoint_sha256": self.resume_checkpoint_sha256,
            "schema_version": self.schema_version,
            "segment_start_mode": self.segment_start_mode.value,
            "trainer_started_at": self.trainer_started_at.isoformat(),
        }

    @property
    def identity_digest(self) -> str:
        return canonical_digest(self.manifest)


def validate_execution_segments(segments: tuple[RunStartProvenance, ...]) -> None:
    """Require the exact 0,1,2,... restart chain for one logical experiment run."""
    if type(segments) is not tuple or not segments:
        msg = "execution segments must be a non-empty tuple"
        raise ValueError(msg)
    logical_run_id: str | None = None
    for expected_index, segment in enumerate(segments):
        if type(segment) is not RunStartProvenance:
            msg = "execution segments must contain RunStartProvenance values"
            raise TypeError(msg)
        if logical_run_id is None:
            logical_run_id = segment.logical_run_id
        elif segment.logical_run_id != logical_run_id:
            msg = "execution segments must share one logical_run_id"
            raise ValueError(msg)
        if segment.execution_segment_index != expected_index:
            msg = "execution segment indices must be consecutive from zero"
            raise ValueError(msg)
        if segment.execution_segment_id != execution_segment_id(logical_run_id, expected_index):
            msg = "execution segment id is not derived from its logical run and index"
            raise ValueError(msg)
