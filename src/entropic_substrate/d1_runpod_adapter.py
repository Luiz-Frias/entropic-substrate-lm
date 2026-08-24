"""Structured-JSON runpodctl adapter for the D1 accelerator boundary."""

from __future__ import annotations

import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import NoReturn, cast

from entropic_substrate.d1_operator_ports import (
    D1CommandRunner,
    D1OperatorTransportError,
    D1RunPodAuthReceipt,
)
from entropic_substrate.d1_operator_values import (
    D1A100FormFactor,
    D1OperatorLaunchIntent,
    D1PodPlacement,
    D1RuntimeObservation,
)
from entropic_substrate.evidence.resource_billing import (
    BillingGap,
    ExecutionSegmentTermination,
    ProviderBilling,
)
from entropic_substrate.evidence.resource_provenance import (
    BillingUnit,
    EvidenceGap,
    Money,
    PurchaseOption,
    RunStartProvenance,
)
from entropic_substrate.experiments.d1_primitives import EvidenceClass

_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")


def _error(operation: str, subject: str, detail: str, cause: Exception | None = None) -> NoReturn:
    error = D1OperatorTransportError(operation=operation, subject=subject, detail=detail)
    if cause is None:
        raise error
    raise error from cause


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate provider JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    raise ValueError(f"non-finite provider JSON constant {value!r}")


def _decode(raw: str, *, operation: str, subject: str) -> dict[str, object]:
    try:
        value: object = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (json.JSONDecodeError, ValueError) as error:
        _error(operation, subject, f"provider response is invalid JSON: {error}", error)
    if type(value) is not dict:
        _error(operation, subject, "provider response must be a JSON object")
    return cast(dict[str, object], value)


def _exact(value: dict[str, object], keys: set[str], *, operation: str, subject: str) -> None:
    if set(value) != keys:
        _error(operation, subject, "provider response keys differ from the adapter schema")


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _timestamp(value: object, *, label: str) -> datetime:
    text = _text(value, label=label)
    if _TIMESTAMP.fullmatch(text) is None:
        raise ValueError(f"{label} must be a canonical UTC Z timestamp")
    return datetime.fromisoformat(text[:-1] + "+00:00")


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


class RunpodctlD1Adapter:
    """Use configured runpodctl auth; no credential value crosses this API."""

    def __init__(self, *, runner: D1CommandRunner) -> None:
        if not hasattr(runner, "run"):
            raise TypeError("RunPod adapter requires a command runner")
        self._runner = runner

    def _run(
        self,
        operation: str,
        subject: str,
        argv: tuple[str, ...],
        *,
        stdin: str | None = None,
    ) -> dict[str, object]:
        try:
            result = self._runner.run(argv, cwd=None, stdin=stdin)
        except Exception as error:
            _error(operation, subject, str(error) or repr(error), error)
        return _decode(result.stdout, operation=operation, subject=subject)

    def verify_authenticated(self) -> D1RunPodAuthReceipt:
        operation = "authenticate"
        payload = self._run(
            operation,
            "configured-session",
            ("runpodctl", "config", "current", "--output", "json"),
        )
        _exact(
            payload,
            {"account_id", "authenticated", "schema_version"},
            operation=operation,
            subject="configured-session",
        )
        if payload["schema_version"] != "d1-runpod-auth-v1":
            _error(operation, "configured-session", "unsupported authentication schema")
        if payload["authenticated"] is not True:
            _error(operation, "configured-session", "runpodctl is not authenticated")
        try:
            return D1RunPodAuthReceipt(_text(payload["account_id"], label="account_id"))
        except (TypeError, ValueError) as error:
            _error(operation, "configured-session", str(error), error)

    def create(self, intent: D1OperatorLaunchIntent) -> D1PodPlacement:
        if type(intent) is not D1OperatorLaunchIntent:
            raise TypeError("RunPod create requires D1OperatorLaunchIntent")
        if intent.resource.provider != "runpod":
            raise ValueError("RunPod adapter requires provider='runpod'")
        subject = intent.plan.logical_run_id
        request = {
            "gpu_count": intent.resource.accelerator_count,
            "gpu_type_id": intent.resource.provider_gpu_type_id,
            "image": intent.image.reference,
            "logical_run_id": subject,
            "market": intent.resource.market,
            "provider_sku": intent.resource.provider_sku,
            "purchase_option": intent.resource.purchase_option.value,
            "region": intent.resource.region,
            "schema_version": "d1-runpod-create-request-v1",
            "segment_index": intent.execution_segment_index,
            "terminate_after_seconds": intent.terminate_after_seconds,
        }
        payload = self._run(
            "create_pod",
            subject,
            ("runpodctl", "create", "pod", "--spec", "-", "--output", "json"),
            stdin=json.dumps(request, allow_nan=False, separators=(",", ":"), sort_keys=True),
        )
        keys = {
            "accelerator_count",
            "accelerator_form_factor",
            "accelerator_model",
            "market",
            "pod_created_at",
            "pod_id",
            "provider",
            "provider_sku",
            "purchase_option",
            "region",
            "schema_version",
        }
        _exact(payload, keys, operation="create_pod", subject=subject)
        if payload["schema_version"] != "d1-runpod-create-v1":
            _error("create_pod", subject, "unsupported create response schema")
        try:
            return D1PodPlacement(
                pod_id=_text(payload["pod_id"], label="pod_id"),
                accelerator_form_factor=D1A100FormFactor(payload["accelerator_form_factor"]),
                accelerator_model=_text(payload["accelerator_model"], label="accelerator_model"),
                accelerator_count=_integer(payload["accelerator_count"], label="accelerator_count"),
                provider=_text(payload["provider"], label="provider"),
                provider_sku=_text(payload["provider_sku"], label="provider_sku"),
                region=_text(payload["region"], label="region"),
                market=_text(payload["market"], label="market"),
                purchase_option=PurchaseOption(payload["purchase_option"]),
                pod_created_at=_timestamp(payload["pod_created_at"], label="pod_created_at"),
            )
        except (TypeError, ValueError) as error:
            _error("create_pod", subject, f"invalid placement: {error}", error)

    def runtime(self, pod_id: str) -> D1RuntimeObservation:
        pod = _text(pod_id, label="pod_id")
        operation = "inspect_runtime"
        payload = self._run(
            operation,
            pod,
            ("runpodctl", "get", "pod", pod, "--output", "json"),
        )
        keys = {
            "compiler_enabled",
            "compiler_version",
            "cuda_version",
            "driver_version",
            "pod_id",
            "pytorch_version",
            "schema_version",
            "trainer_started_at",
        }
        _exact(payload, keys, operation=operation, subject=pod)
        if payload["schema_version"] != "d1-runpod-runtime-v1":
            _error(operation, pod, "unsupported runtime response schema")
        if payload["pod_id"] != pod:
            _error(operation, pod, "runtime pod identity differs from the request")
        try:
            return D1RuntimeObservation(
                pod_id=pod,
                trainer_started_at=_timestamp(
                    payload["trainer_started_at"], label="trainer_started_at"
                ),
                cuda_version=_text(payload["cuda_version"], label="cuda_version"),
                driver_version=_text(payload["driver_version"], label="driver_version"),
                pytorch_version=_text(payload["pytorch_version"], label="pytorch_version"),
                compiler_enabled=cast(bool, payload["compiler_enabled"]),
                compiler_version=_text(payload["compiler_version"], label="compiler_version"),
            )
        except (TypeError, ValueError) as error:
            _error(operation, pod, f"invalid runtime observation: {error}", error)

    def terminate(
        self,
        pod_id: str,
        *,
        start: RunStartProvenance,
        trainer_ended_at: datetime,
    ) -> ExecutionSegmentTermination:
        pod = _text(pod_id, label="pod_id")
        if type(start) is not RunStartProvenance:
            raise TypeError("RunPod termination requires RunStartProvenance")
        operation = "terminate_pod"
        payload = self._run(
            operation,
            pod,
            ("runpodctl", "remove", "pod", pod, "--output", "json"),
        )
        _exact(
            payload,
            {"pod_id", "pod_terminated_at", "schema_version"},
            operation=operation,
            subject=pod,
        )
        if payload["schema_version"] != "d1-runpod-termination-v1" or payload["pod_id"] != pod:
            _error(operation, pod, "termination response identity or schema differs")
        try:
            return ExecutionSegmentTermination(
                start.logical_run_id,
                start.execution_segment_index,
                start.execution_segment_id,
                trainer_ended_at,
                _timestamp(payload["pod_terminated_at"], label="pod_terminated_at"),
            )
        except (TypeError, ValueError) as error:
            _error(operation, pod, f"invalid termination evidence: {error}", error)

    def billing(self, pod_id: str, *, start: RunStartProvenance) -> ProviderBilling | BillingGap:
        pod = _text(pod_id, label="pod_id")
        if type(start) is not RunStartProvenance:
            raise TypeError("RunPod billing requires RunStartProvenance")
        operation = "provider_billing"
        payload = self._run(
            operation,
            pod,
            ("runpodctl", "billing", "pod", pod, "--output", "json"),
        )
        if payload.get("pod_id") != pod or payload.get("schema_version") != "d1-runpod-billing-v1":
            _error(operation, pod, "billing response identity or schema differs")
        if payload.get("kind") == "billing_gap":
            _exact(
                payload,
                {"kind", "pod_id", "reason", "schema_version"},
                operation=operation,
                subject=pod,
            )
            reason = _text(payload["reason"], label="billing gap reason")
            return BillingGap(
                start.logical_run_id,
                start.execution_segment_index,
                start.execution_segment_id,
                EvidenceGap("provider_billed_duration_seconds", reason),
                EvidenceGap("provider_billed_amount", reason),
                EvidenceGap("provider_billing_unit", reason),
                EvidenceGap("provider_billing_record_id", reason),
            )
        keys = {
            "billed_amount",
            "billed_duration_seconds",
            "billing_record_id",
            "billing_unit",
            "kind",
            "pod_id",
            "schema_version",
        }
        _exact(payload, keys, operation=operation, subject=pod)
        if payload["kind"] != "provider_billing":
            _error(operation, pod, "unsupported billing response kind")
        try:
            money = cast(dict[str, object], payload["billed_amount"])
            return ProviderBilling(
                start.logical_run_id,
                start.execution_segment_index,
                start.execution_segment_id,
                Decimal(_text(payload["billed_duration_seconds"], label="duration")),
                Money(
                    Decimal(_text(money["amount"], label="billing amount")),
                    _text(money["currency"], label="billing currency"),
                ),
                BillingUnit(payload["billing_unit"]),
                _text(payload["billing_record_id"], label="billing_record_id"),
                EvidenceClass.PROVIDER_BILLED,
            )
        except (InvalidOperation, KeyError, TypeError, ValueError) as error:
            _error(operation, pod, f"invalid provider billing: {error}", error)
