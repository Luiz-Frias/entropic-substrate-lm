"""Hermetic structured-JSON contract for the D1 RunPod adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from d1_launch_admission_fixtures import plan, profiles, start
from entropic_substrate.d1_operator_ports import (
    D1CommandResult,
    D1OperatorTransportError,
)
from entropic_substrate.d1_operator_values import (
    D1A100FormFactor,
    D1A100ResourceRequest,
    D1ImmutableImage,
    D1OperatorLaunchIntent,
)
from entropic_substrate.d1_runpod_adapter import RunpodctlD1Adapter
from entropic_substrate.evidence.resource_billing import BillingGap, ProviderBilling

pytestmark = pytest.mark.unit

_DIGEST = "sha256:" + "e" * 64
_STARTED = datetime(2026, 8, 24, 16, 0, tzinfo=UTC)


@dataclass
class _Runner:
    outputs: list[object]
    error_at: int | None = None
    calls: list[tuple[tuple[str, ...], Path | None, str | None]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return False

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        stdin: str | None,
    ) -> D1CommandResult:
        self.calls.append((argv, cwd, stdin))
        index = len(self.calls) - 1
        if self.error_at == index:
            raise RuntimeError("simulated runpodctl failure")
        return D1CommandResult(json.dumps(self.outputs[index]), "")


def _intent() -> D1OperatorLaunchIntent:
    run = plan(profiles()[0], 101)
    observed = start(run)
    return D1OperatorLaunchIntent(
        plan=run,
        registered_design_digest="f" * 64,
        image=D1ImmutableImage(
            repository="registry.example/entropic-substrate-lm",
            digest=_DIGEST,
            source_commit="c" * 40,
        ),
        resource=D1A100ResourceRequest(
            accelerator_form_factor=D1A100FormFactor.SXM,
            provider_gpu_type_id="NVIDIA A100-SXM4-80GB",
            accelerator_count=1,
            provider="runpod",
            provider_sku="a100-sxm-80gb",
            region="US-KS-2",
            market="secure-cloud",
            purchase_option=observed.purchase_option,
        ),
        quoted_rate=observed.quoted_rate,
        resource_activation=observed.resource_activation,
        execution_segment_index=0,
        resume_checkpoint_sha256=None,
        terminate_after_seconds=7200,
    )


def _create_payload() -> dict[str, object]:
    return {
        "accelerator_count": 1,
        "accelerator_form_factor": "sxm",
        "accelerator_model": "NVIDIA A100-SXM4-80GB",
        "market": "secure-cloud",
        "pod_created_at": "2026-08-24T15:58:00Z",
        "pod_id": "pod-123",
        "provider": "runpod",
        "provider_sku": "a100-sxm-80gb",
        "purchase_option": "on-demand",
        "region": "US-KS-2",
        "schema_version": "d1-runpod-create-v1",
    }


def test_authentication_and_create_use_json_without_credentials_in_argv() -> None:
    runner = _Runner(
        [
            {
                "account_id": "account-123",
                "authenticated": True,
                "schema_version": "d1-runpod-auth-v1",
            },
            _create_payload(),
        ]
    )
    adapter = RunpodctlD1Adapter(runner=runner)

    auth = adapter.verify_authenticated()
    placement = adapter.create(_intent())

    assert auth.account_id == "account-123"
    assert placement.pod_id == "pod-123"
    assert runner.calls[0] == (
        ("runpodctl", "config", "current", "--output", "json"),
        None,
        None,
    )
    argv, cwd, stdin = runner.calls[1]
    assert argv == ("runpodctl", "create", "pod", "--spec", "-", "--output", "json")
    assert cwd is None
    assert stdin is not None
    request = json.loads(stdin)
    assert request["image"] == f"registry.example/entropic-substrate-lm@{_DIGEST}"
    assert request["terminate_after_seconds"] == 7200
    assert request["gpu_type_id"] == "NVIDIA A100-SXM4-80GB"
    assert "secret" not in stdin.lower()
    assert all("token" not in argument.lower() for argument in argv)


def test_status_returns_measured_runtime_and_rejects_wrong_pod_identity() -> None:
    payload = {
        "compiler_enabled": False,
        "compiler_version": "disabled",
        "cuda_version": "12.6",
        "driver_version": "560.35.03",
        "pod_id": "pod-123",
        "pytorch_version": "2.13.0",
        "schema_version": "d1-runpod-runtime-v1",
        "trainer_started_at": "2026-08-24T16:00:00Z",
    }
    adapter = RunpodctlD1Adapter(runner=_Runner([payload]))

    runtime = adapter.runtime("pod-123")

    assert runtime.trainer_started_at == _STARTED
    with pytest.raises(D1OperatorTransportError, match="pod identity"):
        RunpodctlD1Adapter(runner=_Runner([{**payload, "pod_id": "wrong"}])).runtime("pod-123")


def test_terminate_and_provider_billing_are_typed_evidence() -> None:
    intent = _intent()
    start_provenance = start(intent.plan)
    ended = _STARTED + timedelta(hours=1)
    runner = _Runner(
        [
            {
                "pod_id": "pod-123",
                "pod_terminated_at": "2026-08-24T17:00:05Z",
                "schema_version": "d1-runpod-termination-v1",
            },
            {
                "billed_amount": {"amount": "1.99", "currency": "USD"},
                "billed_duration_seconds": "3605",
                "billing_record_id": "bill-123",
                "billing_unit": "hour",
                "kind": "provider_billing",
                "pod_id": "pod-123",
                "schema_version": "d1-runpod-billing-v1",
            },
        ]
    )
    adapter = RunpodctlD1Adapter(runner=runner)

    termination = adapter.terminate("pod-123", start=start_provenance, trainer_ended_at=ended)
    billing = adapter.billing("pod-123", start=start_provenance)

    assert termination.pod_terminated_at == ended + timedelta(seconds=5)
    assert type(billing) is ProviderBilling
    assert billing.billed_amount.amount == Decimal("1.99")
    assert [call[0] for call in runner.calls] == [
        ("runpodctl", "remove", "pod", "pod-123", "--output", "json"),
        ("runpodctl", "billing", "pod", "pod-123", "--output", "json"),
    ]


def test_unavailable_provider_billing_is_a_complete_typed_gap() -> None:
    payload = {
        "kind": "billing_gap",
        "pod_id": "pod-123",
        "reason": "provider billing export unavailable",
        "schema_version": "d1-runpod-billing-v1",
    }

    result = RunpodctlD1Adapter(runner=_Runner([payload])).billing(
        "pod-123", start=start(_intent().plan)
    )

    assert type(result) is BillingGap
    assert result.quality_evidence_admissible is True
    assert result.billed_amount.reason == "provider billing export unavailable"


def test_cancel_returns_provider_measured_termination_timestamp() -> None:
    payload = {
        "pod_id": "pod-123",
        "pod_terminated_at": "2026-08-24T17:00:05Z",
        "schema_version": "d1-runpod-termination-v1",
    }

    observed = RunpodctlD1Adapter(runner=_Runner([payload])).cancel("pod-123")

    assert observed == datetime(2026, 8, 24, 17, 0, 5, tzinfo=UTC)


def test_command_or_noncanonical_json_failure_never_falls_back() -> None:
    with pytest.raises(D1OperatorTransportError, match="simulated runpodctl failure"):
        RunpodctlD1Adapter(runner=_Runner([], error_at=0)).verify_authenticated()

    class _DuplicateRunner(_Runner):
        def run(self, *args: object, **kwargs: object) -> D1CommandResult:
            del args, kwargs
            return D1CommandResult('{"authenticated":true,"authenticated":false}', "")

    with pytest.raises(D1OperatorTransportError, match="duplicate"):
        RunpodctlD1Adapter(runner=_DuplicateRunner([])).verify_authenticated()
