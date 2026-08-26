"""Hermetic structured-JSON contract for the D1 RunPod adapter."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field, replace
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
from entropic_substrate.evidence.resource_provenance import PurchaseOption

pytestmark = pytest.mark.unit

_DIGEST = "sha256:" + "e" * 64
_STARTED = datetime(2026, 8, 24, 16, 0, tzinfo=UTC)
_CONTROL_AT = datetime(2026, 8, 24, 15, 50, tzinfo=UTC)
_REGISTRY_AUTH = "registry-auth-123"
_NETWORK_VOLUME = "network-volume-123"
_VOLUME_MOUNT = "/workspace"
_WAIT_TIMEOUT = "12m"
_VERSION = "runpodctl 2.9.0-c094cac"


@dataclass
class _Runner:
    results: list[D1CommandResult | Exception]
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
        result = self.results[len(self.calls) - 1]
        if isinstance(result, Exception):
            raise result
        return result


def _result(value: object, *, stderr: str = "", returncode: int = 0) -> D1CommandResult:
    return D1CommandResult(json.dumps(value), stderr, returncode)


def _adapter(
    runner: _Runner,
    *,
    clock=lambda: _CONTROL_AT,
) -> RunpodctlD1Adapter:
    return RunpodctlD1Adapter(
        runner=runner,
        clock=clock,
        registry_auth_id=_REGISTRY_AUTH,
        network_volume_id=_NETWORK_VOLUME,
        network_volume_mount_path=_VOLUME_MOUNT,
        wait_timeout=_WAIT_TIMEOUT,
    )


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
            provider_sku="NVIDIA A100-SXM4-80GB",
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
    # Sanitized known-answer shape captured 2026-08-26 from exact local
    # runpodctl 2.9.0-c094cac (`pod create --wait`, secure RTX A4000).
    # The probe pod was deleted successfully before this fixture was filed.
    return {
        "id": "pod-123",
        "name": "d1-confirmatory-none-s101",
        "desiredStatus": "RUNNING",
        "runtimeStatus": "running",
        "imageName": f"registry.example/entropic-substrate-lm@{_DIGEST}",
        "gpuCount": 1,
        "createdAt": "2026-08-24 15:58:00.000 +0000 UTC",
        "machine": {
            "dataCenterId": "US-KS-2",
            "gpuId": "NVIDIA A100-SXM4-80GB",
            "location": "US",
            "secureCloud": True,
        },
        "ssh": {
            "id": "pod-123",
            "name": "d1-confirmatory-none-s101",
            "ssh_command": "ssh -i key root@203.0.113.10 -p 22022",
            "ip": "203.0.113.10",
            "port": 22022,
            "ssh_key": {
                "path": "/root/.ssh/id_ed25519",
                "exists": True,
                "in_account": True,
            },
        },
        "containerDiskInGb": 20,
        "volumeInGb": 0,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "costPerHr": 1.99,
    }


def test_authentication_and_create_use_json_without_credentials_in_argv() -> None:
    runner = _Runner(
        [
            D1CommandResult(_VERSION, "warning: local version note"),
            _result({"id": "account-123", "email": "user@example.test", "balance": 12.5}),
            _result(_create_payload(), stderr="note: waiting for ssh\n"),
        ]
    )
    adapter = _adapter(runner)

    auth = adapter.verify_authenticated()
    placement = adapter.create(_intent())

    assert auth.account_id == "account-123"
    assert placement.pod_id == "pod-123"
    assert placement.pod_created_at == datetime(2026, 8, 24, 15, 58, tzinfo=UTC)
    assert placement.provider_sku == "NVIDIA A100-SXM4-80GB"
    assert runner.calls[0] == (
        ("runpodctl", "version"),
        None,
        None,
    )
    assert runner.calls[1] == (("runpodctl", "user", "--output", "json"), None, None)
    argv, cwd, stdin = runner.calls[2]
    assert argv == (
        "runpodctl",
        "pod",
        "create",
        "--name",
        _intent().plan.logical_run_id,
        "--image",
        f"registry.example/entropic-substrate-lm@{_DIGEST}",
        "--gpu-id",
        "NVIDIA A100-SXM4-80GB",
        "--gpu-count",
        "1",
        "--cloud-type",
        "SECURE",
        "--data-center-ids",
        "US-KS-2",
        "--ssh",
        "--ports",
        "22/tcp",
        "--registry-auth-id",
        _REGISTRY_AUTH,
        "--network-volume-id",
        _NETWORK_VOLUME,
        "--volume-mount-path",
        _VOLUME_MOUNT,
        "--terminate-after",
        "2026-08-24T17:50:00Z",
        "--wait",
        "--wait-timeout",
        _WAIT_TIMEOUT,
        "--output",
        "json",
    )
    assert cwd is None
    assert stdin is None
    assert all("token" not in argument.lower() for argument in argv)


def test_community_market_maps_to_cli_cloud_and_public_ip() -> None:
    intent = _intent()
    intent = replace(intent, resource=replace(intent.resource, market="community-cloud"))
    payload = _create_payload()
    payload["machine"] = {
        "dataCenterId": "US-KS-2",
        "gpuId": "NVIDIA A100-SXM4-80GB",
        "secureCloud": False,
    }
    runner = _Runner([_result(payload)])

    placement = _adapter(runner).create(intent)

    argv = runner.calls[0][0]
    assert argv[argv.index("--cloud-type") + 1] == "COMMUNITY"
    assert "--public-ip" in argv
    assert placement.market == "community-cloud"


def test_additive_documented_provider_fields_do_not_break_placement() -> None:
    payload = _create_payload()
    payload.update({"lastStatusChange": "Rented by User", "runtimeStatusReason": ""})
    placement = _adapter(_Runner([_result(payload)])).create(_intent())
    assert placement.pod_id == "pod-123"


def test_create_refuses_spot_before_any_provider_command() -> None:
    intent = _intent()
    intent = replace(intent, resource=replace(intent.resource, purchase_option=PurchaseOption.SPOT))
    runner = _Runner([])
    with pytest.raises(ValueError, match="ON_DEMAND"):
        _adapter(runner).create(intent)
    assert runner.calls == []


def test_create_refuses_unmeasurable_provider_sku_before_provider_command() -> None:
    intent = _intent()
    intent = replace(intent, resource=replace(intent.resource, provider_sku="private-label"))
    runner = _Runner([])
    with pytest.raises(ValueError, match="gpuTypeId"):
        _adapter(runner).create(intent)
    assert runner.calls == []


def test_terminate_and_provider_billing_are_typed_evidence() -> None:
    intent = _intent()
    start_provenance = start(intent.plan)
    ended = _STARTED + timedelta(hours=1)
    runner = _Runner(
        [
            _result({"deleted": True, "id": "pod-123"}),
            _result(
                {
                    "billed_amount": {"amount": "1.99", "currency": "USD"},
                    "billed_duration_seconds": "3605",
                    "billing_record_id": "bill-123",
                    "billing_unit": "hour",
                    "kind": "provider_billing",
                    "pod_id": "pod-123",
                    "schema_version": "d1-runpod-billing-v1",
                }
            ),
        ]
    )
    adapter = _adapter(runner, clock=lambda: ended + timedelta(seconds=5))

    termination = adapter.terminate("pod-123", start=start_provenance, trainer_ended_at=ended)
    billing = adapter.billing("pod-123", start=start_provenance)

    assert termination.pod_terminated_at == ended + timedelta(seconds=5)
    assert type(billing) is ProviderBilling
    assert billing.billed_amount.amount == Decimal("1.99")
    assert [call[0] for call in runner.calls] == [
        ("runpodctl", "pod", "delete", "pod-123", "--output", "json"),
        ("runpodctl", "billing", "pod", "pod-123", "--output", "json"),
    ]


def test_unavailable_provider_billing_is_a_complete_typed_gap() -> None:
    payload = {
        "kind": "billing_gap",
        "pod_id": "pod-123",
        "reason": "provider billing export unavailable",
        "schema_version": "d1-runpod-billing-v1",
    }

    result = _adapter(_Runner([_result(payload)])).billing("pod-123", start=start(_intent().plan))

    assert type(result) is BillingGap
    assert result.quality_evidence_admissible is True
    assert result.billed_amount.reason == "provider billing export unavailable"


def test_billing_refuses_unsafe_pod_id_before_provider_command() -> None:
    runner = _Runner([])
    with pytest.raises(ValueError, match="lifecycle reconciliation"):
        _adapter(runner).billing("--output", start=start(_intent().plan))
    assert runner.calls == []


def test_cancel_returns_locally_measured_termination_timestamp() -> None:
    payload = {"deleted": True, "id": "pod-123"}
    measured = datetime(2026, 8, 24, 17, 0, 5, tzinfo=UTC)
    runner = _Runner([_result(payload)])
    calls = 0

    def clock() -> datetime:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert runner.calls, "post-delete clock ran before delete"
        return measured - timedelta(seconds=1) if calls == 1 else measured

    observed = _adapter(runner, clock=clock).cancel("pod-123")

    assert observed == measured
    assert calls == 2


def test_cancel_refuses_invalid_pre_delete_clock_without_mutation() -> None:
    runner = _Runner([_result({"deleted": True, "id": "pod-123"})])
    with pytest.raises(ValueError, match="UTC"):
        _adapter(runner, clock=lambda: datetime(2026, 8, 24, 17, 0, 5)).cancel("pod-123")
    assert runner.calls == []


@pytest.mark.parametrize(
    "pod_id",
    ["--help", "../pod-123", "pod id", "pod\nforged", "pоd", "p" * 129],
)
def test_cancel_refuses_unsafe_pod_id_before_provider_command(pod_id: str) -> None:
    runner = _Runner([_result({"deleted": True, "id": pod_id})])
    with pytest.raises(ValueError, match="lifecycle reconciliation"):
        _adapter(runner).cancel(pod_id)
    assert runner.calls == []


@pytest.mark.parametrize(
    "payload", [{"deleted": False, "id": "pod-123"}, {"deleted": True, "id": "other"}]
)
def test_cancel_refuses_delete_identity_drift(payload: dict[str, object]) -> None:
    with pytest.raises(D1OperatorTransportError, match="termination response identity"):
        _adapter(_Runner([_result(payload)])).cancel("pod-123")


def test_exact_cli_version_is_required_before_user_lookup() -> None:
    runner = _Runner([D1CommandResult("runpodctl 2.9.1-other", "")])
    with pytest.raises(D1OperatorTransportError, match="unsupported runpodctl version"):
        _adapter(runner).verify_authenticated()
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    "stderr",
    [
        '{"error":"secret detail","code":"unauthorized","status":401}',
        '{"error":"secret detail"}',
        "secret non-json failure",
    ],
)
def test_nonzero_failure_surfaces_only_coded_status_without_raw_retention(stderr: str) -> None:
    runner = _Runner([D1CommandResult("", stderr, 1)])
    with pytest.raises(D1OperatorTransportError) as caught:
        _adapter(runner).create(_intent())
    rendered = "".join(traceback.format_exception(caught.value))
    if '"code"' in stderr:
        assert "code=unauthorized status=401" in str(caught.value)
    else:
        assert "failure payload is malformed" in str(caught.value)
    assert "secret detail" not in rendered
    assert "secret non-json failure" not in rendered
    assert "COST RISK" in rendered
    assert len(runner.calls) == 1


def test_failure_code_cannot_inject_control_characters_into_error() -> None:
    runner = _Runner([D1CommandResult("", '{"error":"secret","code":"bad\\nforged"}', 1)])
    with pytest.raises(D1OperatorTransportError, match="failure payload is malformed") as caught:
        _adapter(runner).create(_intent())
    rendered = "".join(traceback.format_exception(caught.value))
    assert "forged" not in rendered
    assert "secret" not in rendered
    assert len(runner.calls) == 1


def test_partial_wait_failure_with_resource_id_deletes_once_before_raising() -> None:
    runner = _Runner(
        [
            D1CommandResult(
                "",
                "waiting for ssh on pod pod-partial (timeout 12m)\n"
                '{"error":"secret wait detail","code":"wait_timeout","id":"pod-partial"}',
                1,
            ),
            _result({"deleted": True, "id": "pod-partial"}),
        ]
    )

    with pytest.raises(D1OperatorTransportError) as caught:
        _adapter(runner).create(_intent())

    rendered = "".join(traceback.format_exception(caught.value))
    assert "code=wait_timeout" in rendered
    assert "resource_id=pod-partial" in rendered
    assert "cleanup=deleted" in rendered
    assert "secret wait detail" not in rendered
    assert [call[0][:3] for call in runner.calls] == [
        ("runpodctl", "pod", "create"),
        ("runpodctl", "pod", "delete"),
    ]


def test_create_error_without_resource_id_requires_operator_reconciliation() -> None:
    runner = _Runner(
        [D1CommandResult("", '{"error":"secret","code":"bad_request","status":400}', 1)]
    )
    with pytest.raises(D1OperatorTransportError, match="code=bad_request status=400") as caught:
        _adapter(runner).create(_intent())
    assert "COST RISK" in str(caught.value)
    assert "reconcile by logical run name" in str(caught.value)
    assert len(runner.calls) == 1


def test_create_error_with_unsafe_resource_id_reports_unreconciled_cost_risk() -> None:
    runner = _Runner(
        [
            D1CommandResult(
                "",
                '{"error":"secret","code":"wait_timeout","id":"--unsafe"}',
                1,
            )
        ]
    )

    with pytest.raises(D1OperatorTransportError, match="COST LEAK") as caught:
        _adapter(runner).create(_intent())

    rendered = "".join(traceback.format_exception(caught.value))
    assert "cleanup impossible" in rendered
    assert "--unsafe" not in rendered
    assert "secret" not in rendered
    assert len(runner.calls) == 1


@pytest.mark.parametrize("pod_id", [None, "pod id with spaces", "pod-123\nforged"])
def test_success_without_safe_resource_id_reports_unreconciled_cost_risk(
    pod_id: object,
) -> None:
    payload = _create_payload()
    payload["id"] = pod_id
    runner = _Runner([_result(payload)])

    with pytest.raises(D1OperatorTransportError, match="COST LEAK") as caught:
        _adapter(runner).create(_intent())

    assert "cleanup impossible" in str(caught.value)
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    "stdout",
    ["not-json", "[]", '{"id":"one","id":"two"}', '{"id":NaN}'],
)
def test_success_with_malformed_payload_reports_unreconciled_cost_risk(stdout: str) -> None:
    runner = _Runner([D1CommandResult(stdout, "waiting for ssh\n", 0)])

    with pytest.raises(D1OperatorTransportError, match="COST LEAK") as caught:
        _adapter(runner).create(_intent())

    assert "cleanup impossible" in str(caught.value)
    assert stdout not in "".join(traceback.format_exception(caught.value))
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    "cleanup",
    [
        D1CommandResult("not-json", "secret cleanup output", 0),
        _result({"deleted": False, "id": "pod-partial"}),
        _result({"deleted": True, "id": "foreign-pod"}),
    ],
)
def test_malformed_cleanup_acknowledgement_remains_a_cost_leak(
    cleanup: D1CommandResult,
) -> None:
    runner = _Runner(
        [
            D1CommandResult(
                "",
                '{"error":"secret create","code":"wait_timeout","id":"pod-partial"}',
                1,
            ),
            cleanup,
        ]
    )

    with pytest.raises(D1OperatorTransportError, match="COST LEAK") as caught:
        _adapter(runner).create(_intent())

    rendered = "".join(traceback.format_exception(caught.value))
    assert "cleanup_code=uncoded_failure" in rendered
    assert "secret create" not in rendered
    assert "secret cleanup output" not in rendered
    assert len(runner.calls) == 2


def test_cleanup_runner_exception_remains_a_sanitized_cost_leak() -> None:
    runner = _Runner(
        [
            D1CommandResult(
                "",
                '{"error":"secret create","code":"wait_timeout","id":"pod-partial"}',
                1,
            ),
            RuntimeError("secret cleanup exception"),
        ]
    )

    with pytest.raises(D1OperatorTransportError, match="COST LEAK") as caught:
        _adapter(runner).create(_intent())

    rendered = "".join(traceback.format_exception(caught.value))
    assert "cleanup_code=uncoded_failure" in rendered
    assert "secret create" not in rendered
    assert "secret cleanup exception" not in rendered
    assert len(runner.calls) == 2


def test_post_success_ssh_refusal_deletes_created_pod_before_raising() -> None:
    payload = _create_payload()
    payload["ssh"] = {"error": "pod not ready"}
    runner = _Runner([_result(payload), _result({"deleted": True, "id": "pod-123"})])

    with pytest.raises(D1OperatorTransportError) as caught:
        _adapter(runner).create(_intent())

    assert "pod SSH is not usable" in str(caught.value)
    assert "resource_id=pod-123" in str(caught.value)
    assert "cleanup=deleted" in str(caught.value)
    assert str(caught.value).count("D1 operator create_pod") == 1
    assert len(runner.calls) == 2


def test_cleanup_failure_preserves_primary_and_makes_cost_leak_unmistakable() -> None:
    runner = _Runner(
        [
            D1CommandResult(
                "",
                '{"error":"secret wait detail","code":"wait_interrupted","id":"pod-leak"}',
                1,
            ),
            D1CommandResult(
                "",
                '{"error":"secret cleanup detail","code":"unauthorized","status":401}',
                1,
            ),
        ]
    )

    with pytest.raises(D1OperatorTransportError) as caught:
        _adapter(runner).create(_intent())

    rendered = "".join(traceback.format_exception(caught.value))
    assert "code=wait_interrupted" in rendered
    assert "resource_id=pod-leak" in rendered
    assert "COST LEAK" in rendered
    assert "cleanup_code=unauthorized" in rendered
    assert "cleanup_status=401" in rendered
    assert "secret wait detail" not in rendered
    assert "secret cleanup detail" not in rendered
    assert len(runner.calls) == 2


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("desiredStatus", "EXITED", "desiredStatus"),
        ("runtimeStatus", "initializing", "runtimeStatus"),
        ("imageName", "mutable:latest", "image"),
        (
            "machine",
            {"dataCenterId": "US-KS-2", "gpuId": "NVIDIA A100 PCIe", "secureCloud": True},
            "GPU",
        ),
        ("gpuCount", 2, "count"),
        (
            "machine",
            {
                "dataCenterId": "EU-RO-1",
                "gpuId": "NVIDIA A100-SXM4-80GB",
                "secureCloud": True,
            },
            "region",
        ),
        ("name", "foreign-run", "name"),
        ("ssh", {"error": "pod not ready"}, "SSH"),
        (
            "ssh",
            {
                "id": "foreign-pod",
                "name": "d1-confirmatory-none-s101",
                "ssh_command": "ssh root@203.0.113.10 -p 22022",
                "ip": "203.0.113.10",
                "port": 22022,
                "ssh_key": {"exists": True, "in_account": True},
            },
            "SSH identity",
        ),
        ("missing:machine", None, "fields"),
    ],
)
def test_create_refuses_unusable_or_identity_drifted_pod(
    field: str, value: object, match: str
) -> None:
    payload = _create_payload()
    if field.startswith("missing:"):
        payload.pop(field.removeprefix("missing:"))
    else:
        payload[field] = value
    runner = _Runner([_result(payload), _result({"deleted": True, "id": "pod-123"})])
    with pytest.raises(D1OperatorTransportError, match=match) as caught:
        _adapter(runner).create(_intent())
    assert "cleanup=deleted" in str(caught.value)
    assert len(runner.calls) == 2
