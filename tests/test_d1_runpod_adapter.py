"""Hermetic structured-JSON contract for the D1 RunPod adapter."""

from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from d1_launch_admission_fixtures import plan, profiles, start
from entropic_substrate.d1_command_runner import BoundedSubprocessD1CommandRunner
from entropic_substrate.d1_operator_ports import (
    D1CommandResult,
    D1OperatorTransportError,
    D1PodDeletionAcknowledgedError,
)
from entropic_substrate.d1_operator_values import (
    D1A100FormFactor,
    D1A100ResourceRequest,
    D1ImmutableImage,
    D1OperatorLaunchIntent,
)
from entropic_substrate.d1_runpod_adapter import RunpodctlD1Adapter
from entropic_substrate.evidence.resource_billing import (
    ProviderBillingPartial,
    ProviderBillingPending,
)
from entropic_substrate.evidence.resource_provenance import PurchaseOption

pytestmark = pytest.mark.unit

_DIGEST = "sha256:" + "e" * 64
_STARTED = datetime(2026, 8, 24, 16, 0, tzinfo=UTC)
_CONTROL_AT = datetime(2026, 8, 24, 15, 50, tzinfo=UTC)
_REGISTRY_AUTH = "registry-auth-123"
_NETWORK_VOLUME = "network-volume-123"
_VOLUME_MOUNT = "/workspace"
_VERSION = "runpodctl 2.11.0-0592468"


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
        trainer_mailbox_wait_seconds=300.0,
        trainer_mailbox_poll_seconds=0.25,
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


def test_default_process_boundary_requires_explicit_local_bounds() -> None:
    kwargs = {
        "clock": lambda: _CONTROL_AT,
        "registry_auth_id": _REGISTRY_AUTH,
        "network_volume_id": _NETWORK_VOLUME,
        "network_volume_mount_path": _VOLUME_MOUNT,
        "trainer_mailbox_wait_seconds": 300.0,
        "trainer_mailbox_poll_seconds": 0.25,
    }
    with pytest.raises(ValueError, match="process bounds"):
        RunpodctlD1Adapter(**kwargs)  # type: ignore[arg-type]
    adapter = RunpodctlD1Adapter(
        **kwargs,  # type: ignore[arg-type]
        command_timeout_seconds=60,
        max_output_bytes=1024 * 1024,
    )
    assert isinstance(adapter._runner, BoundedSubprocessD1CommandRunner)  # pyright: ignore[reportPrivateUsage]


def _create_payload() -> dict[str, object]:
    # GPU identity fields retain the measured GPU fixture; required allocation
    # keys follow the 2026-08-27 runpodctl 2.11 no-wait CPU schema capture.
    # The guarded schema pod was deleted immediately and had no billing rows.
    return {
        "id": "pod-123",
        "name": "d1-confirmatory-none-s101",
        "desiredStatus": "RUNNING",
        "imageName": f"registry.example/entropic-substrate-lm@{_DIGEST}",
        "gpuCount": 1,
        "createdAt": "2026-08-24 15:58:00.000 +0000 UTC",
        "machine": {
            "dataCenterId": "US-KS-2",
            "gpuId": "NVIDIA A100-SXM4-80GB",
            "location": "US",
            "secureCloud": True,
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
            _result(_create_payload(), stderr="note: allocation accepted\n"),
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
        "--docker-args",
        (
            "python -m entropic_substrate.d1_trainer_mailbox_command "
            "--volume-root /workspace --maximum-wait-seconds 300.0 "
            "--poll-interval-seconds 0.25"
        ),
        "--ssh",
        "--registry-auth-id",
        _REGISTRY_AUTH,
        "--network-volume-id",
        _NETWORK_VOLUME,
        "--volume-mount-path",
        _VOLUME_MOUNT,
        "--terminate-after",
        "2026-08-24T17:50:00Z",
        "--output",
        "json",
    )
    assert cwd is None
    assert stdin is None
    assert all("token" not in argument.lower() for argument in argv)


def test_proxy_only_create_does_not_require_tcp_ssh_or_wait() -> None:
    payload = _create_payload()
    runner = _Runner([_result(payload)])

    placement = _adapter(runner).create(_intent())

    argv = runner.calls[0][0]
    assert placement.pod_id == "pod-123"
    assert "--ssh" in argv
    assert "--ports" not in argv
    assert "--wait" not in argv
    assert "--wait-timeout" not in argv
    assert "--public-ip" not in argv


def test_community_market_maps_to_cli_cloud_without_direct_public_ip() -> None:
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
    assert "--public-ip" not in argv
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


def test_terminate_and_live_provider_buckets_are_typed_partial_evidence() -> None:
    intent = _intent()
    start_provenance = start(intent.plan)
    ended = _STARTED + timedelta(hours=1)
    billing_result = _result(
        [
            {
                "amount": 0.0017170833889394999,
                "diskSpaceBilledGb": 20,
                "podId": "pod-123",
                "time": "2026-08-24 17:00:00",
                "timeBilledMs": 24726,
            }
        ]
    )
    runner = _Runner([_result({"deleted": True, "id": "pod-123"}), billing_result])
    adapter = _adapter(runner, clock=lambda: ended + timedelta(seconds=5))

    termination = adapter.terminate("pod-123", start=start_provenance, trainer_ended_at=ended)
    billing = adapter.billing("pod-123", start=start_provenance)

    assert termination.pod_terminated_at == ended + timedelta(seconds=5)
    assert type(billing) is ProviderBillingPartial
    assert billing.billed_duration_seconds == Decimal("24.726")
    assert billing.billed_amount.amount == Decimal("0.0017170833889394999")
    assert billing.billed_amount.currency.field == "provider_billed_currency"
    assert billing.provider_timestamp == "2026-08-24 17:00:00"
    assert billing.observation_sha256 == hashlib.sha256(billing_result.stdout.encode()).hexdigest()
    assert [call[0] for call in runner.calls] == [
        ("runpodctl", "pod", "delete", "pod-123", "--output", "json"),
        (
            "runpodctl",
            "billing",
            "pods",
            "--pod-id",
            "pod-123",
            "--bucket-size",
            "hour",
            "--grouping",
            "podId",
            "--output",
            "json",
        ),
    ]


def test_empty_provider_query_is_pending_and_cannot_enter_terminal_billing() -> None:
    adapter = _adapter(_Runner([_result([])]))
    pending = adapter.observe_billing("pod-123", start=start(_intent().plan))
    assert type(pending) is ProviderBillingPending
    assert pending.is_terminal is False

    with pytest.raises(D1OperatorTransportError, match="pending"):
        _adapter(_Runner([_result([])])).billing("pod-123", start=start(_intent().plan))


def test_billing_clock_failure_is_normalized_at_the_control_adapter() -> None:
    adapter = _adapter(
        _Runner([_result([])]),
        clock=lambda: datetime(2026, 8, 26, 18, 0),
    )

    with pytest.raises(D1OperatorTransportError, match="control clock"):
        adapter.observe_billing("pod-123", start=start(_intent().plan))


def test_multiple_provider_buckets_sum_exactly_and_retain_latest_provider_time() -> None:
    rows = [
        {
            "amount": 0.1,
            "podId": "pod-123",
            "time": "2026-08-24 17:00:00",
            "timeBilledMs": 1000,
        },
        {
            "amount": 0.2,
            "diskSpaceBilledGb": 20,
            "podId": "pod-123",
            "time": "2026-08-24 18:00:00",
            "timeBilledMs": 2500,
        },
    ]

    observed = _adapter(_Runner([_result(rows)])).observe_billing(
        "pod-123", start=start(_intent().plan)
    )

    assert type(observed) is ProviderBillingPartial
    assert observed.billed_duration_seconds == Decimal("3.5")
    assert observed.billed_amount.amount == Decimal("0.3")
    assert observed.provider_timestamp == "2026-08-24 18:00:00"


@pytest.mark.parametrize(
    "payload",
    [
        {"not": "an array"},
        [{"amount": 0.1, "podId": "foreign", "time": "x", "timeBilledMs": 1}],
        [{"amount": 0.1, "podId": "pod-123", "time": "x"}],
        [
            {
                "amount": -0.1,
                "podId": "pod-123",
                "time": "2026-08-24 17:00:00",
                "timeBilledMs": 1,
            }
        ],
        [
            {
                "amount": 0.1,
                "podId": "pod-123",
                "time": "2026-99-99 99:99:99",
                "timeBilledMs": 1,
            }
        ],
    ],
)
def test_provider_billing_refuses_root_row_identity_and_domain_drift(payload: object) -> None:
    with pytest.raises(D1OperatorTransportError, match="billing"):
        _adapter(_Runner([_result(payload)])).observe_billing(
            "pod-123", start=start(_intent().plan)
        )


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


def test_cancel_sanitizes_raising_pre_delete_clock_without_mutation() -> None:
    runner = _Runner([])

    def clock() -> datetime:
        raise RuntimeError("secret clock failure")

    with pytest.raises(ValueError, match="pre-delete clock") as caught:
        _adapter(runner, clock=clock).cancel("pod-123")

    assert "secret clock failure" not in "".join(traceback.format_exception(caught.value))
    assert runner.calls == []


def test_cancel_reports_typed_acknowledgement_when_post_delete_clock_fails() -> None:
    runner = _Runner([_result({"deleted": True, "id": "pod-123"})])
    calls = 0

    def clock() -> datetime:
        nonlocal calls
        calls += 1
        return _CONTROL_AT if calls == 1 else datetime(2026, 8, 24, 17, 0, 5)

    with pytest.raises(D1PodDeletionAcknowledgedError) as caught:
        _adapter(runner, clock=clock).cancel("pod-123")

    assert caught.value.pod_id == "pod-123"
    assert caught.value.deletion_acknowledged is True
    assert "do not retry deletion" in str(caught.value)
    assert len(runner.calls) == 1


def test_cancel_reports_typed_acknowledgement_when_post_delete_clock_raises() -> None:
    runner = _Runner([_result({"deleted": True, "id": "pod-123"})])
    calls = 0

    def clock() -> datetime:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("secret clock failure")
        return _CONTROL_AT

    with pytest.raises(D1PodDeletionAcknowledgedError) as caught:
        _adapter(runner, clock=clock).cancel("pod-123")

    rendered = "".join(traceback.format_exception(caught.value))
    assert "secret clock failure" not in rendered
    assert caught.value.pod_id == "pod-123"
    assert len(runner.calls) == 1


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
    with pytest.raises(D1OperatorTransportError, match="termination response identity") as caught:
        _adapter(_Runner([_result(payload)])).cancel("pod-123")
    assert not isinstance(caught.value, D1PodDeletionAcknowledgedError)


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


def test_create_runner_exception_is_allocation_unknown_without_raw_retention() -> None:
    runner = _Runner([RuntimeError("secret runner failure")])

    with pytest.raises(D1OperatorTransportError, match="COST RISK") as caught:
        _adapter(runner).create(_intent())

    rendered = "".join(traceback.format_exception(caught.value))
    assert "allocation state is unknown" in rendered
    assert "reconcile by logical run name" in rendered
    assert "secret runner failure" not in rendered
    assert len(runner.calls) == 1


@pytest.mark.parametrize("pod_id", ["--bad", "pod\nforged", "pоd", "p" * 129])
def test_terminate_refuses_unsafe_pod_id_without_reflection_or_mutation(pod_id: str) -> None:
    runner = _Runner([])

    with pytest.raises(ValueError, match="lifecycle reconciliation") as caught:
        _adapter(runner).terminate(
            pod_id,
            start=start(_intent().plan),
            trainer_ended_at=_STARTED,
        )

    assert pod_id not in "".join(traceback.format_exception(caught.value))
    assert runner.calls == []


@pytest.mark.parametrize("pod_id", ["--bad", "pod\nforged", "pоd", "p" * 129])
def test_deletion_acknowledgement_refuses_unsafe_pod_identity(pod_id: str) -> None:
    with pytest.raises(ValueError, match="lifecycle reconciliation"):
        D1PodDeletionAcknowledgedError(pod_id=pod_id, detail="safe fixed detail")


def test_terminate_refuses_invalid_trainer_end_before_delete() -> None:
    runner = _Runner([])

    with pytest.raises(ValueError, match="trainer_ended_at"):
        _adapter(runner).terminate(
            "pod-123",
            start=start(_intent().plan),
            trainer_ended_at=datetime(2026, 8, 24, 17, 0, 5),
        )

    assert runner.calls == []


def test_terminate_reports_typed_acknowledgement_on_post_delete_ordering_failure() -> None:
    started = _STARTED + timedelta(hours=1)
    runner = _Runner([_result({"deleted": True, "id": "pod-123"})])

    with pytest.raises(D1PodDeletionAcknowledgedError) as caught:
        _adapter(runner, clock=lambda: started - timedelta(seconds=1)).terminate(
            "pod-123",
            start=start(_intent().plan),
            trainer_ended_at=started,
        )

    assert caught.value.pod_id == "pod-123"
    assert "termination evidence is invalid after deletion" in str(caught.value)
    assert len(runner.calls) == 1


def test_partial_create_failure_with_resource_id_deletes_once_before_raising() -> None:
    runner = _Runner(
        [
            D1CommandResult(
                "",
                "allocation response retained pod id\n"
                '{"error":"secret create detail","code":"create_timeout","id":"pod-partial"}',
                1,
            ),
            _result({"deleted": True, "id": "pod-partial"}),
        ]
    )

    with pytest.raises(D1OperatorTransportError) as caught:
        _adapter(runner).create(_intent())

    rendered = "".join(traceback.format_exception(caught.value))
    assert "code=create_timeout" in rendered
    assert "resource_id=pod-partial" in rendered
    assert "cleanup=deleted" in rendered
    assert "secret create detail" not in rendered
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
                '{"error":"secret","code":"create_timeout","id":"--unsafe"}',
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
    runner = _Runner([D1CommandResult(stdout, "allocation response\n", 0)])

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
                '{"error":"secret create","code":"create_timeout","id":"pod-partial"}',
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
                '{"error":"secret create","code":"create_timeout","id":"pod-partial"}',
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


def test_cleanup_failure_preserves_primary_and_makes_cost_leak_unmistakable() -> None:
    runner = _Runner(
        [
            D1CommandResult(
                "",
                '{"error":"secret create detail","code":"create_interrupted","id":"pod-leak"}',
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
    assert "code=create_interrupted" in rendered
    assert "resource_id=pod-leak" in rendered
    assert "COST LEAK" in rendered
    assert "cleanup_code=unauthorized" in rendered
    assert "cleanup_status=401" in rendered
    assert "secret create detail" not in rendered
    assert "secret cleanup detail" not in rendered
    assert len(runner.calls) == 2


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("desiredStatus", "EXITED", "desiredStatus"),
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
