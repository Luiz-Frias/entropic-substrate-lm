"""Production campaign controller composition over RunPod and the R2 mailbox."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

import d1_launch_admission_fixtures as fixtures
from entropic_substrate.d1_campaign_controller_factory import (
    D1CampaignControllerFactory,
    D1CampaignControllerFactoryError,
    D1CampaignControllerRuntimeConfig,
)
from entropic_substrate.d1_operator_ports import D1RunPodAuthReceipt

from .test_d1_campaign_cli_service import _case

pytestmark = pytest.mark.unit

_AT = datetime(2026, 8, 28, tzinfo=UTC)
_PRIVATE = "provider-private-detail"


@dataclass
class _Objects:
    values: dict[str, bytes] = field(default_factory=dict)

    def publish(self, key: str, payload: bytes, *, expected_sha256: str) -> object:
        del key, payload, expected_sha256
        raise AssertionError("factory construction must not publish")

    def read(self, key: str) -> None:
        del key
        return None

    def list(self, prefix: str) -> tuple[()]:
        del prefix
        return ()


@dataclass
class _Runtime:
    mailbox: _Objects = field(default_factory=_Objects)


@dataclass
class _RunPod:
    calls: list[str] = field(default_factory=list)

    def verify_authenticated(self) -> D1RunPodAuthReceipt:
        self.calls.append("authenticate")
        raise AssertionError("factory construction must not authenticate")

    def create(self, intent: object) -> object:
        del intent
        raise AssertionError("factory construction must not create a pod")

    def cancel(self, pod_id: str) -> datetime:
        del pod_id
        raise AssertionError("factory construction must not delete a pod")

    terminate = billing = cancel


@dataclass(frozen=True)
class _Clock:
    def now_utc(self) -> datetime:
        return _AT


def _locations(loaded) -> None:  # type: ignore[no-untyped-def]
    locations = loaded.request.locations
    for path in (
        locations.repo_root,
        locations.artifact_root,
        locations.finalized_cell_custody_root,
    ):
        path.mkdir(parents=True, exist_ok=True)


def test_runtime_config_reads_only_required_nonsecret_operator_identifiers() -> None:
    config = D1CampaignControllerRuntimeConfig.from_environment(
        {
            "D1_RUNPOD_REGISTRY_AUTH_ID": "registry-auth-123",
            "D1_RUNPOD_NETWORK_VOLUME_ID": "network-volume-123",
            "AWS_ACCESS_KEY_ID": "not-part-of-config",
        }
    )

    assert config.registry_auth_id == "registry-auth-123"
    assert config.network_volume_id == "network-volume-123"
    assert config.network_volume_mount_path == "/workspace"
    assert config.trainer_repo_root == "/app"
    assert "not-part-of-config" not in repr(config)


def test_environment_template_declares_required_runpod_resource_identifiers() -> None:
    template = Path(__file__).resolve().parents[1].joinpath(".env.example").read_text()
    assert "D1_RUNPOD_REGISTRY_AUTH_ID=" in template
    assert "D1_RUNPOD_NETWORK_VOLUME_ID=" in template


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"D1_RUNPOD_REGISTRY_AUTH_ID": "registry-auth-123"},
        {"D1_RUNPOD_NETWORK_VOLUME_ID": "network-volume-123"},
        {
            "D1_RUNPOD_REGISTRY_AUTH_ID": "",
            "D1_RUNPOD_NETWORK_VOLUME_ID": "network-volume-123",
        },
    ],
)
def test_runtime_config_refuses_missing_or_blank_operator_identifiers(
    environ: dict[str, str],
) -> None:
    with pytest.raises(D1CampaignControllerFactoryError) as raised:
        D1CampaignControllerRuntimeConfig.from_environment(environ)
    assert raised.value.code == "configuration"


@pytest.mark.parametrize(
    "changes",
    [
        {"trainer_mailbox_poll_seconds": float("nan")},
        {"command_timeout_seconds": float("inf")},
        {"trainer_mailbox_max_objects": 0},
        {"max_output_bytes": 0},
    ],
)
def test_runtime_config_refuses_nonfinite_or_nonpositive_bounds(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "registry_auth_id": "registry-auth-123",
        "network_volume_id": "network-volume-123",
        **changes,
    }
    with pytest.raises(ValueError):
        D1CampaignControllerRuntimeConfig(**values)  # type: ignore[arg-type]


def test_factory_composes_existing_adapters_without_provider_calls(tmp_path: Path) -> None:
    loaded, _controller, _manifest = _case(tmp_path, 0)
    _locations(loaded)
    runtime_calls: list[dict[str, object]] = []
    runpod_calls: list[dict[str, object]] = []
    runpod_adapter = _RunPod()

    def runtime_builder(**kwargs: object) -> _Runtime:
        runtime_calls.append(dict(kwargs))
        return _Runtime()

    def runpod_builder(**kwargs: object) -> _RunPod:
        runpod_calls.append(dict(kwargs))
        return runpod_adapter

    environ = {"D1_TEST_OPERATOR_CONTEXT": "factory-fixture"}
    config = D1CampaignControllerRuntimeConfig(
        registry_auth_id="registry-auth-123",
        network_volume_id="network-volume-123",
    )
    factory = D1CampaignControllerFactory(
        catalog=fixtures.profiles(),
        environ=environ,
        config=config,
        clock=_Clock(),
        runtime_builder=runtime_builder,
        runpod_builder=runpod_builder,
    )

    managed = factory.build(loaded)

    assert managed.process_manager == loaded.process_manager
    assert managed.attempt_ledger == loaded.checkpoint.attempt_ledger
    assert runtime_calls == [
        {
            "environ": environ,
            "max_objects": config.trainer_mailbox_max_objects,
            "repo_root": loaded.request.locations.repo_root,
        }
    ]
    assert len(runpod_calls) == 1
    runpod = runpod_calls[0]
    assert runpod["registry_auth_id"] == config.registry_auth_id
    assert runpod["network_volume_id"] == config.network_volume_id
    assert runpod["trainer_mailbox_wait_seconds"] == max(
        intent.terminate_after_seconds for intent in loaded.request.intents
    )
    assert callable(runpod["clock"])
    assert runpod["clock"]() == _AT  # type: ignore[operator]
    assert not any(callable(getattr(value, "create", None)) for value in runtime_calls)
    managed.close()
    managed.close()
    with pytest.raises(RuntimeError, match="closed"):
        managed.execute_fresh(loaded.request.intents[0])
    assert runpod_adapter.calls == []


def test_factory_sanitizes_r2_composition_failure_before_runpod_or_custody(
    tmp_path: Path,
) -> None:
    loaded, _controller, _manifest = _case(tmp_path, 0)
    _locations(loaded)
    runpod_calls: list[object] = []

    def fail_runtime(**_kwargs: object) -> object:
        raise RuntimeError(_PRIVATE)

    factory = D1CampaignControllerFactory(
        catalog=fixtures.profiles(),
        environ={"D1_TEST_OPERATOR_CONTEXT": "failure-fixture"},
        config=D1CampaignControllerRuntimeConfig("registry", "volume"),
        clock=_Clock(),
        runtime_builder=fail_runtime,
        runpod_builder=lambda **kwargs: runpod_calls.append(kwargs),
    )

    with pytest.raises(D1CampaignControllerFactoryError) as raised:
        factory.build(loaded)

    assert raised.value.code == "r2"
    assert _PRIVATE not in str(raised.value)
    assert runpod_calls == []
