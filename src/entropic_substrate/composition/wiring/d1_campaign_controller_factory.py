"""Production D1 campaign controller composition over RunPod and R2 objects."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from entropic_substrate.composition.wiring.d1_campaign_authority_loader import (
    D1LoadedCampaignAuthority,
)
from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.kernel.lifecycle import ActorIdentity, ActorRole
from entropic_substrate.ladder.capture.ledger.d1_process_manager import D1ProcessManager
from entropic_substrate.ladder.register.d1_config import D1LaunchProfile
from entropic_substrate.ladder.register.d1_launch_attempts import D1LaunchAttemptLedger
from entropic_substrate.ladder.run.campaign.d1_operator_controller import (
    D1OperatorClock,
    D1OperatorRunResult,
    D1SequentialOperatorController,
)
from entropic_substrate.ladder.run.campaign.d1_operator_custody_adapter import (
    D1OperatorCustodyAdapter,
)
from entropic_substrate.ladder.run.campaign.d1_operator_values import D1OperatorLaunchIntent
from entropic_substrate.ladder.run.construction.d1_r2_runtime_composition import (
    build_d1_r2_runtime_composition,
)
from entropic_substrate.ladder.run.provider.d1_runpod_adapter import RunpodctlD1Adapter
from entropic_substrate.ladder.run.transport.d1_trainer_evidence_adapter import (
    D1TrainerEvidenceAdapter,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_worker import (
    D1TrainerMailboxWaitPolicy,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1ObjectMailboxTrainerTransport,
)

__all__ = [
    "D1CampaignControllerFactory",
    "D1CampaignControllerFactoryError",
    "D1CampaignControllerRuntimeConfig",
    "D1ManagedCampaignController",
]

_REGISTRY_AUTH_ENV = "D1_RUNPOD_REGISTRY_AUTH_ID"
_NETWORK_VOLUME_ENV = "D1_RUNPOD_NETWORK_VOLUME_ID"

type D1R2RuntimeBuilder = Callable[..., object]
type D1RunPodBuilder = Callable[..., object]


class D1CampaignControllerFactoryError(RuntimeError):
    """A bounded configuration or production composition refusal."""

    def __init__(self, *, code: str) -> None:
        self.code = code
        super().__init__(f"D1 campaign controller factory: {code}")


def _error(code: str) -> D1CampaignControllerFactoryError:
    return D1CampaignControllerFactoryError(code=code)


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be trimmed nonempty text")
    return value


@dataclass(frozen=True, slots=True)
class D1CampaignControllerRuntimeConfig:
    registry_auth_id: str
    network_volume_id: str
    network_volume_mount_path: str = "/workspace"
    trainer_repo_root: str = "/app"
    trainer_mailbox_poll_seconds: float = 0.25
    trainer_mailbox_max_objects: int = 128
    r2_access_key_secret_name: str = "r2access"
    r2_private_key_secret_name: str = "r2key"
    command_timeout_seconds: float = 60.0
    max_output_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        for label, value in (
            ("registry_auth_id", self.registry_auth_id),
            ("network_volume_id", self.network_volume_id),
            ("network_volume_mount_path", self.network_volume_mount_path),
            ("trainer_repo_root", self.trainer_repo_root),
            ("r2_access_key_secret_name", self.r2_access_key_secret_name),
            ("r2_private_key_secret_name", self.r2_private_key_secret_name),
        ):
            _text(value, label=label)
        if (
            type(self.trainer_mailbox_poll_seconds) not in (int, float)
            or not math.isfinite(float(self.trainer_mailbox_poll_seconds))
            or self.trainer_mailbox_poll_seconds <= 0
            or type(self.command_timeout_seconds) not in (int, float)
            or not math.isfinite(float(self.command_timeout_seconds))
            or self.command_timeout_seconds <= 0
        ):
            raise ValueError("D1 campaign controller time bounds must be positive")
        if type(self.trainer_mailbox_max_objects) is not int or (
            self.trainer_mailbox_max_objects < 1
        ):
            raise ValueError("D1 campaign mailbox object bound must be positive")
        if type(self.max_output_bytes) is not int or self.max_output_bytes < 1:
            raise ValueError("D1 campaign command output bound must be positive")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str],
    ) -> D1CampaignControllerRuntimeConfig:
        if not isinstance(cast(object, environ), Mapping) or any(
            type(key) is not str or type(value) is not str for key, value in environ.items()
        ):
            raise _error("configuration")
        registry, volume = environ.get(_REGISTRY_AUTH_ENV), environ.get(_NETWORK_VOLUME_ENV)
        try:
            return cls(
                _text(registry, label=_REGISTRY_AUTH_ENV),
                _text(volume, label=_NETWORK_VOLUME_ENV),
            )
        except (TypeError, ValueError) as cause:
            raise _error("configuration") from cause


class D1ManagedCampaignController:
    """Expose the queue port while owning controller custody descriptors."""

    def __init__(
        self,
        *,
        controller: D1SequentialOperatorController,
        custody: D1OperatorCustodyAdapter,
    ) -> None:
        if type(controller) is not D1SequentialOperatorController:
            raise TypeError("managed campaign requires the sequential controller")
        if type(custody) is not D1OperatorCustodyAdapter:
            raise TypeError("managed campaign requires operator custody")
        self._controller, self._custody, self._closed = controller, custody, False

    @property
    def process_manager(self) -> D1ProcessManager:
        return self._controller.process_manager

    @property
    def attempt_ledger(self) -> D1LaunchAttemptLedger:
        return self._controller.attempt_ledger

    def execute_fresh(self, intent: D1OperatorLaunchIntent) -> D1OperatorRunResult:
        if self._closed:
            raise RuntimeError("D1 managed campaign controller is closed")
        return self._controller.execute_fresh(intent)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._custody.close()


class D1CampaignControllerFactory:
    """Compose one sequential controller from already-admitted campaign authority."""

    def __init__(
        self,
        *,
        catalog: tuple[D1LaunchProfile, ...],
        environ: Mapping[str, str],
        config: D1CampaignControllerRuntimeConfig,
        clock: D1OperatorClock,
        runtime_builder: D1R2RuntimeBuilder = build_d1_r2_runtime_composition,
        runpod_builder: D1RunPodBuilder = RunpodctlD1Adapter,
    ) -> None:
        if (
            type(catalog) is not tuple
            or not catalog
            or any(type(item) is not D1LaunchProfile for item in catalog)
        ):
            raise ValueError("D1 campaign controller factory requires a profile catalog")
        if not isinstance(cast(object, environ), Mapping) or any(
            type(key) is not str or type(value) is not str for key, value in environ.items()
        ):
            raise TypeError("D1 campaign controller environment must be a string mapping")
        if type(config) is not D1CampaignControllerRuntimeConfig:
            raise TypeError("D1 campaign controller factory requires runtime config")
        if not callable(getattr(clock, "now_utc", None)):
            raise TypeError("D1 campaign controller factory requires an operator clock")
        if not callable(runtime_builder) or not callable(runpod_builder):
            raise TypeError("D1 campaign controller factory requires adapter builders")
        self._catalog = catalog
        self._environ = MappingProxyType(dict(environ))
        self._config, self._clock = config, clock
        self._runtime_builder, self._runpod_builder = runtime_builder, runpod_builder

    @staticmethod
    def _directory(path: Path) -> Path:
        target = Path(os.path.abspath(path))
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as cause:
            raise _error("locations") from cause
        if target.is_symlink() or not target.is_dir():
            raise _error("locations")
        return target

    def _runtime(self, loaded: D1LoadedCampaignAuthority) -> object:
        try:
            runtime = self._runtime_builder(
                repo_root=loaded.request.locations.repo_root,
                environ=self._environ,
                max_objects=self._config.trainer_mailbox_max_objects,
            )
        except Exception as cause:
            raise _error("r2") from cause
        mailbox = getattr(runtime, "mailbox", None)
        if not all(callable(getattr(mailbox, name, None)) for name in ("publish", "read", "list")):
            raise _error("r2")
        return runtime

    def _runpod(self, wait_seconds: int) -> object:
        config = self._config
        try:
            runpod = self._runpod_builder(
                clock=self._clock.now_utc,
                registry_auth_id=config.registry_auth_id,
                network_volume_id=config.network_volume_id,
                network_volume_mount_path=config.network_volume_mount_path,
                trainer_mailbox_wait_seconds=float(wait_seconds),
                trainer_mailbox_poll_seconds=config.trainer_mailbox_poll_seconds,
                trainer_mailbox_max_objects=config.trainer_mailbox_max_objects,
                trainer_repo_root=config.trainer_repo_root,
                r2_access_key_secret_name=config.r2_access_key_secret_name,
                r2_private_key_secret_name=config.r2_private_key_secret_name,
                command_timeout_seconds=config.command_timeout_seconds,
                max_output_bytes=config.max_output_bytes,
            )
        except Exception as cause:
            raise _error("runpod") from cause
        required = ("verify_authenticated", "create", "cancel", "terminate", "billing")
        if not all(callable(getattr(runpod, name, None)) for name in required):
            raise _error("runpod")
        return runpod

    def build(self, loaded: D1LoadedCampaignAuthority) -> D1ManagedCampaignController:
        if type(loaded) is not D1LoadedCampaignAuthority:
            raise TypeError("D1 campaign controller factory requires loaded authority")
        wait_seconds = max(intent.terminate_after_seconds for intent in loaded.request.intents)
        runtime = self._runtime(loaded)
        runpod = self._runpod(wait_seconds)
        locations = loaded.request.locations
        artifact_root = self._directory(locations.artifact_root)
        finalized_root = self._directory(locations.finalized_cell_custody_root)
        try:
            policy = D1TrainerMailboxWaitPolicy(
                float(wait_seconds),
                self._config.trainer_mailbox_poll_seconds,
            )
            transport = D1ObjectMailboxTrainerTransport(
                objects=runtime.mailbox,  # type: ignore[attr-defined]
                materialization_root=artifact_root,
                wait_policy=policy,
            )
            trainer = D1TrainerEvidenceAdapter(
                transport=transport,
                catalog=self._catalog,
            )
            custody = D1OperatorCustodyAdapter(
                root=artifact_root,
                finalized_cell_root=finalized_root,
            )
        except Exception as cause:
            raise _error("adapters") from cause
        request_digest = canonical_digest(loaded.request.manifest)
        try:
            controller = D1SequentialOperatorController(
                design=loaded.design,
                process_manager=loaded.process_manager,
                attempt_ledger=loaded.checkpoint.attempt_ledger,
                runpod=runpod,  # type: ignore[arg-type]
                trainer=trainer,
                custody=custody,
                operator=ActorIdentity(
                    "d1-campaign-operator",
                    f"campaign-{request_digest[:24]}",
                    ActorRole.OPERATOR,
                ),
                automation=ActorIdentity(
                    "d1-campaign-automation",
                    f"campaign-{request_digest[:24]}",
                    ActorRole.AUTOMATION,
                ),
                clock=self._clock,
            )
            return D1ManagedCampaignController(controller=controller, custody=custody)
        except Exception as cause:
            custody.close()
            raise _error("controller") from cause
