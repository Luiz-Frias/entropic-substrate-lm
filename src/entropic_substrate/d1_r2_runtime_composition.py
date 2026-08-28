"""Credential-bound R2 mailbox and DVC checkpoint sync composition for D1."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from entropic_substrate.d1_checkpoint_dvc_recovery import (
    D1CheckpointDvcRecoveryService,
)
from entropic_substrate.d1_checkpoint_dvc_sync import (
    D1CheckpointDvcSyncAdapter,
    D1PublishedCheckpointDvcSync,
)
from entropic_substrate.d1_dvc_remote_config import (
    D1DvcRemoteConfigError,
    D1DvcS3Remote,
    load_d1_dvc_s3_remote,
)
from entropic_substrate.d1_operator_values import require_d1_pod_id
from entropic_substrate.d1_segment_engine import D1CheckpointSyncPort
from entropic_substrate.d1_trainer_s3_object_adapter import (
    Boto3D1TrainerMailboxObjectAdapter,
)
from entropic_substrate.evidence.d1_capture_ports import (
    D1CloudObjectVerifierPort,
    D1DvcCommandPort,
    D1DvcPullPort,
)
from entropic_substrate.evidence.d1_capture_subprocess import (
    SubprocessD1CloudObjectVerifier,
    SubprocessD1DvcAdapter,
)
from entropic_substrate.experiments.d1_primitives import require_sha256
from entropic_substrate.predictive_state.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
)

__all__ = [
    "D1BoundR2CheckpointSync",
    "D1R2RuntimeComposition",
    "D1R2RuntimeCompositionError",
    "build_d1_r2_runtime_composition",
]

type D1S3ClientFactory = Callable[..., object]


class D1R2DvcPort(D1DvcCommandPort, D1DvcPullPort, Protocol):
    """The one DVC adapter shared by cadence upload and crash recovery."""


class D1R2RuntimeCompositionError(RuntimeError):
    """A bounded topology, credential, client, or port composition failure."""

    def __init__(self, *, code: str) -> None:
        self.code = code
        super().__init__(f"D1 R2 runtime composition: {code}")


def _error(code: str) -> D1R2RuntimeCompositionError:
    return D1R2RuntimeCompositionError(code=code)


@dataclass(slots=True, repr=False)
class D1BoundR2CheckpointSync(D1CheckpointSyncPort):
    _adapter: D1CheckpointDvcSyncAdapter
    _pod_id: str
    _attempt_sha256: str
    _logical_run_id: str
    _latest: D1PublishedCheckpointDvcSync | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if type(self._adapter) is not D1CheckpointDvcSyncAdapter:
            raise TypeError("bound R2 sync requires the DVC sync adapter")
        require_d1_pod_id(self._pod_id)
        require_sha256(self._attempt_sha256, label="bound R2 sync attempt")
        if (
            type(self._logical_run_id) is not str
            or not self._logical_run_id
            or self._logical_run_id.strip() != self._logical_run_id
        ):
            raise ValueError("bound R2 sync logical_run_id must be trimmed")

    def sync(
        self,
        *,
        successful_step: int,
        run_root: Path,
        checkpoint: D1CheckpointArtifactReceipt,
        journal_path: Path,
    ) -> None:
        published = self._adapter.sync(
            pod_id=self._pod_id,
            attempt_sha256=self._attempt_sha256,
            logical_run_id=self._logical_run_id,
            successful_step=successful_step,
            run_root=run_root,
            checkpoint=checkpoint,
            journal_path=journal_path,
        )
        if self._latest is not None and (
            published.receipt.successful_step <= self._latest.receipt.successful_step
        ):
            raise ValueError("bound R2 checkpoint sync step did not advance")
        self._latest = published

    @property
    def latest(self) -> D1PublishedCheckpointDvcSync | None:
        return self._latest


@dataclass(frozen=True, slots=True)
class D1R2RuntimeComposition:
    topology: D1DvcS3Remote
    mailbox: Boto3D1TrainerMailboxObjectAdapter = field(repr=False)
    checkpoint_recovery: D1CheckpointDvcRecoveryService = field(repr=False)
    _checkpoint_adapter: D1CheckpointDvcSyncAdapter = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.topology) is not D1DvcS3Remote:
            raise TypeError("R2 runtime composition requires DVC topology")
        if type(self.mailbox) is not Boto3D1TrainerMailboxObjectAdapter:
            raise TypeError("R2 runtime composition requires mailbox adapter")
        if type(self.checkpoint_recovery) is not D1CheckpointDvcRecoveryService:
            raise TypeError("R2 runtime composition requires checkpoint recovery")
        if type(self._checkpoint_adapter) is not D1CheckpointDvcSyncAdapter:
            raise TypeError("R2 runtime composition requires checkpoint adapter")

    def bind_checkpoint_sync(
        self,
        *,
        pod_id: str,
        attempt_sha256: str,
        logical_run_id: str,
    ) -> D1BoundR2CheckpointSync:
        return D1BoundR2CheckpointSync(
            self._checkpoint_adapter,
            pod_id,
            attempt_sha256,
            logical_run_id,
        )


def _string_environment(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or any(
        type(key) is not str or type(item) is not str for key, item in value.items()
    ):
        raise TypeError("R2 runtime environment must be a string mapping")
    return cast(Mapping[str, str], value)


def _credentials(environ: Mapping[str, str]) -> tuple[str, str]:
    access = environ.get("AWS_ACCESS_KEY_ID")
    private = environ.get("AWS_SECRET_ACCESS_KEY")
    if type(access) is not str or not access or type(private) is not str or not private:
        raise _error("credentials")
    return access, private


def _default_client_factory(service: str, **kwargs: object) -> object:
    import boto3

    return boto3.client(service, **kwargs)


def build_d1_r2_runtime_composition(
    *,
    repo_root: Path,
    environ: Mapping[str, str],
    max_objects: int,
    client_factory: D1S3ClientFactory | None = None,
    dvc: D1R2DvcPort | None = None,
    cloud: D1CloudObjectVerifierPort | None = None,
) -> D1R2RuntimeComposition:
    """Join committed topology and process credentials without persisting credentials."""
    if type(max_objects) is not int or max_objects < 1:
        raise ValueError("R2 runtime max_objects must be a positive integer")
    if dvc is not None and not all(
        callable(getattr(dvc, name, None)) for name in ("add", "pull", "push")
    ):
        raise TypeError("R2 runtime DVC port is invalid")
    if cloud is not None and not callable(getattr(cloud, "object_is_present", None)):
        raise TypeError("R2 runtime cloud port is invalid")
    selected_factory = _default_client_factory if client_factory is None else client_factory
    if not callable(selected_factory):
        raise TypeError("R2 runtime client factory must be callable")
    try:
        topology = load_d1_dvc_s3_remote(repo_root)
    except D1DvcRemoteConfigError as error:
        raise _error("topology") from error
    observed = _string_environment(environ)
    access, private = _credentials(observed)
    try:
        client = selected_factory(
            "s3",
            endpoint_url=topology.endpoint_url,
            region_name="auto",
            aws_access_key_id=access,
            aws_secret_access_key=private,
        )
        mailbox = Boto3D1TrainerMailboxObjectAdapter(
            client=client,  # type: ignore[arg-type]
            bucket=topology.bucket,
            max_objects=max_objects,
        )
    except Exception as error:
        raise _error("client") from error
    root = Path(os.path.abspath(repo_root))
    selected_dvc = SubprocessD1DvcAdapter(repo_root=root) if dvc is None else dvc
    selected_cloud = SubprocessD1CloudObjectVerifier(repo_root=root) if cloud is None else cloud
    checkpoint = D1CheckpointDvcSyncAdapter(
        repo_root=root,
        dvc=selected_dvc,
        cloud=selected_cloud,
        mailbox=mailbox,
    )
    recovery = D1CheckpointDvcRecoveryService(
        repo_root=root,
        objects=mailbox,
        dvc=selected_dvc,
    )
    return D1R2RuntimeComposition(topology, mailbox, recovery, checkpoint)
