"""Real D1 fresh/resume trainer services behind the public command boundary."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from entropic_substrate.d1_billing_finalization_values import D1TrainerHandoff
from entropic_substrate.d1_cli import (
    D1CliCommand,
    D1CliInvalidEvidence,
    D1CliInvocation,
)
from entropic_substrate.d1_fresh_trainer import (
    D1FreshRunFailure,
    run_d1_fresh,
)
from entropic_substrate.d1_observability import D1JournalObserver
from entropic_substrate.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    D1TrainerExportReceipt,
    verify_d1_trainer_export,
)
from entropic_substrate.d1_resource_activation import D1ResourceActivationFactory
from entropic_substrate.d1_resume_trainer import (
    D1ResumeRunFailure,
    run_d1_resume,
)
from entropic_substrate.d1_trainer_output_publication import (
    load_d1_published_trainer_output,
    publish_d1_trainer_output,
)
from entropic_substrate.d1_trainer_request_codec import (
    D1FreshTrainerRequest,
    D1ResumeTrainerRequest,
    D1TrainerRequestError,
    parse_d1_trainer_request,
)
from entropic_substrate.d1_trainer_runtime import (
    D1TrainerClock,
    SystemD1TrainerClock,
)
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.experiments.d1_config import D1LaunchProfile

__all__ = ["D1FreshTrainerCliService", "D1ResumeTrainerCliService"]


def _root(value: Path) -> Path:
    if not isinstance(cast(object, value), Path) or value.is_symlink() or not value.is_dir():
        raise ValueError("D1 trainer service artifact_root must be a real directory")
    return value.resolve()


def _catalog(values: Iterable[D1LaunchProfile]) -> tuple[D1LaunchProfile, ...]:
    result = tuple(values)
    if not result or any(type(value) is not D1LaunchProfile for value in result):
        raise ValueError("D1 trainer service requires an installed profile catalog")
    if len({(value.name, value.scale) for value in result}) != len(result):
        raise ValueError("D1 trainer service profile name/scale identities must be unique")
    return result


def _request_bytes(invocation: D1CliInvocation) -> bytes:
    try:
        return json.dumps(
            dict(invocation.payload),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise D1CliInvalidEvidence(
            f"D1 trainer request payload is not canonical: {error}"
        ) from error


def _result(
    receipt: D1TrainerExportReceipt,
    marker: ArtifactIdentity,
) -> Mapping[str, object]:
    termination = receipt.handoff.termination
    return {
        "execution_segment_index": termination.execution_segment_index,
        "export": receipt.manifest,
        "handoff_kind": receipt.handoff_kind,
        "logical_run_id": termination.logical_run_id,
        "marker": {
            "path": marker.path,
            "sha256": marker.content_hash.digest,
        },
        "schema_version": "d1-trainer-cli-result-v1",
        "successful_step": termination.successful_step,
    }


def _publish_result(
    handoff: D1TrainerHandoff,
    *,
    journal_path: Path,
    checkpoint_path: Path,
    artifact_root: Path,
) -> Mapping[str, object]:
    try:
        prefix = checkpoint_path.relative_to(artifact_root).as_posix()
        candidate = D1ExportedTrainerHandoff(handoff, journal_path)
        marker = publish_d1_trainer_output(
            candidate,
            artifact_root=artifact_root,
            publication_prefix=prefix,
        )
        published = load_d1_published_trainer_output(
            artifact_root=artifact_root,
            publication_prefix=prefix,
        )
        return _result(verify_d1_trainer_export(published), marker)
    except ValueError as error:
        raise D1CliInvalidEvidence(f"D1 trainer export publication failed: {error}") from error


class D1FreshTrainerCliService:
    def __init__(
        self,
        *,
        catalog: Iterable[D1LaunchProfile],
        artifact_root: Path | None = None,
        clock: D1TrainerClock | None = None,
        resource_activation_factory: D1ResourceActivationFactory | None = None,
    ) -> None:
        self._catalog = _catalog(catalog)
        self._artifact_root = None if artifact_root is None else _root(artifact_root)
        self._clock = clock or SystemD1TrainerClock()
        self._resource_activation_factory = resource_activation_factory

    def execute(self, invocation: D1CliInvocation) -> Mapping[str, object]:
        if type(invocation) is not D1CliInvocation:
            raise TypeError("D1 fresh trainer service requires a CLI invocation")
        if invocation.command is not D1CliCommand.RUN_FRESH:
            raise D1CliInvalidEvidence("D1 fresh trainer service received another command")
        try:
            decoded = parse_d1_trainer_request(
                _request_bytes(invocation),
                catalog=self._catalog,
            )
            if type(decoded) is not D1FreshTrainerRequest:
                raise D1CliInvalidEvidence("D1 fresh command requires a fresh trainer request")
            root = self._artifact_root or _root(invocation.manifest_path.parent)
            locations = decoded.resolve_locations(root)
        except (D1TrainerRequestError, ValueError) as error:
            raise D1CliInvalidEvidence(str(error)) from error
        try:
            handoff = run_d1_fresh(
                plan=decoded.plan,
                start=decoded.start,
                origin=decoded.origin,
                locations=locations,
                clock=self._clock,
                resource_activation_factory=self._resource_activation_factory,
            )
        except D1FreshRunFailure as error:
            handoff = error.handoff
        return _publish_result(
            handoff,
            journal_path=locations.journal_path,
            checkpoint_path=locations.checkpoint_path,
            artifact_root=root,
        )


class D1ResumeTrainerCliService:
    def __init__(
        self,
        *,
        catalog: Iterable[D1LaunchProfile],
        artifact_root: Path | None = None,
        observer: D1JournalObserver,
        clock: D1TrainerClock | None = None,
        resource_activation_factory: D1ResourceActivationFactory | None = None,
    ) -> None:
        self._catalog = _catalog(catalog)
        self._artifact_root = None if artifact_root is None else _root(artifact_root)
        self._observer = observer
        self._clock = clock or SystemD1TrainerClock()
        self._resource_activation_factory = resource_activation_factory

    def execute(self, invocation: D1CliInvocation) -> Mapping[str, object]:
        if type(invocation) is not D1CliInvocation:
            raise TypeError("D1 resume trainer service requires a CLI invocation")
        if invocation.command is not D1CliCommand.RUN_RESUME:
            raise D1CliInvalidEvidence("D1 resume trainer service received another command")
        try:
            decoded = parse_d1_trainer_request(
                _request_bytes(invocation),
                catalog=self._catalog,
            )
            if type(decoded) is not D1ResumeTrainerRequest:
                raise D1CliInvalidEvidence("D1 resume command requires a resume trainer request")
            root = self._artifact_root or _root(invocation.manifest_path.parent)
            locations = decoded.resolve_locations(root)
        except (D1TrainerRequestError, ValueError) as error:
            raise D1CliInvalidEvidence(str(error)) from error
        try:
            handoff = run_d1_resume(
                plan=decoded.plan,
                start=decoded.start,
                origin=decoded.origin,
                locations=locations,
                observer=self._observer,
                clock=self._clock,
                resource_activation_factory=self._resource_activation_factory,
            )
        except D1ResumeRunFailure as error:
            handoff = error.handoff
        return _publish_result(
            handoff,
            journal_path=locations.journal_path,
            checkpoint_path=locations.checkpoint_path,
            artifact_root=root,
        )
