"""Pure filesystem admission for an authoritative D1 capture bundle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import NoReturn, cast

import entropic_substrate.telemetry as telemetry_api
from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1CaptureArtifactError,
    verify_d1_artifact,
)
from entropic_substrate.evidence.d1_capture_bundle import (
    D1AttemptEvidence,
    D1CaptureBundle,
    D1CaptureInput,
    D1CaptureSetProof,
    D1DvcCaptureHandoff,
    D1ValidatedRunCapture,
)
from entropic_substrate.evidence.d1_capture_roster import (
    D1AttemptLane,
    D1ExpectedRun,
    D1LaunchRoster,
    D1PairExclusion,
)
from entropic_substrate.evidence.d1_telemetry_values import (
    D1RunEvidence,
    D1RunStatus,
    D1TailQuarantineKind,
)
from entropic_substrate.experiments.d1_primitives import RunClass

__all__ = ["D1CaptureBundleError", "build_d1_capture_bundle"]


class D1CaptureBundleError(ValueError):
    """Stable fail-loud refusal from D1 capture admission."""

    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = MappingProxyType({"code": code, "path": path})
        super().__init__(f"D1 capture {code} at {path}: {detail}")


def _fail(code: str, path: str, detail: str, cause: Exception | None = None) -> NoReturn:
    error = D1CaptureBundleError(code=code, path=path, detail=detail)
    if cause is None:
        raise error
    raise error from cause


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        _fail("identity", path, "must be a mapping with string keys")
    return cast(Mapping[str, object], value)


def _exact(value: Mapping[str, object], expected: Mapping[str, object], path: str) -> None:
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            _fail("identity", f"{path}.{field}", "differs from the launch roster")


def _validate_plan(run: D1RunEvidence, expected: D1ExpectedRun, roster: D1LaunchRoster) -> None:
    plan = cast(Mapping[str, object], run.plan_manifest)
    _exact(
        plan,
        {
            "arm": expected.arm.value,
            "construction_digest": expected.construction_digest,
            "family_digest": expected.family_digest,
            "logical_run_id": expected.logical_run_id,
            "pairing_id": expected.pairing_id,
            "run_class": RunClass.CONFIRMATORY.value,
            "seed": expected.seed,
        },
        "plan",
    )
    _exact(
        _mapping(plan.get("code"), "plan.code"),
        {"commit": expected.code_commit, "repository_digest": expected.repository_digest},
        "plan.code",
    )
    _exact(
        _mapping(plan.get("container"), "plan.container"),
        {"image_digest": expected.image_digest},
        "plan.container",
    )
    _exact(
        _mapping(plan.get("data"), "plan.data"),
        {
            "dataset_digest": expected.dataset_digest,
            "sampler_construction_digest": expected.sampler_construction_digest,
            "split_digest": expected.split_digest,
        },
        "plan.data",
    )
    _exact(
        _mapping(plan.get("preregistration"), "plan.preregistration"),
        {
            "commit": roster.preregistration_commit,
            "family_digest": roster.family_digest,
            "path": roster.preregistration_path,
        },
        "plan.preregistration",
    )
    evidence = _mapping(plan.get("evidence"), "plan.evidence")
    if evidence.get("dvc_capture_required") is not True:
        _fail("identity", "plan.evidence.dvc_capture_required", "must be true")
    if any(segment.image_digest != expected.image_digest for segment in run.segments):
        _fail("identity", "segments.image_digest", "differs from the launch roster")


def _verify_artifacts(root: Path, artifacts: tuple[D1ArtifactIdentity, ...]) -> None:
    for artifact in artifacts:
        try:
            verify_d1_artifact(root, artifact)
        except D1CaptureArtifactError as error:
            _fail("artifact", artifact.normalized_path, str(error), error)


def _read_confirmatory(
    root: Path, capture: D1CaptureInput, expected: D1ExpectedRun, roster: D1LaunchRoster
) -> D1ValidatedRunCapture:
    _verify_artifacts(root, capture.artifacts)
    path = root / capture.journal.normalized_path
    try:
        run = telemetry_api.read_d1_run_evidence(
            path,
            expected_logical_run_id=expected.logical_run_id,
            expected_family_digest=expected.family_digest,
            expected_run_class=RunClass.CONFIRMATORY,
            expected_plan_digest=expected.plan_digest,
        )
    except Exception as error:
        _fail("authoritative_reader", capture.journal.normalized_path, str(error), error)
    if run.status is not D1RunStatus.COMPLETE or run.terminal is None:
        _fail("terminal_journal", capture.journal.normalized_path, "run is not complete")
    identity = run.identity
    _exact(
        cast(
            Mapping[str, object],
            {
                "aggregation_digest": identity.aggregation_digest,
                "construction_digest": identity.construction_digest,
                "execution_digest": identity.execution_digest,
                "family_digest": identity.family_digest,
            },
        ),
        {
            "aggregation_digest": expected.aggregation_digest,
            "construction_digest": expected.construction_digest,
            "execution_digest": expected.execution_digest,
            "family_digest": expected.family_digest,
        },
        "journal",
    )
    _validate_plan(run, expected, roster)
    if capture.checkpoint.sha256 != run.final_checkpoint_sha256:
        _fail(
            "checkpoint_receipt",
            capture.checkpoint.normalized_path,
            "terminal checkpoint bytes differ from the journal receipt",
        )
    _validate_quarantine(run, capture)
    return D1ValidatedRunCapture(
        expected=expected,
        capture_input=capture,
        terminal_step=run.terminal.successful_step,
        data_order_segments=run.data_order_segments,
        checkpoint_receipts=run.checkpoint_receipts,
    )


def _validate_quarantine(run: D1RunEvidence, capture: D1CaptureInput) -> None:
    expected = tuple(
        item
        for item in run.tail_quarantine_evidence
        if item.kind is D1TailQuarantineKind.QUARANTINED_TAIL
    )
    if len(expected) != len(capture.quarantined_siblings):
        _fail("quarantine_set", capture.logical_run_id, "quarantined sibling count differs")
    parent = PurePosixPath(capture.journal.normalized_path).parent
    by_path = {item.normalized_path: item for item in capture.quarantined_siblings}
    for evidence in expected:
        path = (parent / cast(str, evidence.artifact_file_name)).as_posix()
        artifact = by_path.get(path)
        if (
            artifact is None
            or artifact.sha256 != evidence.artifact_sha256
            or artifact.size_bytes != evidence.artifact_byte_count
        ):
            _fail("quarantine_identity", path, "does not match authoritative journal evidence")


def _validate_pairing(captures: tuple[D1ValidatedRunCapture, ...]) -> None:
    by_pair: dict[str, set[tuple[tuple[int, str, int], ...]]] = {}
    for capture in captures:
        by_pair.setdefault(capture.expected.pairing_id, set()).add(capture.data_order_segments)
    for pairing_id, identities in by_pair.items():
        if len(identities) != 1:
            _fail("data_order_pairing", pairing_id, "paired arms consumed different source order")


def _validate_attempts(
    root: Path,
    roster: D1LaunchRoster,
    evidence: tuple[D1AttemptEvidence, ...],
) -> tuple[D1ArtifactIdentity, ...]:
    if type(evidence) is not tuple:
        raise TypeError("attempt_evidence must be a tuple")
    by_id = {item.attempt.logical_run_id: item for item in evidence}
    if len(by_id) != len(evidence) or frozenset(by_id) != frozenset(
        item.logical_run_id for item in roster.attempts
    ):
        _fail("attempt evidence set", "attempts", "does not equal the declared attempt roster")
    artifacts: list[D1ArtifactIdentity] = []
    for declared in roster.attempts:
        item = by_id[declared.logical_run_id]
        if item.attempt != declared:
            _fail("attempt_identity", declared.logical_run_id, "differs from the declared lane")
        _verify_artifacts(root, item.artifacts)
        artifacts.extend(item.artifacts)
        if item.journal is None:
            continue
        try:
            run = telemetry_api.read_d1_run_evidence(
                root / item.journal.normalized_path,
                expected_logical_run_id=declared.logical_run_id,
                expected_run_class=declared.run_class,
                allow_incomplete=True,
            )
        except Exception as error:
            _fail("authoritative_reader", item.journal.normalized_path, str(error), error)
        allowed = {
            D1AttemptLane.EXCLUDED_PILOT: {D1RunStatus.COMPLETE},
            D1AttemptLane.ABORTED: {D1RunStatus.TRAINER_ABORTED, D1RunStatus.RUN_ABORTED},
            D1AttemptLane.ABANDONED: {D1RunStatus.PARTIAL},
        }[declared.lane]
        if run.status not in allowed:
            _fail("attempt_status", declared.logical_run_id, "journal contradicts its lane")
    return tuple(artifacts)


def build_d1_capture_bundle(
    root: str | Path,
    *,
    roster: D1LaunchRoster,
    capture_inputs: tuple[D1CaptureInput, ...],
    attempt_evidence: tuple[D1AttemptEvidence, ...] = (),
    pair_exclusions: tuple[D1PairExclusion, ...] = (),
) -> D1CaptureBundle:
    """Admit a complete local bundle without invoking DVC, MLflow, or a remote."""
    if type(roster) is not D1LaunchRoster:
        raise TypeError("roster must be a D1LaunchRoster")
    if type(capture_inputs) is not tuple:
        raise TypeError("capture_inputs must be a tuple")
    by_id = {item.logical_run_id: item for item in capture_inputs}
    if len(by_id) != len(capture_inputs) or frozenset(by_id) != roster.expected_run_ids:
        _fail("capture input set", "confirmatory", "does not equal the expected launch roster")
    exclusions = roster.validate_pair_exclusions(pair_exclusions)
    capture_root = Path(root)
    attempt_artifacts = _validate_attempts(capture_root, roster, attempt_evidence)
    validated = tuple(
        _read_confirmatory(capture_root, by_id[expected.logical_run_id], expected, roster)
        for expected in roster.expected_confirmatory
    )
    _validate_pairing(validated)
    all_artifacts = (
        tuple(artifact for item in capture_inputs for artifact in item.artifacts)
        + attempt_artifacts
    )
    if len({item.normalized_path for item in all_artifacts}) != len(all_artifacts):
        _fail("artifact_set", "bundle", "artifact path is reused across run or attempt lanes")
    terminal_ids = frozenset(item.expected.logical_run_id for item in validated)
    proof = D1CaptureSetProof(
        roster.expected_run_ids,
        terminal_ids,
        terminal_ids,
        frozenset(by_id),
    )
    return D1CaptureBundle.from_validated_evidence(
        roster=roster,
        set_proof=proof,
        confirmatory=validated,
        attempts=attempt_evidence,
        pair_exclusions=exclusions,
        dvc_handoff=D1DvcCaptureHandoff(all_artifacts),
    )
