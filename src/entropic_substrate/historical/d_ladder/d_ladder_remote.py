"""D2-D5 worker bridge over the existing immutable R2 object port.

This module never allocates resources or grants successor authority. The trusted
worker composition supplies observed runtime provenance and a verified filing
loader. Ready/start receipts acknowledge admission/dispatch; only validated
training artifacts establish scientific completion.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from entropic_substrate.historical.d_ladder.d_ladder_remote_request import (
    RemoteLadderRequest,
    parse_remote_request,
    serialize_remote_request,
)
from entropic_substrate.historical.d_ladder.d_ladder_request import ScientificRequest
from entropic_substrate.historical.d_ladder.d_ladder_resources import LadderResources
from entropic_substrate.historical.d_ladder.ladder_profile import load_rung_profile
from entropic_substrate.kernel.artifacts import publish_d1_atomic_bytes, read_d1_artifact_bytes
from entropic_substrate.kernel.evidence_class.resource_provenance import RunStartProvenance
from entropic_substrate.kernel.identity.d1_primitives import canonical_json
from entropic_substrate.ladder.gate.ladder_campaign import (
    FiledLadderGate,
    authorize_ladder_run,
    object_fields,
    read_bound_artifact,
    read_json_artifact,
)
from entropic_substrate.ladder.run.resource.d1_resource_activation import (
    D1ResourceActivationFactory,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)


def load_remote_predecessor(root: Path, request: RemoteLadderRequest) -> FiledLadderGate:
    """Select the actual filing verifier for the immediate predecessor rung."""
    spec = load_rung_profile(root, request.profile)
    if spec.rung.value == "d2":
        from entropic_substrate.ladder.run.worker.d1_predecessor_loader import load_d1_predecessor

        return load_d1_predecessor(root, request.predecessor_declaration)
    from entropic_substrate.ladder.gate.ladder_predecessor import load_ladder_predecessor

    return load_ladder_predecessor(root, request.predecessor_declaration)


def _publish(
    objects: D1TrainerMailboxObjectPort, key: str, payload: bytes
) -> D1TrainerMailboxObjectIdentity:
    expected = D1TrainerMailboxObjectIdentity(
        key, len(payload), hashlib.sha256(payload).hexdigest()
    )
    observed = objects.publish(key, payload, expected_sha256=expected.sha256)
    if type(observed) is not D1TrainerMailboxObjectIdentity or observed != expected:
        raise ValueError("remote object publication identity mismatch")
    return observed


def _read(objects: D1TrainerMailboxObjectPort, identity: D1TrainerMailboxObjectIdentity) -> bytes:
    observed = objects.read(identity.key)
    if type(observed) is not D1TrainerMailboxObjectRead or observed.identity != identity:
        raise ValueError("remote object read identity mismatch")
    return observed.payload


def publish_remote_request(
    objects: D1TrainerMailboxObjectPort,
    request: RemoteLadderRequest,
    root: Path,
) -> D1TrainerMailboxObjectIdentity:
    """Publish verified inputs first and the request commit marker last."""
    for artifact in request.inputs:
        payload = read_bound_artifact(root, artifact)
        _publish(objects, f"{request.prefix}/inputs/{artifact.path}", payload)
    return _publish(objects, f"{request.prefix}/request.json", serialize_remote_request(request))


def materialize_remote_request(
    objects: D1TrainerMailboxObjectPort,
    identity: D1TrainerMailboxObjectIdentity,
    root: Path,
) -> RemoteLadderRequest:
    """Snapshot each expected object into an empty local workspace, refusing drift."""
    request = parse_remote_request(_read(objects, identity))
    if identity.key != f"{request.prefix}/request.json":
        raise ValueError("remote request object key does not match its attempt")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or any(root.iterdir()):
        raise FileExistsError("remote workspace must be an empty real directory")
    for artifact in request.inputs:
        key = f"{request.prefix}/inputs/{artifact.path}"
        observed = objects.read(key)
        if (
            type(observed) is not D1TrainerMailboxObjectRead
            or observed.identity.key != key
            or observed.identity.sha256 != artifact.content_hash.digest
        ):
            raise ValueError("remote input object identity mismatch")
        destination = root / artifact.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        publish_d1_atomic_bytes(root, artifact.path, observed.payload, label="remote input")
    return request


def _receipt(
    objects: D1TrainerMailboxObjectPort,
    request: RemoteLadderRequest,
    kind: str,
    fields: Mapping[str, object],
    *,
    version: int = 1,
) -> D1TrainerMailboxObjectIdentity:
    return _publish(
        objects,
        f"{request.prefix}/{kind}.json",
        canonical_json(
            {
                "schema": f"d-ladder-remote-{kind}-v{version}",
                "request_sha256": request.digest,
                "authorization_digest": request.authorization_digest,
                "logical_run_id": request.start.logical_run_id,
                "execution_segment_id": request.start.execution_segment_id,
                **fields,
            }
        ).encode(),
    )


def _export_run(
    objects: D1TrainerMailboxObjectPort,
    request: RemoteLadderRequest,
    output: Path,
    exported: dict[str, dict[str, object]],
) -> None:
    for path in sorted(output.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(output).as_posix()
        if relative in exported:
            continue  # The shared producer writes immutable, exclusive-create artifacts.
        _, payload, digest = read_d1_artifact_bytes(output, relative, label="remote output")
        published = _publish(objects, f"{request.prefix}/outputs/{relative}", payload)
        exported[relative] = {
            "path": relative,
            "key": published.key,
            "sha256": digest,
            "size_bytes": published.size_bytes,
        }


def materialize_remote_output(
    objects: D1TrainerMailboxObjectPort,
    request: RemoteLadderRequest,
    identity: D1TrainerMailboxObjectIdentity,
    root: Path,
) -> dict[str, object]:
    """Verify a complete/checkpoint receipt and its exact exported byte inventory."""
    kind = identity.key.removeprefix(f"{request.prefix}/").removesuffix(".json")
    if identity.key != f"{request.prefix}/{kind}.json" or not (
        kind == "complete" or re.fullmatch(r"checkpoint-step-\d{6,}", kind)
    ):
        raise ValueError("remote output receipt key is outside this attempt")
    payload = _read(objects, identity)
    value = json.loads(payload)
    timed = (
        kind == "complete"
        and type(value) is dict
        and value.get("schema") == "d-ladder-remote-complete-v2"
    )
    fields = "schema request_sha256 authorization_digest logical_run_id execution_segment_id result artifacts meaning"
    row = object_fields(value, fields + (" trainer_ended_at" if timed else ""))
    if (
        canonical_json(row).encode() != payload
        or row["schema"] != f"d-ladder-remote-{kind}-v{2 if timed else 1}"
        or row["request_sha256"] != request.digest
        or row["authorization_digest"] != request.authorization_digest
        or row["logical_run_id"] != request.start.logical_run_id
        or row["execution_segment_id"] != request.start.execution_segment_id
    ):
        raise ValueError("remote output receipt identity mismatch")
    if timed:
        if type(row["trainer_ended_at"]) is not str:
            raise ValueError("trainer end timestamp must be a UTC ISO string")
        try:
            ended = datetime.fromisoformat(row["trainer_ended_at"])
        except ValueError as error:
            raise ValueError("invalid trainer end timestamp") from error
        if ended.utcoffset() != timedelta(0) or ended < request.start.trainer_started_at:
            raise ValueError("trainer end timestamp must be UTC and not precede start")
    if type(row["artifacts"]) is not list or not row["artifacts"]:
        raise ValueError("remote output receipt requires artifact inventory")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or any(root.iterdir()):
        raise FileExistsError("remote recovery requires an empty real directory")
    for item in row["artifacts"]:
        entry = object_fields(item, "path key sha256 size_bytes")
        if (
            type(entry["path"]) is not str
            or entry["key"] != f"{request.prefix}/outputs/{entry['path']}"
        ):
            raise ValueError("remote output object path identity mismatch")
        expected = D1TrainerMailboxObjectIdentity(
            cast(str, entry["key"]), cast(int, entry["size_bytes"]), cast(str, entry["sha256"])
        )
        content = _read(objects, expected)
        (root / entry["path"]).parent.mkdir(parents=True, exist_ok=True)
        publish_d1_atomic_bytes(root, entry["path"], content, label="remote recovered output")
    return row


def run_remote_worker(
    objects: D1TrainerMailboxObjectPort,
    identity: D1TrainerMailboxObjectIdentity,
    root: Path,
    *,
    observed_start: RunStartProvenance,
    observed_code_commit: str,
    factory: D1ResourceActivationFactory,
    predecessor_loader: Callable[[Path], FiledLadderGate] | None = None,
) -> D1TrainerMailboxObjectIdentity:
    """Reauthorize captured bytes, dispatch the shared executor, and export artifacts.

    A failed dispatch emits a typed, request-bound failure receipt and rethrows;
    it never publishes a completion marker or substitutes a local smoke run.
    """
    from entropic_substrate.historical.d_ladder.d_ladder_execution import run_ladder

    request = parse_remote_request(_read(objects, identity))
    try:
        if observed_start != request.start or observed_code_commit != request.code_commit:
            raise ValueError("observed runtime provenance differs from remote request")
        resources = LadderResources(observed_start, factory)
        request = materialize_remote_request(objects, identity, root)
        gate = (
            load_remote_predecessor(root, request)
            if predecessor_loader is None
            else predecessor_loader(root)
        )
        from entropic_substrate.ladder.run.worker.d1_predecessor_loader import verified_git_bundle

        git_context = (
            nullcontext(None)
            if request.registration_git_bundle is None
            else verified_git_bundle(root, request.registration_git_bundle)
        )
        with git_context as registration_git:
            auth = authorize_ladder_run(
                root,
                request.campaign,
                request.start.logical_run_id,
                operator=request.operator,
                predecessor=gate,
                registration_git=registration_git,
            )
        registered_git = read_json_artifact(root, request.campaign)["registration_git"]
        if (
            registered_git is not None
            and cast(Mapping[str, object], registered_git)["implementation_commit"]
            != request.code_commit
        ):
            raise ValueError("observed source commit differs from preregistered implementation")
        if (
            auth.digest != request.authorization_digest
            or auth.source_sha256 != request.source.content_hash.digest
        ):
            raise ValueError("recomputed remote authorization or source differs from request")
        spec = load_rung_profile(root, request.profile)
        if factory.registration.device_scope == "cuda:0" and auth.device not in {"cuda", "cuda:0"}:
            raise ValueError("scientific device differs from registered CUDA resource scope")
        scientific = ScientificRequest(spec, auth, root, request.source.path)
        scientific.load_splits()
        _receipt(objects, request, "ready", {"meaning": "inputs-runtime-authority-verified"})
        _receipt(objects, request, "start", {"meaning": "shared-executor-dispatched"})
        output = root / "run"
        exported: dict[str, dict[str, object]] = {}

        def checkpoint_sink(checkpoint: Path) -> None:
            _export_run(objects, request, output, exported)
            _receipt(
                objects,
                request,
                f"checkpoint-{checkpoint.stem}",
                {
                    "result": {"checkpoint": checkpoint.name},
                    "artifacts": list(exported.values()),
                    "meaning": "closed-checkpoint-and-prior-records-exported",
                },
            )

        result = run_ladder(
            scientific,
            output,
            resources=resources,
            resume=None if request.resume is None else root / request.resume.path,
            checkpoint_sink=checkpoint_sink,
        )
        trainer_ended_at = datetime.now(UTC)
        _export_run(objects, request, output, exported)
        artifacts = list(exported.values())
        if not artifacts:
            raise ValueError("shared executor produced no exportable artifacts")
        return _receipt(
            objects,
            request,
            "complete",
            {
                "result": result,
                "trainer_ended_at": trainer_ended_at.isoformat(),
                "artifacts": artifacts,
                "meaning": "executor-returned-and-output-bytes-exported",
            },
            version=2,
        )
    except Exception as error:
        _receipt(objects, request, "failure", {"error_type": type(error).__name__})
        raise
