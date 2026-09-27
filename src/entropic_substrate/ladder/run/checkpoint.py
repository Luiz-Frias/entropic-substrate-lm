"""Immutable H checkpoints; recovery is not scientific advancement authority."""

import copy
import json
from dataclasses import asdict, dataclass
from importlib import import_module
from io import BytesIO
from pathlib import Path
from platform import python_version
from typing import cast
from zipfile import BadZipFile, ZipFile

import torch

from entropic_substrate.core.sentence.model import SentencePredictor
from entropic_substrate.core.sentence.records import content_identity
from entropic_substrate.core.sentence.recovery_state import (
    capture_rng,
    recovery_record,
    restore_rng,
    rollout_from_payload,
    rollout_to_payload,
)
from entropic_substrate.core.sentence.rollout import SentenceRollout
from entropic_substrate.core.sentence.sampling import SentenceSampler
from entropic_substrate.kernel.artifacts import publish_d1_atomic_bytes, read_d1_artifact_bytes
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.training import (
    SentencePhase,
    require_sentence_adamw_contract,
    sentence_roles,
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _measurement_digest(value: ContentHash | None) -> str | None:
    if value is None:
        return None
    if type(value) is not ContentHash or value.algorithm != "sha256":
        raise ValueError("checkpoint measurement requires a SHA-256 identity")
    return value.digest


# The reader identity's LABEL, named once (FOR-1550). It was an inline literal
# inside _reader_identity() below, which left the door
# docs/identity/read-old-table.md freezes after p4 -- label + module-set shape,
# never the digest values (ruling R-PIN) -- with no importable name for its
# label half, so tests/test_identity_epoch.py could pin the composed digest but
# not the label that composes it.
_READER_IDENTITY_LABEL = "h-checkpoint-reader-v2"

# Faithful restatement (ruling E-2) of the pre-move member set at their landed
# dotted names: the 45 already-landed sentence_prediction.* leaves (the H
# domain's 18 core.sentence modules, 22 ladder.run modules, 1 ladder.capture
# module, 4 composition CLI modules, 1 core.probe module), this reader's own
# file (ladder.run.checkpoint), the landed package member (core.sentence,
# resolving to its own __init__.py rather than the pre-move bare
# "sentence_prediction" name, per the p4 design-gate package-member
# constraint), and the 4 reader-closure rail modules (kernel.artifacts,
# kernel.lifecycle, core.latent.metric, core.latent.dynamics). 51 members,
# reconciled against tools/mapping.json and docs/identity/read-old-table.md.
_READER_MODULES = (
    "composition.capture_cli",
    "composition.cli",
    "composition.source_cli",
    "composition.span_cli",
    "core.latent.dynamics",
    "core.latent.metric",
    "core.probe.score_gain",
    "core.sentence",
    "core.sentence.boundary",
    "core.sentence.codec",
    "core.sentence.consumer",
    "core.sentence.corpus",
    "core.sentence.head",
    "core.sentence.law",
    "core.sentence.model",
    "core.sentence.numerics",
    "core.sentence.objective",
    "core.sentence.profile",
    "core.sentence.records",
    "core.sentence.recovery_state",
    "core.sentence.rollout",
    "core.sentence.runtime_precision",
    "core.sentence.sampling",
    "core.sentence.scoring",
    "core.sentence.segmentation",
    "core.sentence.spine",
    "kernel.artifacts",
    "kernel.lifecycle",
    "ladder.capture.capture",
    "ladder.run.checkpoint",
    "ladder.run.evaluation_records",
    "ladder.run.execution",
    "ladder.run.generation",
    "ladder.run.journal",
    "ladder.run.mechanism_evidence",
    "ladder.run.native_verification",
    "ladder.run.object_worker",
    "ladder.run.runner",
    "ladder.run.seals",
    "ladder.run.span_diagnostics",
    "ladder.run.span_probe",
    "ladder.run.span_probe_artifacts",
    "ladder.run.span_states",
    "ladder.run.trace_capture",
    "ladder.run.training",
    "ladder.run.worker_command",
    "ladder.run.worker_completion",
    "ladder.run.worker_execution",
    "ladder.run.worker_request",
    "ladder.run.worker_snapshot",
    "ladder.run.worker_transport",
)


def _reader_identity() -> str:
    digests = []
    for name in _READER_MODULES:
        module = import_module(f"entropic_substrate.{name}")
        if module.__file__ is None:
            raise RuntimeError(f"checkpoint reader cannot bind module {name}")
        source = Path(module.__file__)
        _, _, digest = read_d1_artifact_bytes(source.parent, source.name, label="checkpoint reader")
        digests.append(f"{name}:{digest}")
    return content_identity(
        (_READER_IDENTITY_LABEL, python_version(), str(torch.__version__), *digests)
    )


def checkpoint_binding(
    model: SentencePredictor,
    optimizer: torch.optim.Optimizer,
    sampler: SentenceSampler,
    phase: SentencePhase,
    run_identity: str,
) -> str:
    ContentHash("sha256", run_identity)
    roles, _ = sentence_roles(model, phase)
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    expected = [p for parameters in roles.values() for p in parameters]
    if len(owned) != len(expected) or {id(p) for p in owned} != {id(p) for p in expected}:
        raise ValueError("checkpoint optimizer ownership mismatch")
    require_sentence_adamw_contract(optimizer, phase)
    names = {id(p): name for name, p in model.named_parameters()}
    return _canonical(
        {
            "run_identity": run_identity,
            "phase": phase,
            "source": sampler.identity,
            "codec": asdict(model.codec.spec),
            "spine": asdict(model.spine.spec),
            "law": asdict(model.head.spec),
            "consumer": model.consumer_mode,
            "solver": asdict(model.solver),
            "optimizer": optimizer.defaults,
            "groups": [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups],
            "device": str(model.spine.bos.device),
            "dtype": str(model.spine.bos.dtype),
        }
    )


def _validate_optimizer_state(optimizer: torch.optim.Optimizer, step: int) -> None:
    _optimizer_payload(optimizer.state_dict(), optimizer, step)


def _optimizer_payload(
    value: object, optimizer: torch.optim.Optimizer, step: int
) -> dict[str, object]:
    row = recovery_record(value, "state param_groups")
    if _canonical(row["param_groups"]) != _canonical(optimizer.state_dict()["param_groups"]):
        raise ValueError("serialized optimizer grouping differs from construction")
    if type(row["state"]) is not dict:
        raise ValueError("optimizer moments must be a mapping")
    states = cast(dict[int, object], row["state"])
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if step == 0:
        if states:
            raise ValueError("step-zero checkpoint contains optimizer history")
        return row
    if any(type(key) is not int for key in states) or set(states) != set(range(len(parameters))):
        raise ValueError("checkpoint lacks complete optimizer moments")
    for index, parameter in enumerate(parameters):
        state = recovery_record(states[index], "step exp_avg exp_avg_sq")
        counter = state["step"]
        if not isinstance(counter, torch.Tensor) or counter.ndim != 0 or float(counter) != step:
            raise ValueError("optimizer counter differs from checkpoint step")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("invalid checkpoint optimizer moment")
        if bool((cast(torch.Tensor, state["exp_avg_sq"]) < 0).any()):
            raise ValueError("checkpoint contains negative second moments")
    return row


def _validate_weights(model: SentencePredictor, state: object) -> dict[str, object]:
    expected = cast(dict[str, object], model.state_dict())
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("checkpoint model inventory differs from construction")
    values = cast(dict[str, object], state)
    for name, original in expected.items():
        value = values[name]
        if isinstance(original, torch.Tensor):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != original.shape
                or value.dtype != original.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("checkpoint tensor shape, precision, or finiteness mismatch")
        elif _canonical(value) != _canonical(original):
            raise ValueError("checkpoint component runtime state differs from phase")
    return values


@dataclass(frozen=True)
class RestoredSentenceState:
    model: SentencePredictor
    optimizer: torch.optim.Optimizer
    sampler: SentenceSampler
    step: int
    rollout: SentenceRollout | None


def save_checkpoint(
    root: Path,
    name: str,
    model: SentencePredictor,
    optimizer: torch.optim.Optimizer,
    sampler: SentenceSampler,
    *,
    phase: SentencePhase,
    step: int,
    run_identity: str,
    max_bytes: int,
    rollout: SentenceRollout | None = None,
    measurement: ContentHash | None = None,
) -> ArtifactIdentity:
    """Publish only a completed optimizer/source cursor, with no overwrite."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("checkpoint byte limit must be positive")
    if type(step) is not int or step < 0 or sampler.cursor != step:
        raise ValueError("checkpoint step and source cursor disagree")
    binding = checkpoint_binding(model, optimizer, sampler, phase, run_identity)
    _validate_optimizer_state(optimizer, step)
    weights = _validate_weights(model, model.state_dict())
    payload = {
        "schema": "h-training-checkpoint-v2",
        "measurement": _measurement_digest(measurement),
        "reader": _reader_identity(),
        "binding": binding,
        "step": step,
        "source": sampler.snapshot(),
        "training": model.training,
        "model": weights,
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng(),
        "rollout": rollout_to_payload(rollout, model, sampler.codec),
    }
    stream = BytesIO()
    torch.save(payload, stream)
    encoded = stream.getvalue()
    if len(encoded) > max_bytes:
        raise ValueError("checkpoint exceeds its byte limit")
    return publish_d1_atomic_bytes(root, name, encoded, label="H training checkpoint")


def _checkpoint_payload(
    root: Path,
    name: str,
    expected: ContentHash,
    *,
    max_bytes: int,
) -> dict[str, object]:
    if expected.algorithm != "sha256":
        raise ValueError("checkpoint requires an external SHA-256 identity")
    _, payload, digest = read_d1_artifact_bytes(
        root, name, label="H checkpoint", max_bytes=max_bytes
    )
    if digest != expected.digest:
        raise ValueError("checkpoint content hash mismatch")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            if sum(info.file_size for info in archive.infolist()) > max_bytes:
                raise ValueError("expanded checkpoint exceeds byte limit")
    except BadZipFile as error:
        raise ValueError("checkpoint is not the declared Torch archive format") from error
    saved: object = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    if type(saved) is dict and saved.get("schema") == "h-training-checkpoint-v1":
        raise ValueError("checkpoint v1 requires its original reader; no implicit migration")
    row = recovery_record(
        saved, "schema reader binding step source training model optimizer rng rollout measurement"
    )
    if row["schema"] != "h-training-checkpoint-v2" or row["reader"] != _reader_identity():
        raise ValueError("checkpoint version or reader identity is incompatible")
    return row


def checkpoint_metadata(
    root: Path, name: str, expected: ContentHash, *, max_bytes: int
) -> dict[str, object]:
    """Read verified identities only; full tensor/optimizer admission is load_checkpoint."""
    row = _checkpoint_payload(root, name, expected, max_bytes=max_bytes)
    return {
        key: row[key]
        for key in ("schema", "reader", "binding", "step", "source", "training", "measurement")
    }


def load_checkpoint(
    root: Path,
    name: str,
    expected: ContentHash,
    model: SentencePredictor,
    optimizer: torch.optim.Optimizer,
    sampler: SentenceSampler,
    *,
    phase: SentencePhase,
    run_identity: str,
    max_bytes: int,
    expected_measurement: ContentHash | None = None,
) -> RestoredSentenceState:
    """Validate into separate objects; caller replaces live state only on success."""
    row = _checkpoint_payload(root, name, expected, max_bytes=max_bytes)
    if row["measurement"] != _measurement_digest(expected_measurement):
        raise ValueError("checkpoint measurement identity differs from expected evidence")
    binding = checkpoint_binding(model, optimizer, sampler, phase, run_identity)
    if row["binding"] != binding:
        raise ValueError("checkpoint construction identity differs from expected run")
    step, training = row["step"], row["training"]
    if type(step) is not int or step < 0 or type(training) is not bool:
        raise ValueError("checkpoint step or training mode is invalid")
    weights = _validate_weights(model, row["model"])
    optimizer_state = _optimizer_payload(row["optimizer"], optimizer, step)
    new_model, new_optimizer = copy.deepcopy((model, optimizer))
    new_model.load_state_dict(weights, strict=True)
    new_optimizer.load_state_dict(optimizer_state)
    new_model.train(training)
    new_sampler = copy.copy(sampler)
    new_sampler.restore(row["source"])
    if new_sampler.cursor != step:
        raise ValueError("checkpoint source cursor differs from optimizer step")
    if checkpoint_binding(new_model, new_optimizer, new_sampler, phase, run_identity) != binding:
        raise ValueError("restored optimizer contract differs from expected binding")
    _validate_optimizer_state(new_optimizer, step)
    rollout = rollout_from_payload(row["rollout"], new_model, new_sampler.codec)
    restore_rng(row["rng"])
    return RestoredSentenceState(new_model, new_optimizer, new_sampler, step, rollout)
