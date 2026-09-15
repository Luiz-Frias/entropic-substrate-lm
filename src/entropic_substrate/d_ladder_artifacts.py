"""Descriptor-bound D-ladder checkpoints with explicitly restricted smoke readers."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from entropic_substrate.d1_artifact_snapshot import read_d1_artifact_bytes
from entropic_substrate.experiments.d1_primitives import canonical_digest
from entropic_substrate.experiments.ladder import RungSpec
from entropic_substrate.predictive_state.ladder_runtime import LadderPredictiveSystem

if TYPE_CHECKING:
    from entropic_substrate.d_ladder_execution import SmokeRequest
    from entropic_substrate.d_ladder_request import ScientificRequest


def _mapping(content: bytes, label: str) -> dict[str, object]:
    value = json.loads(content)
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def read_ladder_checkpoint(
    manifest: Mapping[str, object], checkpoint: Path, *, expected_sha256: str | None = None
) -> tuple[dict[str, Any], dict[str, object]]:
    """Load the exact no-follow bytes hashed and bound to this construction.

    This reader provides integrity and smoke provenance. It grants neither
    pilot admissibility nor a scientific result classification.
    """
    root = checkpoint.parent
    smoke = manifest.get("schema") == "d-ladder-local-smoke-v1"
    if not smoke and manifest.get("schema") != "d-ladder-scientific-run-v1":
        raise ValueError("unsupported ladder execution manifest")
    prefix = "d-ladder-local" if smoke else "d-ladder-scientific"
    _, manifest_bytes, _ = read_d1_artifact_bytes(root, "manifest.json", label="smoke manifest")
    if _mapping(manifest_bytes, "smoke manifest") != json.loads(json.dumps(manifest)):
        raise ValueError("smoke manifest mismatch")
    identity = canonical_digest(manifest)
    _, event_bytes, _ = read_d1_artifact_bytes(
        root, checkpoint.with_suffix(".json").name, label="smoke step"
    )
    event = _mapping(event_bytes, "smoke step")
    if (
        event.get("schema") != prefix + "-step-v1"
        or event.get("run_class") != manifest["run_class"]
        or event.get("identity") != identity
    ):
        raise ValueError("step is not an identity-bound smoke/scientific artifact")
    if smoke and event.get("scientific_authority") != "none":
        raise ValueError("smoke has no scientific authority")
    if not smoke and event.get("authorization_digest") != canonical_digest(
        cast(Mapping[str, object], manifest["authorization"])
    ):
        raise ValueError("scientific step authorization digest mismatch")
    _, checkpoint_bytes, digest = read_d1_artifact_bytes(
        root, checkpoint.name, label="smoke checkpoint"
    )
    if event.get("checkpoint_sha256") != digest:
        raise ValueError("smoke checkpoint digest mismatch")
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError("smoke checkpoint caller digest mismatch")
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("smoke checkpoint must be a mapping")
    saved = cast(dict[str, Any], payload)
    if saved.get("schema") != prefix + "-checkpoint-v1" or saved.get("identity") != identity:
        raise ValueError("smoke checkpoint construction or execution identity mismatch")
    step = saved.get("step")
    if type(step) is not int or step < (1 if smoke else 0) or event.get("step") != step:
        raise ValueError("smoke checkpoint and step record disagree")
    if checkpoint.name != f"step-{step:06d}.pt":
        raise ValueError("smoke checkpoint filename disagrees with its step")
    measurements = {key: value for key, value in event.items() if key != "checkpoint_sha256"}
    if canonical_digest(measurements) != saved.get("measurement_digest"):
        raise ValueError("smoke step measurement digest mismatch")
    return saved, event


def read_smoke_checkpoint(
    manifest: Mapping[str, object], checkpoint: Path, *, expected_sha256: str | None = None
) -> tuple[dict[str, Any], dict[str, object]]:
    """Restrict smoke consumers to local evidence; share descriptor-bound reads."""
    if manifest.get("schema") != "d-ladder-local-smoke-v1":
        raise ValueError("smoke reader requires a local smoke manifest")
    return read_ladder_checkpoint(manifest, checkpoint, expected_sha256=expected_sha256)


def load_smoke_model(
    request: SmokeRequest, checkpoint: Path, *, expected_sha256: str | None = None
) -> LadderPredictiveSystem:
    """Rehydrate the verified model while preserving caller RNG state."""
    if request.manifest.get("schema") != "d-ladder-local-smoke-v1":
        raise ValueError("smoke model loader requires a local smoke request")
    return load_ladder_model(request, checkpoint, expected_sha256=expected_sha256)


def load_ladder_model(
    request: SmokeRequest | ScientificRequest,
    checkpoint: Path,
    *,
    expected_sha256: str | None = None,
) -> LadderPredictiveSystem:
    """Load the same verified checkpoint bytes used by diagnostic provenance."""
    return load_bound_ladder_model(
        request.spec, request.manifest, checkpoint, expected_sha256=expected_sha256
    )


def load_bound_ladder_model(
    spec: RungSpec,
    manifest: Mapping[str, object],
    checkpoint: Path,
    *,
    expected_sha256: str | None = None,
) -> LadderPredictiveSystem:
    """Verify diagnostic construction and artifact identity without granting launch authority."""
    if canonical_digest(cast(Mapping[str, object], manifest["construction"])) != spec.digest:
        raise ValueError("diagnostic profile differs from checkpoint construction")
    saved, _ = read_ladder_checkpoint(manifest, checkpoint, expected_sha256=expected_sha256)
    with torch.random.fork_rng(devices=[]):
        system = LadderPredictiveSystem(spec)
        system.load_state_dict(saved["model"], strict=True)
    return system.eval()


def read_smoke_capture(request: SmokeRequest, root: Path) -> dict[str, object]:
    """Validate a complete local step prefix; never synthesize a verdict."""
    steps = []
    for expected, checkpoint in enumerate(sorted(root.glob("step-*.pt")), start=1):
        saved, event = read_smoke_checkpoint(request.manifest, checkpoint)
        if saved["step"] != expected or expected > request.max_steps:
            raise ValueError("smoke capture must contain a consecutive in-budget step prefix")
        steps.append(event)
    if not steps:
        raise ValueError("smoke capture contains no completed steps")
    if len(tuple(root.glob("step-*.json"))) != len(steps):
        raise ValueError("smoke capture contains an unmatched step record")
    return {
        "schema": "d-ladder-local-capture-v1",
        "run_class": "smoke",
        "scientific_authority": "none",
        "manifest": request.manifest,
        "steps": steps,
    }
