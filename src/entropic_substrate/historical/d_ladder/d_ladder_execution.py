"""Shared D-ladder training engine with explicit smoke/scientific admission.

The same numerical path consumes synthetic tiny data or an authorized source
snapshot. Scientific capture records feed the independent registered readers;
synthetic local steps cannot be promoted into confirmatory evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import torch

from entropic_substrate.core.corpus.batching import SequenceSampler, TokenSplit
from entropic_substrate.core.corpus.ladder_batching import paired_predictive_batch
from entropic_substrate.core.corpus.ladder_source_order import SourceOrderAudit
from entropic_substrate.historical.d_ladder.d_ladder_artifacts import read_ladder_checkpoint
from entropic_substrate.historical.d_ladder.d_ladder_request import ScientificRequest
from entropic_substrate.historical.d_ladder.d_ladder_resources import LadderResources
from entropic_substrate.historical.d_ladder.d_ladder_training import (
    evaluate_ladder,
    ladder_optimizer,
    train_ladder_step,
)
from entropic_substrate.kernel.evidence_class.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.ladder.capture.resource.resource_meter_models import ResourcePhase
from entropic_substrate.ladder.capture.resource.resource_meter_runtime import SystemMonotonicClock
from entropic_substrate.ladder.register.d1_execution import D1AmpPolicy, D1ExecutionProtocol
from entropic_substrate.ladder.register.ladder import RungSpec, Treatment
from entropic_substrate.ladder.run.construction.ladder_runtime import LadderPredictiveSystem


@dataclass(frozen=True, slots=True)
class SmokeRequest:
    """CPU/synthetic execution envelope that cannot express scientific authority."""

    spec: RungSpec
    max_steps: int = 2

    def __post_init__(self) -> None:
        if type(self.spec) is not RungSpec:
            raise TypeError("smoke requires a validated RungSpec")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 200:
            raise ValueError("smoke max_steps must be a positive integer at most 200")
        if self.spec.base.encoder_pair.latent_width > 64:
            raise ValueError("local tiny width must not exceed 64")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema": "d-ladder-local-smoke-v1",
            "run_class": "smoke",
            "scientific_authority": "none",
            "dataset": "synthetic-local-smoke-v1",
            "construction": self.spec.manifest,
            "max_steps": self.max_steps,
            "batch_size": 2,
            "sequence_length": 8,
            "baseline_batches": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "clip_norm": 1.0,
            "device": "cpu",
            "amp": "off",
            "reader": "d-ladder-local-reader-v1",
            "execution": self.protocol.manifest,
        }

    @property
    def protocol(self) -> D1ExecutionProtocol:
        base = self.spec.base.execution_protocol
        return replace(
            base,
            batching=replace(base.batching, batch_size=2, sequence_length=8),
            optimization=replace(
                base.optimization,
                max_steps=self.max_steps,
                warmup_steps=0,
                learning_rate=0.001,
                min_learning_rate_ratio=1.0,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                amp=D1AmpPolicy.OFF,
            ),
            evaluation=replace(
                base.evaluation, num_batches=1, eval_interval=1, log_interval=1, save_interval=1
            ),
        )


def _json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(_json(payload) + "\n")


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def smoke_split(*, train: bool) -> TokenSplit:
    """Return canonical local fixture bytes, separate from any scientific corpus."""
    # Two distinct deterministic byte streams; identities are pinned by the schema.
    values = np.arange(4096, dtype=np.int64)
    values = (values * (73 if train else 151) + (19 if train else 37)) % 256
    return TokenSplit(values)


def run_local_smoke(
    request: SmokeRequest,
    output: Path,
    *,
    resume: Path | None = None,
    until_step: int | None = None,
) -> dict[str, object]:
    """The local adapter admits smoke only; it delegates the complete real loop."""
    if type(request) is not SmokeRequest:
        raise TypeError("local runner only admits SmokeRequest")
    return run_ladder(request, output, resume=resume, until_step=until_step)


def run_ladder(
    request: SmokeRequest | ScientificRequest,
    output: Path,
    *,
    resources: LadderResources | None = None,
    resume: Path | None = None,
    until_step: int | None = None,
    checkpoint_sink: Callable[[Path], None] | None = None,
) -> dict[str, object]:
    """One numerical path for synthetic smoke and explicitly authorized real data."""
    if type(request) not in (SmokeRequest, ScientificRequest):
        raise TypeError("runner requires SmokeRequest or ScientificRequest")
    scientific = isinstance(request, ScientificRequest)
    if scientific and resources is None:
        raise TypeError("scientific execution requires explicit LadderResources")
    if scientific and resources is not None:
        if resources.start.logical_run_id != request.authorization.logical_run_id:
            raise ValueError("resource provenance belongs to another authorized run")
    protocol = request.protocol
    final_step = request.max_steps if until_step is None else until_step
    if type(final_step) is not int or not 1 <= final_step <= request.max_steps:
        raise ValueError("until_step must fit the immutable execution budget")
    if final_step != request.max_steps and final_step % protocol.evaluation.eval_interval:
        raise ValueError("interruption endpoint must be an evaluation boundary")
    device = torch.device(request.authorization.device if scientific else "cpu")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("ladder execution supports explicit CPU or CUDA devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered CUDA execution requires available CUDA")
    train_split, eval_split = (
        request.load_splits() if scientific else (smoke_split(train=True), smoke_split(train=False))
    )
    output = output.resolve()
    manifest = request.manifest
    identity = canonical_digest(manifest)
    run_class = request.authorization.run_class if scientific else "smoke"
    schema_prefix = "d-ladder-scientific" if scientific else "d-ladder-local"
    source_order = SourceOrderAudit(protocol.batching.batch_size)
    if resume is None:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError("fresh run refuses nonempty output directory")
        _write_json(output / "manifest.json", manifest)
    saved = None
    start = 0
    online_total = target_total = 0
    if resume is not None:
        if resume.parent.resolve() != output:
            raise ValueError("resume checkpoint must belong to the output lineage")
        resume = output / resume.name
        saved, event = read_ladder_checkpoint(manifest, resume)
        source_order = SourceOrderAudit.restore(
            saved["source_order"], step=saved["step"], batch_size=protocol.batching.batch_size
        )
        start = saved["step"]
        if type(start) is not int or not 0 <= start < final_step:
            raise ValueError("resume step must precede the requested endpoint")
        if sorted(output.glob("step-*.pt"))[-1] != resume:
            raise ValueError("resume must use the latest checkpoint")
        if any(int(p.stem.removeprefix("step-")) > start for p in output.glob("step-*.json")):
            raise ValueError("post-checkpoint evidence requires explicit recovery reconciliation")
        if (
            resources is not None
            and resources.start.resume_checkpoint_sha256 != event["checkpoint_sha256"]
        ):
            raise ValueError("resource resume provenance checkpoint mismatch")
        if any(
            type(event[key]) is not int or cast(int, event[key]) < 0
            for key in ("train_online_positions", "train_target_positions")
        ):
            raise ValueError("checkpoint work counters must be nonnegative integers")
        online_total = cast(int, event["train_online_positions"])
        target_total = cast(int, event["train_target_positions"])
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    segment = nullcontext() if resources is None else resources
    checkpoint = output / f"step-{start:06d}.pt"
    clock = SystemMonotonicClock()
    with torch.random.fork_rng(devices=cuda_devices), segment:
        if resources is not None:
            resources.activate(None if saved is None else saved["resource_ledger"])
        torch.manual_seed(request.spec.seed)
        system = LadderPredictiveSystem(request.spec).to(device)
        optimizer = ladder_optimizer(system, protocol)
        sampler = SequenceSampler(
            train_split,
            protocol.batching.batch_size,
            protocol.batching.sequence_length,
            seed=request.spec.seed,
            device=device,
        )
        evaluation = SequenceSampler(
            eval_split,
            protocol.batching.batch_size,
            protocol.batching.sequence_length,
            train=False,
            seed=request.spec.seed,
            device=device,
        )
        if saved is not None:
            system.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            system.validate_optimizer_ownership(optimizer)
            sampler.load_state_dict(saved["sampler"])
            torch.set_rng_state(saved["rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        elif request.spec.treatment is not Treatment.L:
            calibration = SequenceSampler(
                train_split,
                protocol.batching.batch_size,
                protocol.batching.sequence_length,
                train=False,
                seed=request.spec.seed,
                device=device,
            )
            calibration_phase = (
                nullcontext() if resources is None else resources.measure(ResourcePhase.TRAIN)
            )
            with calibration_phase:
                observed = system.fit_mean_displacement_baseline(
                    paired_predictive_batch(
                        calibration,
                        max(request.spec.horizons),
                        reservation=request.spec.sampling_horizon_reservation,
                    )
                    for _ in range(protocol.baseline.num_batches if scientific else 2)
                )
            if resources is not None:
                resources.baseline(observed)
        steps = range(start if saved is None and scientific else start + 1, final_step + 1)
        for step in steps:
            train_ns = 0
            gradients: dict[str, object] = {}
            try:
                if step > start:
                    begin = clock.monotonic_ns()
                    gradients, online, target = train_ladder_step(
                        system, optimizer, sampler, protocol, step, resources, source_order
                    )
                    train_ns = clock.monotonic_ns() - begin
                    online_total += online
                    target_total += target
                if step != final_step and step % protocol.evaluation.eval_interval:
                    continue
                begin = clock.monotonic_ns()
                metrics, resource_snapshot = evaluate_ladder(
                    system, evaluation, protocol, resources, step
                )
                eval_ns = clock.monotonic_ns() - begin
                measurement: dict[str, object] = {
                    "schema": schema_prefix + "-step-v1",
                    "run_class": run_class,
                    "identity": identity,
                    "step": step,
                    "source_bytes": online_total,
                    "data_order": source_order.attestation,
                    "metrics": metrics,
                    "gradients": gradients,
                    "train_online_positions": online_total,
                    "train_target_positions": target_total,
                    "host_train_ns": train_ns,
                    "host_eval_ns": eval_ns,
                    "parameter_inventory": system.parameter_attestation(),
                    "resource_snapshot": resource_snapshot,
                }
                if scientific:
                    measurement["authorization_digest"] = request.authorization.digest
                    if resources is not None:
                        measurement["run_start"] = run_start_to_mapping(resources.start)
                else:
                    measurement["scientific_authority"] = "none"
                save = step in (0, final_step) or (
                    protocol.evaluation.save_interval > 0
                    and step % protocol.evaluation.save_interval == 0
                )
                checkpoint_hash = None
                if save:
                    checkpoint = output / f"step-{step:06d}.pt"
                    payload = {
                        "schema": schema_prefix + "-checkpoint-v1",
                        "identity": identity,
                        "step": step,
                        "measurement_digest": canonical_digest(measurement),
                        "model": system.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sampler": sampler.state_dict(),
                        "source_order": source_order.manifest,
                        "rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if cuda_devices else [],
                        "resource_ledger": None if resources is None else resources.state(),
                    }
                    with checkpoint.open("xb") as handle:
                        torch.save(payload, handle)
                    checkpoint_hash = _digest(checkpoint)
                _write_json(
                    output / f"step-{step:06d}.json",
                    {**measurement, "checkpoint_sha256": checkpoint_hash},
                )
                if save and checkpoint_sink is not None:
                    checkpoint_sink(checkpoint)
            except Exception as error:
                _write_json(
                    output / f"defect-{step:06d}.json",
                    {
                        "schema": "d-ladder-execution-defect-v1",
                        "identity": identity,
                        "step": step,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "run_class": run_class,
                    },
                )
                raise
    receipt: dict[str, object] = {
        "run_class": run_class,
        "step": final_step,
        "manifest_digest": identity,
        "checkpoint": str(checkpoint),
    }
    if scientific and final_step == request.max_steps:
        capture = output / "capture.json"
        _write_json(
            capture,
            {
                "schema": "d-ladder-scientific-capture-v1",
                "authorization_digest": request.authorization.digest,
                "manifest": {"path": "manifest.json", "sha256": _digest(output / "manifest.json")},
                "records": [
                    {"path": p.name, "sha256": _digest(p)}
                    for p in sorted(output.glob("step-*.json"))
                ],
                "checkpoints": [
                    {"path": p.name, "sha256": _digest(p)} for p in sorted(output.glob("step-*.pt"))
                ],
                "completed_step": final_step,
            },
        )
        receipt["capture"] = str(capture)
    elif not scientific:
        receipt["scientific_authority"] = "none"
    return receipt
