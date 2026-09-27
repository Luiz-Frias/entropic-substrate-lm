"""D1 checkpoint admission for the shared observation-only score diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from entropic_substrate.core.corpus.batching import SequenceSampler, TokenSplit
from entropic_substrate.core.latent.initialization import SeededD1BranchInitializer
from entropic_substrate.core.probe.predictive_probe import ProbeConfig
from entropic_substrate.historical.d_ladder.ladder_probe import (
    CheckpointProbeCapture,
    ProbeSource,
    run_model_probe,
)
from entropic_substrate.kernel.artifacts import normalize_d1_artifact_root
from entropic_substrate.ladder.register.d1_plan import D1RunPlan
from entropic_substrate.ladder.run.construction.runtime import D1PredictiveSystem
from entropic_substrate.ladder.run.trainer.checkpoint_repository import load_checkpoint_candidate
from entropic_substrate.ladder.run.trainer.checkpoint_validation import validate_d1_checkpoint
from entropic_substrate.ladder.run.trainer.d1_optimizer import build_d1_optimizer


@dataclass(frozen=True, slots=True)
class _D1Encoder:
    system: D1PredictiveSystem

    @property
    def block_size(self) -> int:
        return self.system.draft.encoder_pair.online.block_size

    @property
    def codec_digest(self) -> str:
        return self.system.draft.codec.digest

    @property
    def construction_identity(self) -> str:
        return self.system.construction_identity

    def encode_observations(self, canonical_ids: torch.Tensor) -> torch.Tensor:
        return self.system.online_branch(canonical_ids, self.system.draft.codec).values


def run_d1_checkpoint_probe(
    plan: D1RunPlan,
    checkpoint: Path,
    expected_sha256: str,
    training: ProbeSource,
    heldout: ProbeSource,
    output: Path,
    config: ProbeConfig | None = None,
) -> CheckpointProbeCapture:
    """Admit exact D1 identities, then diagnose on CPU without resuming training.

    The caller provides independently hashed source spans and explicit split
    identities. They remain diagnostic source evidence; the function does not
    guess that a new source span is a registered evaluation epoch. Original
    checkpoint run class is retained, and the output has no filing authority.
    """
    if not isinstance(cast(object, plan), D1RunPlan):
        raise TypeError("D1 score probe requires a typed run plan")
    if training.split != "train" or heldout.split != "heldout":
        raise ValueError("D1 probe requires training and held-out source roles")
    normalize_d1_artifact_root(checkpoint.absolute().parent)
    candidate, receipt = load_checkpoint_candidate(checkpoint, expected_sha256=expected_sha256)
    training_bytes = training.read()
    protocol = plan.profile.draft.execution_protocol
    # Materialize the exact production ownership without invoking device selection,
    # corpus downloads, training updates, target encoding, or RNG restoration.
    with torch.random.fork_rng(devices=[]):
        system = D1PredictiveSystem.from_family(
            plan.profile.family,
            plan.profile.arm,
            initializer=SeededD1BranchInitializer(seed=plan.seed),
        )
    optimizer = build_d1_optimizer(system, protocol.optimization).optimizer
    sampler = SequenceSampler(
        TokenSplit(np.frombuffer(training_bytes, dtype=np.uint8).astype(np.int64)),
        batch_size=protocol.batching.batch_size,
        seq_len=protocol.batching.sequence_length,
        seed=plan.seed,
        train=True,
        device=torch.device("cpu"),
    )
    validated = validate_d1_checkpoint(
        candidate,
        system=system,
        optimizer=optimizer,
        sampler=sampler,
        run=plan.run,
        protocol=protocol,
        for_resume=False,
    )
    system.load_state_dict(cast(dict[str, object], validated["model"]), strict=True)
    system.eval()
    return run_model_probe(
        _D1Encoder(system),
        receipt.sha256,
        plan.run.run_class.value,
        plan.run.execution_digest,
        training,
        heldout,
        output,
        ProbeConfig() if config is None else config,
    )
