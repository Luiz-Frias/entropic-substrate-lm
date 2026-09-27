"""Checkpoint/journal identity seals with consecutive, source-bound ancestry."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.model import SentencePredictor
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.core.sentence.sampling import SamplingSpec, SentenceSampler
from entropic_substrate.kernel.artifacts import artifact_from_mapping
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.capture.capture import read_prepared_corpus
from entropic_substrate.ladder.run.checkpoint import (
    checkpoint_binding,
    checkpoint_metadata,
    save_checkpoint,
)
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import (
    artifact_reference,
    read_journal,
    read_json,
    write_journal,
    write_json,
)
from entropic_substrate.ladder.run.training import SentencePhase, build_sentence_optimizer


@dataclass(frozen=True)
class SealedStep:
    """A verified identity chain, not a scientific verdict or unchecked resume token."""

    phase: SentencePhase
    step: int
    checkpoint: ArtifactIdentity
    journal: ArtifactIdentity
    chain_length: int
    seal_hashes: tuple[str, ...] = ()


def _profile_binding(profile: HProfile, phase: SentencePhase, sampler: SentenceSampler) -> str:
    # Instantiate names/shapes without allocating another model's numerical weights.
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        model = SentencePredictor(
            profile.codec,
            profile.spine,
            profile.law,
            consumer=profile.consumer,
            solver=profile.solver,
        ).float()
        if phase == "prediction":
            model.codec.freeze_encoder()
        policy = profile.optimization
        optimizer = build_sentence_optimizer(
            model,
            phase,
            learning_rate=policy.learning_rate,
            weight_decay=policy.weight_decay,
            betas=policy.betas,
            epsilon=policy.epsilon,
        )
        value = json.loads(checkpoint_binding(model, optimizer, sampler, phase, profile.identity))
    value["device"] = "cuda:0" if profile.device == "cuda" else "cpu"
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _continuation(previous: SealedStep | None, phase: str, start: int, profile: HProfile) -> None:
    if previous is None:
        if phase != "codec_warmup" or start != 1:
            raise ValueError("initial seal must start codec warm-up at step one")
    elif previous.phase == phase:
        if start != previous.step + 1:
            raise ValueError("seal continuation is not consecutive")
    elif not (
        previous.phase == "codec_warmup"
        and phase == "prediction"
        and previous.step == profile.schedule.warmup_steps
        and start == 1
    ):
        raise ValueError("seal phase transition skipped the complete warm-up")


def read_seal(
    root: Path,
    artifact: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_bytes: int,
    max_source_bytes: int,
) -> SealedStep:
    """Validate every ancestor; full tensor admission still occurs when resuming."""
    corpus = read_prepared_corpus(
        source_root, ContentHash("sha256", profile.source_digest), max_bytes=max_source_bytes
    )
    nodes: list[tuple[SealedStep, int]] = []
    seen: set[str] = set()
    chain_hashes: list[str] = []
    cursor: ArtifactIdentity | None = artifact
    while cursor is not None:
        if (
            cursor.content_hash.digest in seen
            or len(seen) >= profile.schedule.warmup_steps + profile.schedule.steps
        ):
            raise ValueError("seal chain is cyclic or exceeds its step budget")
        seen.add(cursor.content_hash.digest)
        chain_hashes.append(cursor.content_hash.digest)
        row = recovery_record(
            read_json(root, cursor, max_bytes=max_bytes),
            "schema run_identity phase step checkpoint journal",
        )
        phase, step = row["phase"], row["step"]
        if (
            row["schema"] != "h-checkpoint-seal-v1"
            or row["run_identity"] != profile.identity
            or phase not in ("codec_warmup", "prediction")
            or type(step) is not int
            or step < 1
        ):
            raise ValueError("seal construction, phase, or step is invalid")
        checkpoint, journal = (
            artifact_from_mapping(row["checkpoint"]),
            artifact_from_mapping(row["journal"]),
        )
        measured = read_journal(
            root,
            journal,
            profile,
            source_root,
            max_bytes=max_bytes,
            max_source_bytes=max_source_bytes,
        )
        if measured["schema"] != "h-journal-shard-v2":
            raise ValueError(
                "current restart seals require journal v2; use the original reader for v1"
            )
        if measured["phase"] != phase or measured["end_step"] != step:
            raise ValueError("seal and journal endpoints disagree")
        meta = checkpoint_metadata(
            root,
            artifact_reference(root, checkpoint)["path"],
            checkpoint.content_hash,
            max_bytes=max_bytes,
        )
        if meta["measurement"] != journal.content_hash.digest:
            raise ValueError("checkpoint measurement does not belong to this journal")
        if type(meta["step"]) is not int or meta["step"] != step:
            raise ValueError("checkpoint and seal steps disagree")
        if type(meta["binding"]) is not str:
            raise ValueError("checkpoint construction binding must be serialized")
        sampler = SentenceSampler(
            corpus,
            SamplingSpec(
                "train",
                profile.schedule.batch_size,
                0 if phase == "codec_warmup" else profile.spine.context_sentences,
                profile.source_seed,
            ),
            SentenceByteCodec(profile.permutation),
            torch.device("cpu"),
        )
        if meta["binding"] != _profile_binding(profile, phase, sampler):
            raise ValueError("checkpoint construction differs from sealed profile")
        expected_source = sampler.snapshot() | {"cursor": step}
        if json.dumps(meta["source"], sort_keys=True) != json.dumps(
            expected_source, sort_keys=True
        ):
            raise ValueError("checkpoint source cursor differs from journal")
        start = measured["start_step"]
        if type(start) is not int:
            raise ValueError("journal start step must be integer")
        nodes.append((SealedStep(phase, step, checkpoint, journal, 0), start))
        cursor = None if measured["parent"] is None else artifact_from_mapping(measured["parent"])
    previous = None
    for node, start in reversed(nodes):
        _continuation(previous, node.phase, start, profile)
        previous = node
    final = nodes[0][0]
    return SealedStep(
        final.phase, final.step, final.checkpoint, final.journal, len(nodes), tuple(chain_hashes)
    )


def seal_step(
    root: Path,
    prefix: str,
    engine: SentenceExecution,
    rows: tuple[dict[str, object], ...],
    *,
    parent: ArtifactIdentity | None,
    max_bytes: int,
) -> ArtifactIdentity:
    """Publish a receipt last; partial writes are not restart seals and are retained."""
    if type(prefix) is not str or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", prefix) is None:
        raise ValueError("seal prefix must be a safe segment identifier")
    if not rows or type(rows[0].get("step")) is not int:
        raise ValueError("seal requires measured step rows")
    previous = (
        None
        if parent is None
        else read_seal(
            root,
            parent,
            engine.profile,
            engine.source_root,
            max_bytes=max_bytes,
            max_source_bytes=engine.max_source_bytes,
        )
    )
    start = rows[0]["step"]
    if type(start) is not int:
        raise ValueError("seal start step must be integer")
    _continuation(previous, engine.phase, start, engine.profile)
    if checkpoint_binding(
        engine.model, engine.optimizer, engine.sampler, engine.phase, engine.profile.identity
    ) != _profile_binding(engine.profile, engine.phase, engine.sampler):
        raise ValueError("runtime construction differs from sealed profile")
    name = f"{prefix}-{engine.phase}-{engine.step:08d}"
    journal = write_journal(
        root, name + "-journal.json", engine, rows, parent=parent, max_bytes=max_bytes
    )
    checkpoint = save_checkpoint(
        root,
        name + ".pt",
        engine.model,
        engine.optimizer,
        engine.sampler,
        phase=engine.phase,
        step=engine.step,
        run_identity=engine.profile.identity,
        measurement=journal.content_hash,
        rollout=engine.rollout,
        max_bytes=max_bytes,
    )
    return write_json(
        root,
        name + "-seal.json",
        {
            "schema": "h-checkpoint-seal-v1",
            "run_identity": engine.profile.identity,
            "phase": engine.phase,
            "step": engine.step,
            "checkpoint": artifact_reference(root, checkpoint),
            "journal": artifact_reference(root, journal),
        },
        max_bytes=max_bytes,
    )


def resume_seal(
    root: Path, artifact: ArtifactIdentity, engine: SentenceExecution, *, max_bytes: int
) -> None:
    seal = read_seal(
        root,
        artifact,
        engine.profile,
        engine.source_root,
        max_bytes=max_bytes,
        max_source_bytes=engine.max_source_bytes,
    )
    engine.resume(
        root,
        artifact_reference(root, seal.checkpoint)["path"],
        seal.checkpoint.content_hash,
        phase=seal.phase,
        max_bytes=max_bytes,
        measurement=seal.journal.content_hash,
    )
