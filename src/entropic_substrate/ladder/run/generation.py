"""Generation snapshots reference a sealed trained model and retain exact feedback.

Receipt v3 (FOR-1674, operator ruling D1 2026-10-23, provisional): a run draws
``profile.schedule.generation_continuations`` independent continuations of ONE prompt. Each
continuation draws only inside its own ``torch.random.fork_rng`` island, seeded exactly once
with ``torch.manual_seed(continuation_seed(...))``, so the ambient RNG neither leaks in nor
out and a continuation is reproducible from its seed alone (design gate G2; G1, an explicit
``torch.Generator`` threaded through ``advance_rollout``, is the recorded alternative: it
needs ``Beta.rsample`` to accept a generator, which ``core/sentence/law.py`` does not).
"""

import hashlib
import json
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4
from zipfile import ZipFile

import torch

from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.profile import MAX_GENERATION_CONTINUATIONS, HProfile
from entropic_substrate.core.sentence.recovery_state import (
    capture_rng,
    recovery_record,
    restore_rng,
    rollout_from_payload,
    rollout_to_payload,
)
from entropic_substrate.core.sentence.rollout import (
    SentenceRollout,
    advance_rollout,
    begin_rollout,
)
from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.capture.resource.resource_meter_models import ResourcePhase
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import artifact_reference, read_json, write_json
from entropic_substrate.ladder.run.runner import read_training_progress
from entropic_substrate.ladder.run.seals import SealMemo, resume_seal

RECEIPT_SCHEMA = "h-generation-receipt-v3"
_OLD_RECEIPT_SCHEMAS = ("h-generation-receipt-v1", "h-generation-receipt-v2")
_RECEIPT_KEYS = (
    "schema run_identity scientific_authority training snapshot status summary "
    "wall_seconds cuda_seconds parent_receipt continuations sampling"
)
_ENTRY_KEYS = "index seed sampling_law sentences log_scores outcome boundary snapshot"
_SUMMARY_KEYS = "outcome sentences observations pending log_scores prompt"
# ONE declaration of the sampler and its temperature label: the rollout draws
# ``torch.multinomial`` over ``log_softmax`` of the decoder logits with no truncation
# (core/sentence/rollout.py), i.e. temperature 1. ``TEMPERATURE`` labels that unchanged
# sampler; it is never read from configuration.
SAMPLER = "multinomial_temperature_1"
LATENT = "rsample"
TEMPERATURE = 1.0
_SENTENCE_END, _DOCUMENT_END = 256, 257  # core/sentence/boundary.py


@dataclass(frozen=True)
class GenerationProgress:
    status: Literal["paused", "generation_complete"]
    receipt: ArtifactIdentity


def continuation_seed(seed: int, steps: int, index: int) -> int:
    """``(seed * 0x9E3779B1 + ((steps << 8) | index)) % 2**32``: the seed of continuation ``index``.

    Domain (refused outside it): ``0 <= seed < 2**32`` (HProfile refuses larger),
    ``1 <= steps < 2**24`` and ``0 <= index < 256``, so ``(steps << 8) | index`` stays
    below 2**32 and is injective in ``index`` at a fixed step. The multiplier is odd, so at
    a fixed ``(steps, index)`` the map is a bijection on ``seed`` in ``[0, 2**32)``: paired
    seeds never share draws. Index 256 would equal index 0 of ``steps + 1`` (measured:
    ``(42*M + (4<<8|256)) % 2**32 == (42*M + (5<<8|0)) % 2**32``), hence the 256 cap on the
    continuation count. The CPU Mersenne Twister consumes only the low 32 bits of
    ``torch.manual_seed`` (the measurement ladder/run/execution.py records for the
    evaluation seed ``(seed * 0x9E3779B1 + step) % 2**32``), so the key is a 32-bit value.
    The two families cannot collide: an evaluation step ``t <= steps`` is below
    ``256 * steps <= (steps << 8) | index``.
    """
    if (
        any(type(n) is not int for n in (seed, steps, index))
        or not 0 <= seed < 2**32
        or not 1 <= steps < 2**24
        or not 0 <= index < MAX_GENERATION_CONTINUATIONS
    ):
        raise ValueError(
            "continuation seed requires 0 <= seed < 2**32, 1 <= steps < 2**24 and "
            f"0 <= index < {MAX_GENERATION_CONTINUATIONS}"
        )
    return (seed * 0x9E3779B1 + ((steps << 8) | index)) % 2**32


def rng_state_digest(rng: dict[str, object]) -> str:
    """sha256 of the island RNG state, CPU stream first then CUDA streams in device order.

    Framing: an 8-byte big-endian stream count, then per stream an 8-byte big-endian byte
    length followed by the raw uint8 state. Read from ``capture_rng()`` after
    ``torch.manual_seed`` and before the first draw, never computed from the seed integer.
    """
    streams = (cast(torch.Tensor, rng["cpu"]), *cast(tuple[torch.Tensor, ...], rng["cuda"]))
    digest = hashlib.sha256()
    digest.update(len(streams).to_bytes(8, "big"))
    for stream in streams:
        raw = bytes(stream.numpy().tobytes())
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def sampling_digest(digests: Sequence[str]) -> str:
    """sha256 over the per-continuation ``rng_digest`` strings in index order.

    Framing: an 8-byte big-endian count, then per digest an 8-byte big-endian length and
    its UTF-8 bytes.
    """
    outer = hashlib.sha256()
    outer.update(len(digests).to_bytes(8, "big"))
    for value in digests:
        raw = value.encode()
        outer.update(len(raw).to_bytes(8, "big"))
        outer.update(raw)
    return outer.hexdigest()


def _island(execution: SentenceExecution) -> AbstractContextManager[None]:
    """The RNG island: restores the caller's CPU (and device 0) state on exit."""
    return torch.random.fork_rng(devices=[0] if execution.device.type == "cuda" else [])


def _autocast(execution: SentenceExecution) -> torch.autocast:
    return torch.autocast(
        execution.device.type,
        dtype=torch.bfloat16,
        enabled=execution.profile.precision == "bf16",
    )


def _seed_island(profile: HProfile, index: int) -> dict[str, object]:
    """Seed the CURRENT island once for continuation ``index``; return its sampling law.

    Must run inside :func:`_island`. The digest is read from the freshly seeded RNG, so a
    writer that forgot ``manual_seed`` cannot reproduce what the reader re-derives.
    """
    seed = continuation_seed(profile.seed, profile.schedule.steps, index)
    torch.manual_seed(seed)
    return {
        "sampler": SAMPLER,
        "latent": LATENT,
        "seed": seed,
        "rng_digest": rng_state_digest(capture_rng()),
    }


def _derived_law(execution: SentenceExecution, index: int) -> dict[str, object]:
    with _island(execution):
        return _seed_island(execution.profile, index)


def _sampling_block(execution: SentenceExecution) -> dict[str, object]:
    count = execution.profile.schedule.generation_continuations
    laws = [_derived_law(execution, index) for index in range(count)]
    return {
        "sampler": SAMPLER,
        "temperature": TEMPERATURE,
        "continuations": count,
        "seed": execution.profile.seed,
        "rng_digest": sampling_digest([cast(str, law["rng_digest"]) for law in laws]),
    }


@dataclass(frozen=True)
class _Started:
    """One started continuation: its state, the island RNG it stopped at, its snapshot."""

    index: int
    state: SentenceRollout
    rng: object
    snapshot: dict[str, str]


def _trained(
    root: Path,
    training: ArtifactIdentity,
    profile: HProfile,
    source: Path,
    source_limit: int,
    artifact_limit: int,
    memo: SealMemo | None = None,
) -> tuple[SentenceExecution, dict[str, object]]:
    row = read_training_progress(
        root,
        training,
        profile,
        source,
        max_source_bytes=source_limit,
        max_artifact_bytes=artifact_limit,
        memo=memo,
    )
    if row["status"] != "training_complete":
        raise ValueError("generation requires complete training")
    with torch.random.fork_rng(devices=[0] if profile.device == "cuda" else []):
        execution = SentenceExecution(profile, source, max_source_bytes=source_limit)
        resume_seal(
            root,
            artifact_from_mapping(row["last_seal"]),
            execution,
            max_bytes=artifact_limit,
            memo=memo,
        )
        rng = capture_rng()
    if execution.step != profile.schedule.steps:
        # every continuation seed is keyed by schedule.steps, so a trace sealed anywhere
        # else would derive seeds the profile alone cannot reproduce
        raise ValueError("generation requires the sealed final step of the profile")
    execution.model.eval()
    return execution, rng


def _prompt(execution: SentenceExecution) -> tuple[tuple[bytes, ...], dict[str, object]]:
    record = next(r for r in execution.corpus.records if r.split == "validation")
    span = record.segmentation.spans[0]
    return (span.source,), {
        "document": span.document.identity,
        "span": [span.start, span.end, span.index],
    }


def _outcome(state: SentenceRollout, execution: SentenceExecution) -> str:
    return (
        state.status
        if state.status != "active"
        else "sentence_budget"
        if len(state.emissions) >= execution.profile.schedule.generation_sentences
        else "paused"
    )


def _entry(
    execution: SentenceExecution, started: _Started, law: dict[str, object]
) -> dict[str, object]:
    state = started.state
    last = state.observations[-1] if state.observations else None
    return {
        "index": started.index,
        "seed": law["seed"],
        "sampling_law": law,
        "sentences": [e.source.decode("utf-8") for e in state.emissions],
        "log_scores": list(state.log_scores),
        "outcome": _outcome(state, execution),
        # the continuation's last observation when it is a boundary token; never inferred
        # from text. A continuation that ended on a byte, or has no observation, is null.
        "boundary": last if last in (_SENTENCE_END, _DOCUMENT_END) else None,
        "snapshot": started.snapshot,
    }


def _status(
    entries: Sequence[dict[str, object]], execution: SentenceExecution
) -> Literal["paused", "generation_complete"]:
    """Paused iff a continuation is unfinished or fewer than N have started."""
    unfinished = (
        len(entries) < execution.profile.schedule.generation_continuations
        or entries[-1]["outcome"] == "paused"
    )
    return "paused" if unfinished else "generation_complete"


def _summary(
    execution: SentenceExecution, started: Sequence[_Started], entries: Sequence[dict[str, object]]
) -> dict[str, object]:
    """The aggregate over the started continuations, in index order.

    ``sentences``, ``observations`` and ``log_scores`` are concatenations. ``pending`` is the
    canonical pending bytes of the LAST started continuation (the only one that can hold an
    unfinished sentence). ``outcome`` is ``"paused"`` iff the receipt status is paused,
    otherwise the last continuation's own outcome.
    """
    return {
        "outcome": "paused" if _status(entries, execution) == "paused" else entries[-1]["outcome"],
        "sentences": [s for e in entries for s in cast(list[str], e["sentences"])],
        "observations": [o for s in started for o in s.state.observations],
        "pending": [execution.codec.canonical_byte(b) for b in started[-1].state.pending],
        "log_scores": [x for e in entries for x in cast(list[float], e["log_scores"])],
        "prompt": [source.decode("utf-8") for source in _prompt(execution)[0]],
    }


def _snapshot(root: Path, artifact: ArtifactIdentity, limit: int) -> dict[str, object]:
    ref = artifact_reference(root, artifact)
    _, data, digest = read_d1_artifact_bytes(
        root, ref["path"], label="generation snapshot", max_bytes=limit
    )
    if digest != ref["sha256"]:
        raise ValueError("generation snapshot hash mismatch")
    with ZipFile(BytesIO(data)) as archive:
        if sum(i.file_size for i in archive.infolist()) > limit:
            raise ValueError("expanded generation snapshot exceeds byte limit")
    return recovery_record(
        torch.load(BytesIO(data), map_location="cpu", weights_only=True),
        "schema run_identity training prompt rollout rng",
    )


def _refuse_old_receipt(schema: object) -> None:
    """Ruling D3: v1 and v2 receipts replay only against their own epoch's reader."""
    if schema in _OLD_RECEIPT_SCHEMAS:
        version = cast(str, schema).rsplit("-", 1)[1]
        raise ValueError(
            f"generation {version} requires its original reader; no implicit migration"
        )


def _load(
    root: Path,
    artifact: ArtifactIdentity,
    training: ArtifactIdentity,
    execution: SentenceExecution,
    limit: int,
) -> tuple[dict[str, object], list[_Started]]:
    raw = read_json(root, artifact, max_bytes=limit)
    _refuse_old_receipt(raw.get("schema"))
    receipt = recovery_record(raw, _RECEIPT_KEYS)
    reference = artifact_reference(root, training)
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["run_identity"] != execution.profile.identity
        or receipt["training"] != reference
        or receipt["scientific_authority"] != "none"
    ):
        raise ValueError("generation receipt belongs to another trained lineage")
    items = receipt["continuations"]
    count = execution.profile.schedule.generation_continuations
    if type(items) is not list or not 1 <= len(items) <= count:
        raise ValueError("generation continuations differ from the profile count")
    prompt = _prompt(execution)
    started: list[_Started] = []
    for position, item in enumerate(items):
        entry = recovery_record(item, _ENTRY_KEYS)
        if entry["index"] != position or type(entry["index"]) is not int:
            raise ValueError("generation continuation index differs from its position")
        snapshot = artifact_from_mapping(entry["snapshot"])
        saved = _snapshot(root, snapshot, limit)
        if (
            saved["schema"] != "h-generation-state-v1"
            or saved["run_identity"] != execution.profile.identity
            or saved["training"] != reference
            or saved["prompt"] != prompt[1]
        ):
            raise ValueError("generation snapshot binding differs from trained lineage")
        state = rollout_from_payload(saved["rollout"], execution.model, execution.codec)
        if state is None:
            raise ValueError("generation snapshot lacks feedback state")
        if state.history[: len(prompt[0])] != prompt[0]:
            raise ValueError("generation history does not contain the bound prompt")
        with _island(execution):
            restore_rng(saved["rng"])
        started.append(_Started(position, state, saved["rng"], artifact_reference(root, snapshot)))
    # Every recorded field is re-derived from the snapshots, the profile and the seeds.
    expected = [_entry(execution, s, _derived_law(execution, s.index)) for s in started]
    if json.dumps(items, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("generation continuations differ from observed feedback")
    if any(entry["outcome"] == "paused" for entry in expected[:-1]):
        raise ValueError("only the last started continuation can be unfinished")
    if receipt["snapshot"] != expected[-1]["snapshot"]:
        raise ValueError("generation snapshot reference differs from its last continuation")
    if json.dumps(receipt["summary"], sort_keys=True) != json.dumps(
        _summary(execution, started, expected), sort_keys=True
    ):
        raise ValueError("generation summary differs from observed feedback")
    if receipt["status"] != _status(expected, execution):
        raise ValueError("generation status differs from its observed outcome")
    if json.dumps(receipt["sampling"], sort_keys=True) != json.dumps(
        _sampling_block(execution), sort_keys=True
    ):
        raise ValueError("generation sampling law differs from the profile seeds")
    if receipt["parent_receipt"] is not None:
        parent = read_json(root, artifact_from_mapping(receipt["parent_receipt"]), max_bytes=limit)
        _refuse_old_receipt(parent.get("schema"))
        if (
            parent.get("schema") != RECEIPT_SCHEMA
            or parent.get("run_identity") != execution.profile.identity
            or parent.get("training") != reference
            or parent.get("status") != "paused"
        ):
            raise ValueError("generation parent does not belong to this resumable lineage")
        summary = recovery_record(parent.get("summary"), _SUMMARY_KEYS)
        observations = summary["observations"]
        stream = tuple(o for s in started for o in s.state.observations)
        if (
            type(observations) is not list
            or not len(observations) < len(stream)
            or tuple(observations) != stream[: len(observations)]
        ):
            raise ValueError("generation resume does not extend the previous observation prefix")
    return receipt, started


def read_generation(
    root: Path,
    artifact: ArtifactIdentity,
    training: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
    memo: SealMemo | None = None,
) -> dict[str, object]:
    execution, _ = _trained(
        root, training, profile, source_root, max_source_bytes, max_artifact_bytes, memo
    )
    return _load(root, artifact, training, execution, max_artifact_bytes)[0]


def _write_snapshot(
    root: Path,
    name: str,
    training: ArtifactIdentity,
    execution: SentenceExecution,
    state: SentenceRollout,
    rng: object,
    limit: int,
) -> ArtifactIdentity:
    stream = BytesIO()
    torch.save(
        {
            "schema": "h-generation-state-v1",
            "run_identity": execution.profile.identity,
            "training": artifact_reference(root, training),
            "prompt": _prompt(execution)[1],
            "rollout": rollout_to_payload(state, execution.model, execution.codec),
            "rng": rng,
        },
        stream,
    )
    data = stream.getvalue()
    if len(data) > limit:
        raise ValueError("generation snapshot exceeds byte limit")
    return publish_d1_atomic_bytes(root, name, data, label="generation state")


def run_probe_generation(
    root: Path,
    training: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
    resume: ArtifactIdentity | None = None,
    pause_after: int | None = None,
    memo: SealMemo | None = None,
) -> GenerationProgress:
    """Draw ``generation_continuations`` continuations of the prompt; pauses persist partial work.

    ``pause_after`` counts cumulative observations over the continuations in index order. A
    pause that lands exactly on a continuation's last observation still leaves the receipt
    paused (the next continuation starts fresh on resume). An unfinished continuation resumes
    by restoring its own saved island RNG, never by re-seeding; finished continuations'
    snapshots are carried forward by content address.

    ``memo``, when given, is the memo the SAME process's completion walk of this
    ``root`` populated (see :class:`~entropic_substrate.ladder.run.seals.SealMemo`);
    ``None`` keeps the memo-less full walk.
    """
    if pause_after is not None and (type(pause_after) is not int or pause_after < 1):
        raise ValueError("generation pause must be a positive observation count")
    execution, _ = _trained(
        root, training, profile, source_root, max_source_bytes, max_artifact_bytes, memo
    )
    count = profile.schedule.generation_continuations
    carried: list[_Started] = []
    if resume is not None:
        receipt, carried = _load(root, resume, training, execution, max_artifact_bytes)
        if receipt["status"] == "generation_complete":
            return GenerationProgress(
                "generation_complete", artifact_from_mapping(artifact_reference(root, resume))
            )
        recovered = sum(len(s.state.observations) for s in carried)
        if pause_after is not None and pause_after <= recovered:
            raise ValueError("generation pause must follow the recovered observation cursor")
    live = carried[-1] if carried and _outcome(carried[-1].state, execution) == "paused" else None
    done = carried if live is None else carried[:-1]
    with _autocast(execution):
        fresh = begin_rollout(execution.model, _prompt(execution)[0], execution.codec)
    started = time.perf_counter_ns()
    ticket = None
    failure: Exception | None = None
    state = fresh if live is None else live.state
    attempt_rng = capture_rng()
    produced: list[tuple[int, SentenceRollout, object, dict[str, object] | None]] = []
    try:
        ticket = None if execution.timer is None else execution.timer.start(ResourcePhase.EVAL)
        total = sum(len(s.state.observations) for s in done)
        index = len(done)
        paused = False
        while index < count and not paused:
            with _island(execution):
                law = None
                if live is not None and index == live.index:
                    restore_rng(live.rng)
                    state = live.state
                else:
                    law = _seed_island(profile, index)
                    state = fresh
                attempt_rng = capture_rng()
                while (
                    state.status == "active"
                    and len(state.emissions) < profile.schedule.generation_sentences
                ):
                    attempt_rng = capture_rng()
                    with _autocast(execution):
                        state = advance_rollout(execution.model, state, execution.codec)
                    if pause_after is not None and total + len(state.observations) >= pause_after:
                        paused = True
                        break
                island_rng = capture_rng()
            produced.append((index, state, island_rng, law))
            total += len(state.observations)
            index += 1
    except Exception as error:
        failure = error
    finally:
        try:
            cuda_seconds = (
                None
                if execution.timer is None or ticket is None
                else execution.timer.stop(ticket, ResourcePhase.EVAL)
            )
        except Exception as error:
            cuda_seconds = None
            failure = (
                error
                if failure is None
                else ExceptionGroup("generation and meter failures", [failure, error])
            )
    prefix = uuid4().hex
    if failure is not None:
        snapshot = _write_snapshot(
            root,
            prefix + "-generation-failure.pt",
            training,
            execution,
            state,
            attempt_rng,
            max_artifact_bytes,
        )
        write_json(
            root,
            prefix + "-generation-failure.json",
            {
                "schema": "h-generation-failure-v1",
                "run_identity": profile.identity,
                "scientific_authority": "none",
                "training": artifact_reference(root, training),
                "snapshot": artifact_reference(root, snapshot),
                "forensic_only": True,
                "status": "scientific_gap"
                if isinstance(failure, UndefinedDirection)
                else "execution_defect",
                "error_type": type(failure).__name__,
                "detail": str(failure),
            },
            max_bytes=max_artifact_bytes,
        )
        raise failure
    everything = list(done)
    laws: dict[int, dict[str, object]] = {s.index: _derived_law(execution, s.index) for s in done}
    for position, final, island_rng, law in produced:
        snapshot = _write_snapshot(
            root,
            f"{prefix}-generation-{position}.pt",
            training,
            execution,
            final,
            island_rng,
            max_artifact_bytes,
        )
        everything.append(_Started(position, final, island_rng, artifact_reference(root, snapshot)))
        laws[position] = law if law is not None else _derived_law(execution, position)
    entries = [_entry(execution, s, laws[s.index]) for s in everything]
    status = _status(entries, execution)
    receipt = write_json(
        root,
        prefix + "-generation.json",
        {
            "schema": RECEIPT_SCHEMA,
            "parent_receipt": None if resume is None else artifact_reference(root, resume),
            "run_identity": profile.identity,
            "scientific_authority": "none",
            "training": artifact_reference(root, training),
            "snapshot": entries[-1]["snapshot"],
            "status": status,
            "summary": _summary(execution, everything, entries),
            "wall_seconds": (time.perf_counter_ns() - started) / 1e9,
            "cuda_seconds": cuda_seconds,
            "continuations": entries,
            "sampling": _sampling_block(execution),
        },
        max_bytes=max_artifact_bytes,
    )
    return GenerationProgress(status, artifact_from_mapping(artifact_reference(root, receipt)))
