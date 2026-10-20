"""Checkpoint/journal identity seals with consecutive, source-bound ancestry."""

import json
import re
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch

from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.model import SentencePredictor
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.core.sentence.sampling import SamplingSpec, SentenceSampler
from entropic_substrate.kernel.artifacts import artifact_from_mapping
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.kernel.runtime.logging import logger
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
from entropic_substrate.ladder.run.worker_log_events import (
    ADMISSION_CALLERS,
    ADMISSION_END,
    ADMISSION_ROOTS,
    ADMISSION_START,
    render,
)


@dataclass(frozen=True)
class SealedStep:
    """A verified identity chain, not a scientific verdict or unchecked resume token."""

    phase: SentencePhase
    step: int
    checkpoint: ArtifactIdentity
    journal: ArtifactIdentity
    chain_length: int
    seal_hashes: tuple[str, ...] = ()


@dataclass
class SealMemo:
    """Seals ONE PROCESS'S admission of ONE artifact root has itself verified.

    RETRACTION (house rule 4). This docstring used to say the memo is "owned by
    exactly one invocation" of ``run_probe_training`` and that every
    "resume/pause/completion admission" passes ``memo=None``. That claim was
    measured against the 1K probe (2026-10-17, wave ``h0-plumbing-1k-v2``) and
    its consequence was wrong: the completion walk cost 880 s and the SAME walk
    was repeated by five later stages of the same worker process. What holds
    now: a memo is owned by one PROCESS and one ARTIFACT ROOT. The runner's
    completion or pause admission still starts from an EMPTY memo and walks
    every node (DG-B's admission predicate is unchanged); it returns that
    populated memo beside the receipt, and the later stages of the same worker
    process that read the same ``/run`` root reuse it. The capture root has its
    OWN memo, because this memo is keyed by digest alone and the capture holds
    byte-identical copies: a memo shared across roots would stop the seal
    readers from ever running on the captured bytes. A FRESH process (resume,
    host replay, ``composition/cli.py``) has no memo and still walks the full
    chain unconditionally.

    Keyed by the seal artifact's SHA-256 content digest alone -- the same
    content-addressed identity :func:`read_seal`'s own cycle detector already
    trusts (its ``seen`` set is digest-keyed, not path-keyed), because one seal
    artifact is legitimately referenced with different path SPELLINGS
    (absolute at the top-level caller, root-relative once round-tripped
    through a journal's ``parent`` field via
    :func:`~entropic_substrate.ladder.run.journal.artifact_reference`) while
    remaining exactly the same verified bytes. Never module-level state, never
    an ``lru_cache``, never shared across processes. ``remember`` is public
    (basedpyright strict forbids a cross-function protected member), but by
    convention it is called ONLY from :func:`read_seal` itself (and
    ``remember_evaluation`` ONLY from ``read_evaluation``), immediately
    after it has verified that exact node by reading it back from disk
    through its own per-node predicate; a writer's in-memory values (what
    ``seal_step`` just constructed) are never written here directly. A
    fresh, empty memo makes :func:`read_seal` walk the full chain exactly as
    with ``memo=None``.

    FOR-1670 (RETRACTION, house rule 4): this memo used to hold seals only, so
    every later admission re-read and re-replayed every evaluation record
    (roughly 50 s each, measured by the conductor on the 2026-10-19 H0
    plumbing wave). It now also holds the evaluation records
    :func:`~entropic_substrate.ladder.run.evaluation_records.read_evaluation`
    has admitted, keyed by the same content digest. The same convention
    applies: ``remember_evaluation`` is called ONLY from ``read_evaluation``,
    after the whole verification returned for a row it parsed from disk (never
    the writer's in-memory report). TRUST BOUNDARY: a hit does no filesystem
    access, so a file rewritten after admission but presented under the SAME
    identity is still served; detecting that belongs to a fresh memo or a fresh
    process (DG-B), not to the hit path. A hit returns the remembered object
    itself, so callers treat it as read-only.

    The counters are MEASURED, never derived: they advance only where the
    work happens (a node verified from disk, a journal read, an evaluation
    record admitted FROM DISK, wall time inside an outermost admission read),
    so the ``admission.*`` events report deltas of real disk reads rather than
    the chain length a memo hit would also have. ``evaluations_memo_hits``
    counts the evaluation records a hit served instead, beside (never inside)
    ``evaluations_read``.
    """

    _verified: dict[str, SealedStep] = field(default_factory=dict)
    _verified_evaluations: dict[str, dict[str, object]] = field(default_factory=dict, repr=False)
    seals_walked: int = 0
    journals_read: int = 0
    evaluations_read: int = 0
    evaluations_memo_hits: int = 0
    admission_seconds: float = 0.0
    _depth: int = field(default=0, init=False, repr=False)

    @property
    def populated(self) -> bool:
        return bool(self._verified)

    def get(self, artifact: ArtifactIdentity) -> SealedStep | None:
        return self._verified.get(artifact.content_hash.digest)

    def remember(self, artifact: ArtifactIdentity, seal: SealedStep) -> None:
        self._verified[artifact.content_hash.digest] = seal

    def get_evaluation(self, artifact: ArtifactIdentity) -> dict[str, object] | None:
        return self._verified_evaluations.get(artifact.content_hash.digest)

    def remember_evaluation(self, artifact: ArtifactIdentity, record: dict[str, object]) -> None:
        self._verified_evaluations[artifact.content_hash.digest] = record

    @contextmanager
    def timed_read(self) -> Generator[None]:
        """Add the wall time of an OUTERMOST admission read to ``admission_seconds``.

        Re-entrant reads (``read_seal`` inside ``read_training_progress``) add
        nothing, so the total is the time spent admitting, never the sum of
        nested overlaps and never the stage's own compute.
        """
        self._depth += 1
        started = time.perf_counter()
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                self.admission_seconds += time.perf_counter() - started


@contextmanager
def narrated_admission(caller: str, root: str, memo: SealMemo | None) -> Generator[None]:
    """Bracket ONE top-level admission site with ``admission.start``/``admission.end``.

    ``caller`` and ``root`` come from the closed sets in ``worker_log_events``
    (a path or an unknown name is a defect and raises). ``memo`` is read at
    entry: empty means ``fresh`` (the walk visits every node), populated means
    ``shared`` (it trusts an ancestry this process already admitted); it is
    never a per-site literal. ``seals_walked``, ``journals_read`` and
    ``evaluations`` are deltas of what happened inside the bracket, and
    ``seconds`` is the admission reads' own wall time, not the stage's compute.
    ``evaluations`` means evaluation records read from DISK; the records a memo
    hit served are reported separately as ``evaluations_memo_hits`` (log-only on
    ``.end``: the sink projection reads three counts and ignores this one).
    With no memo (a memo-less composition entry point) nothing is measured, so
    nothing is emitted. ``.end`` is logged only on a successful return; a raising
    walk propagates unchanged and logs no ``.end``, so a refused walk is never
    recorded as admitted.
    """
    if caller not in ADMISSION_CALLERS:
        raise ValueError("admission caller is outside the closed vocabulary")
    if root not in ADMISSION_ROOTS:
        raise ValueError("admission root is outside the closed vocabulary")
    if memo is None:
        yield
        return
    provenance = "shared" if memo.populated else "fresh"
    walked, journals = memo.seals_walked, memo.journals_read
    evaluations, seconds = memo.evaluations_read, memo.admission_seconds
    hits = memo.evaluations_memo_hits
    entered = {"caller": caller, "root": root, "memo": provenance}
    logger.bind(event=ADMISSION_START, **entered).info(render(ADMISSION_START, **entered))
    yield
    measured = {
        **entered,
        "seals_walked": memo.seals_walked - walked,
        "evaluations": memo.evaluations_read - evaluations,
        "evaluations_memo_hits": memo.evaluations_memo_hits - hits,
        "journals_read": memo.journals_read - journals,
        "seconds": memo.admission_seconds - seconds,
    }
    logger.bind(event=ADMISSION_END, **measured).info(render(ADMISSION_END, **measured))


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
    memo: SealMemo | None = None,
) -> SealedStep:
    """Validate every ancestor; full tensor admission still occurs when resuming.

    ``memo``, when given, is trusted ONLY for seal identities THIS SAME
    PROCESS has already verified for THIS artifact root (see
    :class:`SealMemo`). A fresh, empty (or absent) memo walks the FULL chain
    exactly as before -- this is what keeps every fresh process (resume, host
    replay, ``composition/cli.py``) and the runner's completion/pause walk an
    unconditional full walk. RETRACTION (house rule 4): this docstring used to
    say every "resume/pause/completion admission" passes ``memo=None``; the
    completion walk now passes an EMPTY memo (the same full walk, populating the
    memo the later stages of that process reuse). When the walk reaches a node
    already in the memo it stops there, trusts that ancestor's already-verified
    chain state instead of re-reading it, and every node it DOES read is
    verified through the identical per-node predicate as the full walk, then
    remembered. The source corpus is verified against ``profile.source_digest``
    on EVERY call, memo or not.
    """
    if memo is None:
        return _walk_seal(
            root,
            artifact,
            profile,
            source_root,
            max_bytes=max_bytes,
            max_source_bytes=max_source_bytes,
            memo=None,
        )
    with memo.timed_read():
        return _walk_seal(
            root,
            artifact,
            profile,
            source_root,
            max_bytes=max_bytes,
            max_source_bytes=max_source_bytes,
            memo=memo,
        )


def _walk_seal(
    root: Path,
    artifact: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_bytes: int,
    max_source_bytes: int,
    memo: SealMemo | None,
) -> SealedStep:
    corpus = read_prepared_corpus(
        source_root, ContentHash("sha256", profile.source_digest), max_bytes=max_source_bytes
    )
    nodes: list[tuple[SealedStep, int, ArtifactIdentity]] = []
    seen: set[str] = set()
    cursor: ArtifactIdentity | None = artifact
    base: SealedStep | None = None
    while cursor is not None:
        hit = None if memo is None else memo.get(cursor)
        if hit is not None:
            base = hit
            break
        if (
            cursor.content_hash.digest in seen
            or len(seen) >= profile.schedule.warmup_steps + profile.schedule.steps
        ):
            raise ValueError("seal chain is cyclic or exceeds its step budget")
        seen.add(cursor.content_hash.digest)
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
        if memo is not None:
            memo.journals_read += 1
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
        nodes.append((SealedStep(phase, step, checkpoint, journal, 0), start, cursor))
        if memo is not None:
            memo.seals_walked += 1
        cursor = None if measured["parent"] is None else artifact_from_mapping(measured["parent"])
    total_length = len(nodes) + (0 if base is None else base.chain_length)
    if total_length > profile.schedule.warmup_steps + profile.schedule.steps:
        raise ValueError("seal chain is cyclic or exceeds its step budget")
    previous = base
    final = base
    for node, start, node_artifact in reversed(nodes):
        _continuation(previous, node.phase, start, profile)
        chain_length = 1 + (0 if previous is None else previous.chain_length)
        seal_hashes = (node_artifact.content_hash.digest,) + (
            () if previous is None else previous.seal_hashes
        )
        sealed = SealedStep(
            node.phase, node.step, node.checkpoint, node.journal, chain_length, seal_hashes
        )
        if memo is not None:
            memo.remember(node_artifact, sealed)
        previous = sealed
        final = sealed
    if final is None:  # pragma: no cover - artifact is never None on entry
        raise ValueError("seal chain requires at least one node")
    return final


def seal_step(
    root: Path,
    prefix: str,
    engine: SentenceExecution,
    rows: tuple[dict[str, object], ...],
    *,
    parent: ArtifactIdentity | None,
    max_bytes: int,
    memo: SealMemo | None = None,
) -> ArtifactIdentity:
    """Publish a receipt last; partial writes are not restart seals and are retained.

    ``memo``, when given, is passed straight through to :func:`read_seal`'s
    lookup of ``parent`` -- see :class:`SealMemo`. A memo hit skips
    re-walking ``parent``'s own ancestry (L1); it never changes which
    continuation the new step is checked against.
    """
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
            memo=memo,
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
    root: Path,
    artifact: ArtifactIdentity,
    engine: SentenceExecution,
    *,
    max_bytes: int,
    memo: SealMemo | None = None,
) -> None:
    """Admit ``artifact`` and restore ``engine`` from it.

    ``memo`` is forwarded to :func:`read_seal` unchanged and defaults to the
    memo-less full walk, so a fresh-process resume stays an unconditional walk.
    A later stage of the SAME process passes the memo its root's admission
    walk populated; the checkpoint restore below is never skipped by a hit.
    """
    seal = read_seal(
        root,
        artifact,
        engine.profile,
        engine.source_root,
        max_bytes=max_bytes,
        max_source_bytes=engine.max_source_bytes,
        memo=memo,
    )
    engine.resume(
        root,
        artifact_reference(root, seal.checkpoint)["path"],
        seal.checkpoint.content_hash,
        phase=seal.phase,
        max_bytes=max_bytes,
        measurement=seal.journal.content_hash,
    )
