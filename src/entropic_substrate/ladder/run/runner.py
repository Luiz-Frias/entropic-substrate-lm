"""Recoverable engineering-probe training driver over the shared numerical engine."""

import re
import time
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, NoReturn, cast
from uuid import uuid4

from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import artifact_from_mapping, normalize_d1_artifact_root
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.run.admission import RUN_ADMISSION
from entropic_substrate.ladder.run.evaluation_records import read_evaluation
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import (
    artifact_reference,
    json_identity,
    read_json,
    write_json,
)
from entropic_substrate.ladder.run.run_authority import SCIENTIFIC_RUN_CLASSES, RunAuthority
from entropic_substrate.ladder.run.seals import (
    CorpusMemo,
    SealMemo,
    narrated_admission,
    read_seal,
    resume_seal,
    seal_step,
)
from entropic_substrate.ladder.run.training import SentencePhase
from entropic_substrate.ladder.run.worker_log_events import (
    CHECKPOINT_SEAL,
    EVALUATION_END,
    EVALUATION_START,
    PHASE_END,
    PHASE_START,
    render,
)

PROGRESS_SCHEMA_V1 = "h-probe-training-progress-v1"
PROGRESS_SCHEMA_V2 = "h-probe-training-progress-v2"
PROGRESS_FIELDS = (
    "schema run_identity run_class scientific_authority profile status phase step "
    "successful_optimizer_steps own_optimizer_steps inherited_warmup_steps block pair cell "
    "authority_digest handoff last_seal evaluations parent_receipt inventory"
)
_ID64 = re.compile(r"[0-9a-f]{64}")
_CELL_TOKEN = re.compile(r"[a-z0-9][a-z0-9-]*")


def refuse_progress_version(presented: object) -> NoReturn:
    """The one named refusal of a training receipt outside the v2 ladder.

    ``h-probe-training-progress-v1`` is retired: its receipts name no block, pair, cell or
    authority and cannot tell inherited warm-up steps from the cell's own. Every reader site
    (this module, ``trace_capture``, ``h_probe_collection``, ``h_probe_resume``) raises this text.
    """
    raise ValueError(
        f"training receipt schema {presented!r} is not admitted; the minimum supported "
        f"version is {PROGRESS_SCHEMA_V2}"
    )


@dataclass(frozen=True, slots=True)
class CellScope:
    """Which cell of which pair a training run is (all three, always together)."""

    block: str
    pair: str
    cell: str

    def __post_init__(self) -> None:
        if (
            any(type(v) is not str for v in (self.block, self.pair, self.cell))
            or _ID64.fullmatch(self.block) is None
            or _ID64.fullmatch(self.pair) is None
            or _CELL_TOKEN.fullmatch(self.cell) is None
        ):
            raise ValueError("a cell scope names a 64-hex block, a 64-hex pair and a safe cell")


@dataclass(frozen=True, slots=True)
class ReceiptBinding:
    """The binding fields a progress-v2 receipt carries; all ``None`` for a single smoke cell.

    ``block``, ``pair`` and ``cell`` are all set or all ``None`` (a pair cell or not), and
    ``authority_digest`` is ``None`` exactly for a smoke run class.
    """

    block: str | None = None
    pair: str | None = None
    cell: str | None = None
    authority_digest: str | None = None

    def __post_init__(self) -> None:
        scope = (self.block, self.pair, self.cell)
        if any(v is None for v in scope) and any(v is not None for v in scope):
            raise ValueError("a receipt binding names its block, pair and cell together or none")
        if self.block is not None:
            CellScope(self.block, cast(str, self.pair), cast(str, self.cell))
        if self.authority_digest is not None and (
            type(self.authority_digest) is not str or _ID64.fullmatch(self.authority_digest) is None
        ):
            raise ValueError("a receipt authority digest is 64 lowercase hex")

    @classmethod
    def of(cls, scope: CellScope | None, authority: RunAuthority | None) -> "ReceiptBinding":
        return cls(
            None if scope is None else scope.block,
            None if scope is None else scope.pair,
            None if scope is None else scope.cell,
            None if authority is None else authority.digest,
        )


@dataclass(frozen=True)
class TrainingProgress:
    """The receipt of one training invocation.

    ``admission_memo`` is a PROCESS-LOCAL hand-off, not part of the result: it
    is the memo the completion/pause admission walk populated (``None`` when
    that walk was not run in this invocation, e.g. an already-complete resume).
    It is excluded from ``==``, ``hash`` and ``repr``, so every consumer that
    compares or prints a progress sees exactly what it saw before, and it never
    enters a serialized artifact.
    """

    status: Literal["paused", "training_complete"]
    last_seal: ArtifactIdentity
    receipt: ArtifactIdentity
    admission_memo: SealMemo | None = field(default=None, compare=False, hash=False, repr=False)


def _progress(
    root: Path,
    status: Literal["paused", "training_complete"],
    seal: ArtifactIdentity,
    receipt: ArtifactIdentity,
    admission_memo: SealMemo | None = None,
) -> TrainingProgress:
    return TrainingProgress(
        status,
        artifact_from_mapping(artifact_reference(root, seal)),
        artifact_from_mapping(artifact_reference(root, receipt)),
        admission_memo,
    )


def write_progress_receipt(
    root: Path,
    name: str,
    profile: HProfile,
    profile_artifact: ArtifactIdentity,
    *,
    status: Literal["checkpoint", "paused", "training_complete"],
    phase: SentencePhase,
    step: int,
    last_seal: ArtifactIdentity,
    evaluations: list[dict[str, object]],
    parent_receipt: ArtifactIdentity | None,
    inventory: dict[str, object],
    max_bytes: int,
    binding: ReceiptBinding | None = None,
    handoff: ArtifactIdentity | None = None,
) -> ArtifactIdentity:
    """The ONE definition of an ``h-probe-training-progress-v2`` receipt.

    ``run_probe_training`` writes every checkpoint receipt through it, and so does the codec
    hand-off materializer (FOR-1692) for the receipt a recipient resumes from: a second copy of
    this mapping would let the two drift, and ``read_training_progress`` admits exactly this shape.

    FOR-1693 (h1-l4b): v2 adds ``block``, ``pair``, ``cell`` and ``authority_digest`` (from
    ``binding``; all ``None`` for a single smoke cell) and ``handoff`` (the recipient's OWN
    hand-off record, root-relative and flat; ``None`` for a donor or a single cell), and splits
    ``inherited_warmup_steps`` (the warm-up the recipient received: ``warmup_steps`` when
    ``handoff`` is given, else 0) from ``own_optimizer_steps``. ``successful_optimizer_steps``
    keeps the v1 arithmetic (the global axis: ``step`` plus the warm-up length in the prediction
    phase), and ``own + inherited == successful`` always. The hand-off receipt is own 0,
    inherited W, successful W.
    """
    bound = ReceiptBinding() if binding is None else binding
    successful = step + (profile.schedule.warmup_steps if phase == "prediction" else 0)
    inherited = profile.schedule.warmup_steps if handoff is not None else 0
    return write_json(
        root,
        name,
        {
            "schema": PROGRESS_SCHEMA_V2,
            "run_identity": profile.identity,
            "run_class": profile.run_class,
            "scientific_authority": "none",
            "profile": artifact_reference(root, profile_artifact),
            "status": status,
            "phase": phase,
            "step": step,
            "successful_optimizer_steps": successful,
            "own_optimizer_steps": successful - inherited,
            "inherited_warmup_steps": inherited,
            "block": bound.block,
            "pair": bound.pair,
            "cell": bound.cell,
            "authority_digest": bound.authority_digest,
            "handoff": None if handoff is None else artifact_reference(root, handoff),
            "last_seal": artifact_reference(root, last_seal),
            "evaluations": evaluations,
            "parent_receipt": None
            if parent_receipt is None
            else artifact_reference(root, parent_receipt),
            "inventory": inventory,
        },
        max_bytes=max_bytes,
    )


def read_training_progress(
    root: Path,
    artifact: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
    memo: SealMemo | None = None,
    expected: ReceiptBinding | None = None,
    corpora: CorpusMemo | None = None,
) -> dict[str, object]:
    """Admit resume/completion only against the exact source and sealed ancestry.

    ``corpora`` (FOR-1693, DH-41) is the process's corpus memo of ``source_root``: the memo
    ARGUMENT only (the corpus is replayed once per process instead of once per seal and
    evaluation).

    ``expected``, when given, is the binding the caller's request declares for this cell (block,
    pair, cell, authority digest); a receipt that names another is refused. Without it the
    in-chain binding is still verified: the receipt's own fields must be well formed, agree with
    its run class and its verified chain root, and every parent hop must carry the identical
    binding. FOR-1693 (progress v2): a ``h-probe-training-progress-v1`` receipt is refused by
    name (:func:`refuse_progress_version`).

    ``memo``, when given, accelerates the seal-chain (and evaluation-lineage)
    verification below exactly as :func:`~entropic_substrate.ladder.run.seals.read_seal`
    documents; it never changes what is admitted. Resume admission (a fresh
    process) calls this with ``memo=None`` and walks the full chain. The
    runner's pause/completion admission calls it with an EMPTY memo, which is
    the same full walk (DG-B's predicate is unchanged) and leaves that memo
    populated for the later stages of the same process. RETRACTION (house rule
    4): this docstring used to say pause and completion "always call this with
    ``memo=None``"; the per-process reading is the one ``SealMemo`` documents,
    and it was measured to cut five repeats of the 880 s completion walk.
    The mid-training, per-checkpoint call passes the invocation's memo.
    """
    if memo is None:
        return _read_training_progress(
            root,
            artifact,
            profile,
            source_root,
            max_source_bytes=max_source_bytes,
            max_artifact_bytes=max_artifact_bytes,
            memo=None,
            expected=expected,
            corpora=corpora,
        )
    with memo.timed_read():
        return _read_training_progress(
            root,
            artifact,
            profile,
            source_root,
            max_source_bytes=max_source_bytes,
            max_artifact_bytes=max_artifact_bytes,
            memo=memo,
            expected=expected,
            corpora=corpora,
        )


def _read_training_progress(
    root: Path,
    artifact: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
    memo: SealMemo | None,
    expected: ReceiptBinding | None = None,
    corpora: CorpusMemo | None = None,
) -> dict[str, object]:
    if not RUN_ADMISSION.admits_class(profile.run_class):
        raise ValueError("probe receipt has no scientific authority")
    payload = read_json(root, artifact, max_bytes=max_artifact_bytes)
    if payload.get("schema") == PROGRESS_SCHEMA_V1:
        refuse_progress_version(PROGRESS_SCHEMA_V1)
    row = recovery_record(payload, PROGRESS_FIELDS)
    if (
        row["schema"] != PROGRESS_SCHEMA_V2
        or row["run_identity"] != profile.identity
        or row["run_class"] != profile.run_class
        or row["scientific_authority"] != "none"
    ):
        raise ValueError("training receipt construction or authority mismatch")
    bound_profile = HProfile.from_manifest(
        read_json(root, artifact_from_mapping(row["profile"]), max_bytes=max_artifact_bytes)
    )
    if bound_profile != profile:
        raise ValueError("training receipt profile differs from requested profile")
    seal = read_seal(
        root,
        artifact_from_mapping(row["last_seal"]),
        profile,
        source_root,
        max_bytes=max_artifact_bytes,
        max_source_bytes=max_source_bytes,
        memo=memo,
        corpora=corpora,
    )
    total = seal.step + (profile.schedule.warmup_steps if seal.phase == "prediction" else 0)
    if (
        row["phase"] != seal.phase
        or type(row["step"]) is not int
        or row["step"] != seal.step
        or type(row["successful_optimizer_steps"]) is not int
        or row["successful_optimizer_steps"] != total
    ):
        raise ValueError("training receipt counters differ from sealed work")
    binding = ReceiptBinding(
        cast(str | None, row["block"]),
        cast(str | None, row["pair"]),
        cast(str | None, row["cell"]),
        cast(str | None, row["authority_digest"]),
    )
    if (binding.authority_digest is None) != (profile.run_class not in SCIENTIFIC_RUN_CLASSES):
        raise ValueError("training receipt authority digest disagrees with its run class")
    if expected is not None and binding != expected:
        raise ValueError("training receipt binding differs from the expected binding")
    inherited = profile.schedule.warmup_steps if seal.root_kind == "handoff" else 0
    if (
        type(row["inherited_warmup_steps"]) is not int
        or row["inherited_warmup_steps"] != inherited
        or type(row["own_optimizer_steps"]) is not int
        or row["own_optimizer_steps"] != total - inherited
    ):
        raise ValueError("training receipt inherited and own steps differ from its sealed chain")
    cited = row["handoff"]
    if inherited == 0:
        if cited is not None:
            raise ValueError("a receipt without a hand-off root cites no hand-off record")
    else:
        record = artifact_from_mapping(cited)
        if (
            PurePosixPath(record.path).name != record.path
            or record.content_hash.digest != seal.root_journal
        ):
            raise ValueError("training receipt hand-off record is not its chain root's record")
    if row["status"] not in ("checkpoint", "paused", "training_complete"):
        raise ValueError("unknown training receipt status")
    if row["status"] == "training_complete" and (
        seal.phase != "prediction" or seal.step != profile.schedule.steps
    ):
        raise ValueError("training completion lacks its full optimizer budget")
    evaluations = row["evaluations"]
    if type(evaluations) is not list:
        raise ValueError("evaluation roster must be an array")
    observed: list[int] = []
    for item in evaluations:
        entry = recovery_record(item, "step artifact")
        step = entry["step"]
        if type(step) is not int:
            raise ValueError("evaluation step must be integer")
        evaluation = read_evaluation(
            root,
            artifact_from_mapping(entry["artifact"]),
            profile,
            source_root,
            max_bytes=max_artifact_bytes,
            max_source_bytes=max_source_bytes,
            memo=memo,
            corpora=corpora,
        )
        point = artifact_from_mapping(evaluation.get("checkpoint_seal"))
        if point.content_hash.digest not in seal.seal_hashes:
            raise ValueError("evaluation belongs to another checkpoint lineage")
        point_state = read_seal(
            root,
            point,
            profile,
            source_root,
            max_bytes=max_artifact_bytes,
            max_source_bytes=max_source_bytes,
            memo=memo,
            corpora=corpora,
        )
        if (
            point_state.phase != "prediction"
            or point_state.step != step
            or evaluation.get("schema") != "h-predictive-evaluation-v1"
            or evaluation.get("run_identity") != profile.identity
            or type(evaluation.get("step")) is not int
            or evaluation["step"] != step
        ):
            raise ValueError("evaluation identity differs from its sealed checkpoint")
        observed.append(step)
    grid = (
        []
        if seal.phase == "codec_warmup"
        else [
            s
            for s in range(1, seal.step + 1)
            if s % profile.schedule.evaluation_interval == 0 or s == profile.schedule.steps
        ]
    )
    if observed != grid:
        raise ValueError("training receipt lacks the complete declared evaluation grid")
    if row["parent_receipt"] is not None:
        parent = read_json(
            root, artifact_from_mapping(row["parent_receipt"]), max_bytes=max_artifact_bytes
        )
        if parent.get("schema") == PROGRESS_SCHEMA_V1:
            refuse_progress_version(PROGRESS_SCHEMA_V1)
        if (
            parent.get("run_identity") != profile.identity
            or artifact_from_mapping(parent.get("last_seal")).content_hash.digest
            not in seal.seal_hashes
        ):
            raise ValueError("resume receipt belongs to another lineage")
        if any(
            parent.get(key) != row[key]
            for key in ("block", "pair", "cell", "authority_digest", "handoff")
        ):
            raise ValueError("resume receipt carries another pair, cell or authority binding")
    return row


def run_probe_training(
    profile: HProfile,
    source_root: Path,
    output_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
    resume: ArtifactIdentity | None = None,
    stop_after_seal: Callable[[str, int], bool] | None = None,
    on_checkpoint: Callable[[ArtifactIdentity], None] | None = None,
    sink: MetricSink | None = None,
    publication_timing: MutableMapping[str, float] | None = None,
    authority: RunAuthority | None = None,
    scope: CellScope | None = None,
    corpora: CorpusMemo | None = None,
) -> TrainingProgress:
    """Run all declared training steps; this result is not a full flywheel verdict.

    ``scope`` (FOR-1693) names the pair cell this run is; every receipt it writes binds it
    beside the authority's digest (progress v2). A resume must present a receipt carrying the
    identical binding, and a recipient resumed from its hand-off receipt keeps citing the same
    hand-off record.

    ``corpora`` (FOR-1693, DH-41) is the process's corpus memo of ``source_root``: the memo
    ARGUMENT only. Every engine, seal walk, journal and evaluation read this run performs shares
    it, so the prepared corpus is replayed once per process instead of on every call.

    Checkpoint/evaluation failures retain partial evidence and propagate. A
    stop callback is observed only after a durable seal and evaluation receipt.
    Scientific adapters and successor remote probes require separate authority.
    A ``pilot``/``confirmatory`` profile runs only under ``authority``, the sealed
    ``RunAuthority`` its request carries (rung B, operator ruling 2026-10-27); a smoke
    profile runs only without one.

    ``sink``, when given, observes every committed training-step row exactly
    as :mod:`entropic_substrate.ladder.run.journal` receives it -- an
    ADDITIVE observability mirror (RUNG 2b), never a substitute for the D1
    journal's authoritative, content-addressed record. This function never
    constructs a sink and never calls ``sink.finish()``; the caller that
    built it owns its lifecycle.

    ``publication_timing``, when given, is overwritten with this checkpoint's
    ``l1_seconds``/``l2_seconds`` immediately BEFORE ``on_checkpoint`` fires,
    so the caller's callback (``object_worker``'s ``checkpoint`` closure) can
    read it synchronously for that same event -- an ADDITIVE instrumentation
    mirror, never part of the training or admission contract. Internally,
    one :class:`~entropic_substrate.ladder.run.seals.SealMemo` is created for
    this WHOLE invocation and threaded to every ``seal_step`` call and to
    the mid-training ``read_training_progress``/``read_evaluation`` calls.
    It is never given to resume admission (a fresh process) or to the
    completion/pause admission: that walk starts from its OWN empty memo, so
    it still verifies every node, and the populated memo it leaves is returned
    as ``TrainingProgress.admission_memo`` for the later stages of the same
    process (FOR-1662). RETRACTION (house rule 4): the previous text said the
    completion/pause admission "always re-walks the full chain" because it
    received no memo; it still walks the full chain, and the repeats of that
    walk by later stages are what was removed.
    """
    refusal = RUN_ADMISSION.authority_refusal(profile.run_class, profile.stage, authority)
    if refusal is not None:
        raise ValueError(refusal)
    surface = RUN_ADMISSION.worker_surface(profile.run_class)
    if not RUN_ADMISSION.admits(profile.run_class, profile.stage, surface, authority):
        raise ValueError("probe driver lacks scientific or successor authority")
    root = normalize_d1_artifact_root(output_root)
    binding = ReceiptBinding.of(scope, authority)
    handoff: ArtifactIdentity | None = None
    evaluations: list[dict[str, object]] = []
    previous_receipt = resume
    last_seal = None
    if resume is None:
        if any(root.iterdir()):
            raise ValueError("fresh probe output must be empty")
        profile_artifact = write_json(
            root, "profile.json", profile.manifest, max_bytes=max_artifact_bytes
        )
    else:
        # PR #430 round 1 (DH-47): a refused resume admission is a training-phase execution
        # stop and writes the failure record below like any other, so the pair's derivation
        # confirms it from disk. The record cites the profile by the identity every writer of a
        # run root's ``profile.json`` produces (write_json of ``profile.manifest``), bound before
        # the admission so the handler never reads an unbound name or the disk that may be what
        # broke; the admitted receipt's own reference replaces it once admission holds.
        profile_artifact = json_identity(
            root, "profile.json", profile.manifest, max_bytes=max_artifact_bytes
        )
    execution: SentenceExecution | None = None
    prefix = uuid4().hex
    pending: list[dict[str, object]] = []
    memo = SealMemo()
    # FOR-1637: the phase being narrated, its first step and its wall clock.
    # Emission is O(phases + seals + evaluations), never per step (C7).
    narrated_phase: str | None = None
    phase_from = 0
    phase_clock = 0.0
    try:
        if resume is not None:
            previous = read_training_progress(
                root,
                resume,
                profile,
                source_root,
                max_source_bytes=max_source_bytes,
                max_artifact_bytes=max_artifact_bytes,
                expected=binding,
                corpora=corpora,
            )
            last_seal = artifact_from_mapping(previous["last_seal"])
            if previous["handoff"] is not None:
                handoff = artifact_from_mapping(previous["handoff"])
            if previous["status"] == "training_complete":
                return _progress(root, "training_complete", last_seal, resume)
            profile_artifact = artifact_from_mapping(previous["profile"])
            evaluations = cast(list[dict[str, object]], previous["evaluations"]).copy()
        execution = SentenceExecution(
            profile, source_root, max_source_bytes=max_source_bytes, corpora=corpora
        )
        if last_seal is not None:
            resume_seal(root, last_seal, execution, max_bytes=max_artifact_bytes)
        while True:
            if (
                execution.phase == "codec_warmup"
                and execution.step == profile.schedule.warmup_steps
            ):
                if narrated_phase == "codec_warmup":
                    phase_seconds = time.perf_counter() - phase_clock
                    ended = {
                        "phase": "codec_warmup",
                        "step_from": phase_from,
                        "step_to": execution.step,
                        "seconds": phase_seconds,
                    }
                    logger.bind(event=PHASE_END, **ended).info(render(PHASE_END, **ended))
                    narrated_phase = None
                execution.begin_prediction()
            if narrated_phase != execution.phase:
                narrated_phase = execution.phase
                phase_from = execution.step
                phase_clock = time.perf_counter()
                started = {
                    "phase": execution.phase,
                    "step_from": phase_from,
                    "step_to": profile.schedule.warmup_steps
                    if execution.phase == "codec_warmup"
                    else profile.schedule.steps,
                }
                logger.bind(event=PHASE_START, **started).info(render(PHASE_START, **started))
            step_row = execution.advance()
            pending.append(step_row)
            if sink is not None:
                sink.log(step_row)
            final = execution.step == (
                profile.schedule.warmup_steps
                if execution.phase == "codec_warmup"
                else profile.schedule.steps
            )
            evaluate = execution.phase == "prediction" and (
                final or execution.step % profile.schedule.evaluation_interval == 0
            )
            if not (
                final or evaluate or execution.step % profile.schedule.checkpoint_interval == 0
            ):
                continue
            seal_clock = time.perf_counter()
            l1_started = None if publication_timing is None else time.perf_counter()
            last_seal = seal_step(
                root,
                prefix,
                execution,
                tuple(pending),
                parent=last_seal,
                max_bytes=max_artifact_bytes,
                memo=memo,
            )
            if publication_timing is not None and l1_started is not None:
                publication_timing["l1_seconds"] = time.perf_counter() - l1_started
            # AFTER the l1_seconds assignment: no emission inside the L1 window
            # (FOR-1567 timing comparability).
            sealed = {
                "phase": execution.phase,
                "step": execution.step,
                "global_step": execution.step
                + (profile.schedule.warmup_steps if execution.phase == "prediction" else 0),
                "seconds": time.perf_counter() - seal_clock,
            }
            logger.bind(event=CHECKPOINT_SEAL, **sealed).info(render(CHECKPOINT_SEAL, **sealed))
            pending.clear()
            if evaluate:
                evaluation_global_step = execution.step + profile.schedule.warmup_steps
                evaluation_clock = time.perf_counter()
                logger.bind(event=EVALUATION_START, global_step=evaluation_global_step).info(
                    render(EVALUATION_START, global_step=evaluation_global_step)
                )
                report = execution.evaluate()
                report["checkpoint_seal"] = artifact_reference(root, last_seal)
                evaluation = write_json(
                    root,
                    f"{prefix}-{execution.step:08d}-evaluation.json",
                    report,
                    max_bytes=max_artifact_bytes,
                )
                read_evaluation(
                    root,
                    evaluation,
                    profile,
                    source_root,
                    max_bytes=max_artifact_bytes,
                    max_source_bytes=max_source_bytes,
                    memo=memo,
                    corpora=corpora,
                )
                evaluation_seconds = time.perf_counter() - evaluation_clock
                logger.bind(
                    event=EVALUATION_END,
                    global_step=evaluation_global_step,
                    seconds=evaluation_seconds,
                ).info(
                    render(
                        EVALUATION_END,
                        global_step=evaluation_global_step,
                        seconds=evaluation_seconds,
                    )
                )
                if sink is not None:
                    # EXIT-GATE LANE (2026-09-26): the sealed, reader-admitted
                    # evaluation report reaches the additive mirror too, after
                    # its receipt and never before; the journal and the
                    # evaluation artifact above are the authoritative record.
                    sink.log(report)
                evaluations.append(
                    {"step": execution.step, "artifact": artifact_reference(root, evaluation)}
                )
            stopped = (
                False
                if stop_after_seal is None
                else stop_after_seal(execution.phase, execution.step)
            )
            if type(stopped) is not bool:
                raise ValueError("stop callback must return a boolean")
            complete = execution.phase == "prediction" and final
            status = "training_complete" if complete else "paused" if stopped else "checkpoint"
            receipt = write_progress_receipt(
                root,
                f"{prefix}-{execution.phase}-{execution.step:08d}-progress.json",
                profile,
                profile_artifact,
                status=status,
                phase=execution.phase,
                step=execution.step,
                last_seal=last_seal,
                evaluations=evaluations,
                parent_receipt=previous_receipt,
                inventory=execution.inventory(),
                max_bytes=max_artifact_bytes,
                binding=binding,
                handoff=handoff,
            )
            previous_receipt = receipt
            completion: SealMemo | None = None
            if complete or stopped or on_checkpoint is not None:
                # DG-B/C12 (per-PROCESS reading, FOR-1662): a genuine completion
                # or pause is the admission predicate for this process, so it
                # starts from an EMPTY memo, constructed here and never the
                # invocation memo, and visits every node. Earlier text here
                # said it "always re-walks the full chain" with memo=None; it
                # still does, and the populated memo is handed on so the later
                # stages do not repeat the walk. Only the mid-training,
                # per-checkpoint call trusts the invocation memo.
                completion = SealMemo() if (complete or stopped) else None
                # The admission pair stays OUTSIDE the l2 window (the l1 rule).
                with narrated_admission("completion", "run", completion):
                    l2_started = None if publication_timing is None else time.perf_counter()
                    read_training_progress(
                        root,
                        receipt,
                        profile,
                        source_root,
                        max_source_bytes=max_source_bytes,
                        max_artifact_bytes=max_artifact_bytes,
                        memo=memo if completion is None else completion,
                        expected=binding,
                        corpora=corpora,
                    )
                    if publication_timing is not None and l2_started is not None:
                        publication_timing["l2_seconds"] = time.perf_counter() - l2_started
            if on_checkpoint is not None:
                on_checkpoint(artifact_from_mapping(artifact_reference(root, receipt)))
            if complete or stopped:
                phase_seconds = time.perf_counter() - phase_clock
                ended = {
                    "phase": execution.phase,
                    "step_from": phase_from,
                    "step_to": execution.step,
                    "seconds": phase_seconds,
                }
                logger.bind(event=PHASE_END, **ended).info(render(PHASE_END, **ended))
                return _progress(
                    root,
                    "training_complete" if complete else "paused",
                    last_seal,
                    receipt,
                    completion,
                )
    except Exception as error:
        write_json(
            root,
            f"{prefix}-failure.json",
            {
                "schema": "h-probe-training-failure-v2",
                "run_identity": profile.identity,
                "run_class": profile.run_class,
                "scientific_authority": "none",
                "profile": artifact_reference(root, profile_artifact),
                "status": "scientific_gap"
                if isinstance(error, UndefinedDirection)
                else "execution_defect",
                "error_type": type(error).__name__,
                "error_code": getattr(error, "code", type(error).__name__),
                "phase": "initialization" if execution is None else execution.phase,
                "step": 0 if execution is None else execution.step,
                "error": str(error),
                "last_attempt": None if execution is None else execution.last_attempt,
                "last_evaluation": None if execution is None else execution.last_evaluation,
                "unsealed_rows": pending,
                "last_seal": None if last_seal is None else artifact_reference(root, last_seal),
                "last_receipt": None
                if previous_receipt is None
                else artifact_reference(root, previous_receipt),
            },
            max_bytes=max_artifact_bytes,
        )
        raise
