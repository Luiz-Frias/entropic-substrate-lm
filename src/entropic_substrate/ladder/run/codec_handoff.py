"""Hand the warmed codec from a donor cell to its pair recipient through a hand-off seal.

FOR-1692 (h1-l4a). Two H1 cells that differ only in the spine carrier share the codec warm-up
by construction: warm-up trains the codec roles only (``training.sentence_roles``), the
objective calls only ``codec.reconstruct``, the law reads ``law.mode`` and never
``spine.carrier``, and prediction builds a fresh optimizer and sampler at ``begin_prediction``.
A block that warms a sharing pair's members independently is excluded as a defect, so the
recipient's codec must ARRIVE, not be re-derived.

This member reads sealed bytes only. ``materialize_codec_handoff`` admits the donor's
warm-up-final seal, builds the recipient engine, copies the donor's codec tensors and warm-up
AdamW state into it, restores the donor's RNG state, and publishes four artifacts in an empty
recipient root: the hand-off record (``h-codec-handoff-v1``), the recipient's OWN checkpoint
(measurement = the record's digest), the hand-off seal (``h-codec-handoff-seal-v1``, the
recipient's chain root) and a progress receipt a fresh ``run_probe_training(resume=...)``
continues from. ``admit_codec_pair`` is the pure reader of the pair proof. It restores both
checkpoints into a ``SentenceExecution`` built on ``profile.device``, so it runs only where
that device topology exists: the pod, for every class but ``local_tiny``. A host-side verifier
for cuda pairs belongs to L4c, and must be proven equal to this one on
identical checkpoint bytes before anything relies on it.

Layering. ``seals`` admits the hand-off seal as a chain root and owns the record's key set;
this module imports ``seals``, never the reverse. It does not import ``ladder.register`` or
``ladder.analyze`` (tests/test_h_codec_handoff.py pins that by AST): the block-exclusion
vocabulary is the caller's, and ``CodecHandoffDefect.code == "independent_warmup"`` equals
``H1ExclusionReason.INDEPENDENT_WARMUP.value`` by a test, not by an import. Importing
``h_preservation`` here would put the registration layer into this reader's member identity.

No random number is drawn by the copy itself. Constructing an engine seeds and draws, so every
construction finishes first and the donor's RNG state, captured right after the donor
checkpoint was admitted, is restored LAST, immediately before the recipient checkpoint reads
it. Both public functions run inside ``torch.random.fork_rng`` so the caller's RNG is untouched.
"""

import copy
import json
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

import torch

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import (
    capture_rng,
    recovery_record,
    restore_rng,
)
from entropic_substrate.kernel.artifacts import artifact_from_mapping, normalize_d1_artifact_root
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.run.checkpoint import checkpoint_binding, save_checkpoint
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import artifact_reference, read_json, write_json
from entropic_substrate.ladder.run.runner import ReceiptBinding, write_progress_receipt
from entropic_substrate.ladder.run.seals import (
    HANDOFF_RECORD_SCHEMA,
    HANDOFF_SEAL_SCHEMA,
    SealedStep,
    check_handoff_record,
    read_seal,
)

_SEAL_FIELDS = "schema run_identity phase step checkpoint journal"
_PREFIX = "handoff"


class CodecHandoffDefect(ValueError):
    """A refusal of the hand-off, from a closed set of six names.

    ``independent_warmup`` is STRUCTURAL: the recipient's chain starts with an ordinary
    warm-up step-one seal, so the recipient trained its own codec. It is detected from the
    verified chain root, never from missing telemetry rows, file names or tensor inequality
    (an independent warm-up on CPU is bit-identical to the donor's, so tensors cannot tell).
    The caller maps the name onto the block-exclusion vocabulary it owns.
    """

    CODES: ClassVar[frozenset[str]] = frozenset(
        {
            "independent_warmup",
            "foreign_donor",
            "pair_predicate",
            "chained_handoff",
            "construction_divergence",
            "payload_mismatch",
        }
    )

    def __init__(self, code: str, detail: str) -> None:
        if code not in self.CODES:
            raise ValueError("unknown codec hand-off defect code")
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class VerifiedHandoff:
    """What ``admit_codec_pair`` proved; the single source of the analysis reader's evidence.

    The field names equal ``h_aggregate.CodecHandoffEvidence``'s (pinned by a test): the
    analysis reader cannot import this member, so the caller converts one into the other.
    """

    recipient_identity: str
    donor_identity: str
    warmup_steps: int
    record_digest: str
    donor_seal_digest: str


@dataclass(frozen=True)
class CodecHandoff:
    """The artifacts ``materialize_codec_handoff`` published (root-relative) and its proof."""

    record: ArtifactIdentity
    checkpoint: ArtifactIdentity
    seal: ArtifactIdentity
    receipt: ArtifactIdentity
    verified: VerifiedHandoff


# --------------------------------------------------------------------------- pair predicate
def _pair_shape(profile: HProfile) -> str:
    manifest = profile.manifest
    del manifest["name"]
    spine = manifest["spine"]
    if not isinstance(spine, dict):  # pragma: no cover - the manifest always nests the spine
        raise ValueError("profile manifest lacks its spine section")
    del spine["carrier"]  # pyright: ignore[reportUnknownMemberType]
    return json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":"))


def require_pair(donor: HProfile, recipient: HProfile) -> None:
    """Manifests equal except ``name`` and ``spine.carrier`` (so the same ``law.mode``), and
    the carriers differ. This is the comparator that refuses; it is not ``h_preservation``'s."""
    if donor.spine.carrier == recipient.spine.carrier:
        raise CodecHandoffDefect("pair_predicate", "a pair needs two different spine carriers")
    if _pair_shape(donor) != _pair_shape(recipient):
        raise CodecHandoffDefect(
            "pair_predicate",
            "donor and recipient manifests differ beyond their name and spine carrier",
        )


# ----------------------------------------------------------------- canonical payload digests
def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


class _Stream:
    """Length-prefixed digest stream: no field boundary is ambiguous, no byte is torch.save's."""

    def __init__(self) -> None:
        self._digest = sha256()

    def feed(self, tag: str, data: bytes) -> None:
        label = tag.encode()
        self._digest.update(len(label).to_bytes(4, "little") + label)
        self._digest.update(len(data).to_bytes(8, "little") + data)

    def value(self, name: str, value: object) -> None:
        if isinstance(value, torch.Tensor):
            if sys.byteorder != "little":  # pragma: no cover - every supported host
                raise RuntimeError("canonical tensor encoding is little-endian")
            self.feed(f"{name}:tensor", f"{value.dtype}:{list(value.shape)}".encode())
            flat = value.detach().cpu().contiguous().reshape(-1)
            self.feed(f"{name}:bytes", flat.view(torch.uint8).numpy().tobytes())
        else:
            self.feed(f"{name}:json", _canonical(value).encode())

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def codec_digest(engine: SentenceExecution) -> str:
    """Every ``codec.*`` key of the full ``state_dict`` (parameters and buffers), by name."""
    stream = _Stream()
    state = engine.model.state_dict()
    for name in sorted(k for k in state if k.startswith("codec.")):
        stream.value(name, state[name])
    return stream.hexdigest()


def optimizer_digest(engine: SentenceExecution) -> str:
    """AdamW state keyed by PARAMETER NAME (its own state is index-keyed), plus its groups."""
    names = {id(p): n for n, p in engine.model.named_parameters()}
    ordered = [names[id(p)] for g in engine.optimizer.param_groups for p in g["params"]]
    saved = engine.optimizer.state_dict()
    stream = _Stream()
    stream.feed("groups", _canonical(ordered).encode())
    hyper = [{k: v for k, v in g.items() if k != "params"} for g in saved["param_groups"]]
    stream.feed("hyperparameters", _canonical(hyper).encode())
    for index, name in enumerate(ordered):
        state = saved["state"].get(index, {})
        for key in sorted(state):
            stream.value(f"{name}/{key}", state[key])
    return stream.hexdigest()


def rng_digest(payload: object) -> str:
    """The CPU state and every CUDA state of a ``capture_rng`` payload, in order."""
    row = recovery_record(payload, "schema cpu cuda")
    cuda = row["cuda"]
    if not isinstance(cuda, tuple):
        raise ValueError("RNG payload must carry a tuple of CUDA states")
    stream = _Stream()
    stream.value("cpu", row["cpu"])
    for index, state in enumerate(cuda):  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        stream.value(f"cuda:{index}", state)
    return stream.hexdigest()


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and bool(torch.equal(left, right))
        )
    return not isinstance(right, torch.Tensor) and _canonical(left) == _canonical(right)


# ------------------------------------------------------------------------------ the guards
def _without_cell_axes(engine: SentenceExecution) -> dict[str, object]:
    binding = json.loads(
        checkpoint_binding(
            engine.model, engine.optimizer, engine.sampler, engine.phase, engine.profile.identity
        )
    )
    del binding["run_identity"]
    del binding["spine"]["carrier"]
    return binding


def guard_construction(donor: SentenceExecution, recipient: SentenceExecution) -> None:
    """Every state key is either COPIED (``codec.*``) or GUARDED here; there is no third class.

    The warm-up never trains or normalizes the spine or the future law, so each non-codec
    tensor of the donor's warmed checkpoint must equal the recipient's fresh construction
    bit for bit (``torch.equal`` after dtype and shape, never ``allclose``). AdamW state is
    index-keyed, so it is meaningful only when both bindings agree on everything but the run
    identity and the carrier (the parameter groups first among them).
    """
    if _without_cell_axes(donor) != _without_cell_axes(recipient):
        raise CodecHandoffDefect(
            "construction_divergence",
            "donor and recipient bindings differ beyond the run identity and spine carrier",
        )
    theirs, ours = donor.model.state_dict(), recipient.model.state_dict()
    if theirs.keys() != ours.keys():
        raise CodecHandoffDefect("construction_divergence", "model inventories differ")
    for name in sorted(k for k in ours if not k.startswith("codec.")):
        if not _same_value(theirs[name], ours[name]):
            raise CodecHandoffDefect(
                "construction_divergence", f"non-codec state {name} differs from the donor's"
            )


def _adopt_codec(recipient: SentenceExecution, donor: SentenceExecution) -> None:
    state = copy.deepcopy(donor.model.codec.state_dict())
    recipient.model.codec.load_state_dict(state, strict=True)


def _adopt_optimizer(recipient: SentenceExecution, donor: SentenceExecution) -> None:
    recipient.optimizer.load_state_dict(copy.deepcopy(donor.optimizer.state_dict()))


def _devices(profile: HProfile) -> list[int]:
    return [0] if profile.device == "cuda" else []


# ------------------------------------------------------------------------- reading the chains
def _read_donor_seal(
    root: Path,
    seal: ArtifactIdentity,
    donor: HProfile,
    source_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
) -> SealedStep:
    """The donor's seal, fully walked; a hand-off recipient is never a donor."""
    sealed = read_seal(
        root,
        seal,
        donor,
        source_root,
        max_bytes=max_artifact_bytes,
        max_source_bytes=max_source_bytes,
    )
    if sealed.root_kind == "handoff":
        raise CodecHandoffDefect(
            "chained_handoff",
            "the donor's own codec arrived by hand-off; a donor warms up for itself",
        )
    return sealed


def _is_warmup_final(sealed: SealedStep, profile: HProfile) -> bool:
    return sealed.phase == "codec_warmup" and sealed.step == profile.schedule.warmup_steps


def _walk(
    root: Path, cited: ArtifactIdentity, profile: HProfile, max_bytes: int
) -> list[tuple[ArtifactIdentity, dict[str, object]]]:
    """Raw seal rows from ``cited`` toward the chain root, newest first.

    UNVERIFIED by itself (no journal replay, no checkpoint): the caller reads whichever node
    it takes from here through ``read_seal``. A hand-off row carries no journal parent.
    """
    rows: list[tuple[ArtifactIdentity, dict[str, object]]] = []
    cursor: ArtifactIdentity | None = cited
    while cursor is not None and len(rows) < profile.schedule.warmup_steps + profile.schedule.steps:
        row = recovery_record(read_json(root, cursor, max_bytes=max_bytes), _SEAL_FIELDS)
        rows.append((cursor, row))
        if row["schema"] == HANDOFF_SEAL_SCHEMA:
            break
        journal = read_json(root, artifact_from_mapping(row["journal"]), max_bytes=max_bytes)
        parent = journal.get("parent")
        cursor = None if parent is None else artifact_from_mapping(parent)
    return rows


def _resolve_warmup_final(
    root: Path, cited: ArtifactIdentity, profile: HProfile, max_bytes: int
) -> ArtifactIdentity:
    for node, row in _walk(root, cited, profile, max_bytes):
        if row["phase"] == "codec_warmup" and row["step"] == profile.schedule.warmup_steps:
            return node
    raise CodecHandoffDefect("foreign_donor", "the donor's chain holds no warm-up-final seal")


def _restore(
    profile: HProfile,
    source_root: Path,
    root: Path,
    sealed: SealedStep,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
) -> SentenceExecution:
    """A fresh engine with ``sealed``'s checkpoint admitted (this also restores the RNG)."""
    engine = SentenceExecution(profile, source_root, max_source_bytes=max_source_bytes)
    engine.resume(
        root,
        artifact_reference(root, sealed.checkpoint)["path"],
        sealed.checkpoint.content_hash,
        phase=sealed.phase,
        max_bytes=max_artifact_bytes,
        measurement=sealed.journal.content_hash,
    )
    return engine


# ----------------------------------------------------------------------------- the reader
def admit_codec_pair(
    donor_root: Path,
    donor_seal: ArtifactIdentity,
    donor_profile: HProfile,
    recipient_root: Path,
    recipient_seal: ArtifactIdentity,
    recipient_profile: HProfile,
    source_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
) -> VerifiedHandoff:
    """Prove that the recipient's codec is the donor's warm-up, from sealed bytes alone.

    Pure reader: writes nothing and leaves the caller's RNG untouched (checkpoint admission
    restores the RNG as a side effect, so it runs inside ``fork_rng``). Each root is walked
    fresh, with no shared memo. ``donor_seal`` is the donor's warm-up-final seal or any later
    seal of its chain; the donor's completion is never required, so a donor that failed after
    its warm-up still lets the recipient be admitted.
    """
    require_pair(donor_profile, recipient_profile)
    limits = {"max_source_bytes": max_source_bytes, "max_artifact_bytes": max_artifact_bytes}
    with torch.random.fork_rng(devices=_devices(recipient_profile)):
        recipient = read_seal(
            recipient_root,
            recipient_seal,
            recipient_profile,
            source_root,
            max_bytes=max_artifact_bytes,
            max_source_bytes=max_source_bytes,
        )
        if recipient.root_kind == "initial":
            raise CodecHandoffDefect(
                "independent_warmup",
                "the recipient's chain starts with an ordinary warm-up step-one seal",
            )
        root_ref, _ = _walk(recipient_root, recipient_seal, recipient_profile, max_artifact_bytes)[
            -1
        ]
        handoff = read_seal(
            recipient_root,
            root_ref,
            recipient_profile,
            source_root,
            max_bytes=max_artifact_bytes,
            max_source_bytes=max_source_bytes,
        )
        if handoff.root_kind != "handoff" or handoff.seal_hashes != recipient.seal_hashes[-1:]:
            raise ValueError("recipient chain root is not its hand-off seal")
        record = check_handoff_record(
            read_json(recipient_root, handoff.journal, max_bytes=max_artifact_bytes),
            recipient_profile,
        )
        cited = _read_donor_seal(donor_root, donor_seal, donor_profile, source_root, **limits)
        final_ref = _resolve_warmup_final(donor_root, donor_seal, donor_profile, max_artifact_bytes)
        donor = (
            cited
            if final_ref == donor_seal
            else _read_donor_seal(donor_root, final_ref, donor_profile, source_root, **limits)
        )
        if not _is_warmup_final(donor, donor_profile):
            raise CodecHandoffDefect("foreign_donor", "the donor seal is not its warm-up-final")
        if (
            record["donor_identity"] != donor_profile.identity
            or record["donor_manifest"] != donor_profile.manifest
            or record["donor_seal_sha256"] != final_ref.content_hash.digest
            or record["donor_checkpoint_sha256"] != donor.checkpoint.content_hash.digest
        ):
            raise CodecHandoffDefect(
                "foreign_donor", "the record cites evidence that is not this donor's warm-up-final"
            )
        donor_engine = _restore(donor_profile, source_root, donor_root, donor, **limits)
        donor_rng = capture_rng()
        recipient_engine = _restore(
            recipient_profile, source_root, recipient_root, handoff, **limits
        )
        recipient_rng = capture_rng()
        guard_construction(donor_engine, recipient_engine)
        claimed = (record["codec_sha256"], record["optimizer_sha256"], record["rng_sha256"])
        for who, engine, rng in (
            ("donor", donor_engine, donor_rng),
            ("recipient", recipient_engine, recipient_rng),
        ):
            if (codec_digest(engine), optimizer_digest(engine), rng_digest(rng)) != claimed:
                raise CodecHandoffDefect(
                    "payload_mismatch", f"the {who}'s warm-up payload differs from the record's"
                )
    return VerifiedHandoff(
        recipient_profile.identity,
        donor_profile.identity,
        donor_profile.schedule.warmup_steps,
        handoff.journal.content_hash.digest,
        final_ref.content_hash.digest,
    )


# -------------------------------------------------------------------------- the materializer
def materialize_codec_handoff(
    donor_root: Path,
    donor_seal: ArtifactIdentity,
    donor_profile: HProfile,
    recipient_root: Path,
    recipient_profile: HProfile,
    source_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
    binding: ReceiptBinding | None = None,
) -> CodecHandoff:
    """Materialize a pair recipient from the donor's warm-up-final seal, then prove it.

    ``recipient_root`` must be empty. Every donor-side, pair and copy refusal fires before the
    first write, so such a call leaves it empty. Only the final proof of the written seal can
    refuse after writes: it leaves the record, checkpoint and seal but NO receipt, and the root
    must be discarded, never resumed. Write order: profile, record, checkpoint (measurement =
    the record's digest), hand-off seal, then the seal is proved from disk, then the progress
    receipt LAST. The receipt is the same ``h-probe-training-progress-v2`` the runner writes
    (``write_progress_receipt``; FOR-1693 moved it from v1: it cites the record above as its
    ``handoff`` and carries ``binding``'s block, pair, cell and authority digest), so ``run_probe_training(resume=receipt)`` continues the
    recipient through prediction, with no recipient-specific branch in the resume path.
    """
    require_pair(donor_profile, recipient_profile)
    root = normalize_d1_artifact_root(recipient_root)
    if any(root.iterdir()):
        raise ValueError("a hand-off recipient root must be empty")
    limits = {"max_source_bytes": max_source_bytes, "max_artifact_bytes": max_artifact_bytes}
    warmup = donor_profile.schedule.warmup_steps
    with torch.random.fork_rng(devices=_devices(recipient_profile)):
        donor = _read_donor_seal(donor_root, donor_seal, donor_profile, source_root, **limits)
        if not _is_warmup_final(donor, donor_profile):
            raise CodecHandoffDefect(
                "foreign_donor", "a hand-off cites the donor's warm-up-final seal and no other"
            )
        # Admitting the donor checkpoint restores the donor's RNG; building the recipient engine
        # afterwards re-seeds and draws, so the donor's state is restored again below, LAST.
        donor_engine = _restore(donor_profile, source_root, donor_root, donor, **limits)
        donor_rng = capture_rng()
        recipient = SentenceExecution(
            recipient_profile, source_root, max_source_bytes=max_source_bytes
        )
        guard_construction(donor_engine, recipient)
        _adopt_codec(recipient, donor_engine)
        _adopt_optimizer(recipient, donor_engine)
        recipient.sampler.restore(donor_engine.sampler.snapshot())
        codec, optimizer, rng = (
            codec_digest(donor_engine),
            optimizer_digest(donor_engine),
            rng_digest(donor_rng),
        )
        if (codec_digest(recipient), optimizer_digest(recipient)) != (codec, optimizer):
            raise CodecHandoffDefect(
                "payload_mismatch", "the recipient's copied warm-up differs from the donor's"
            )
        restore_rng(donor_rng)
        if rng_digest(capture_rng()) != rng:
            raise CodecHandoffDefect("payload_mismatch", "the donor's RNG state was not restored")
        profile_artifact = write_json(
            root, "profile.json", recipient_profile.manifest, max_bytes=max_artifact_bytes
        )
        record = write_json(
            root,
            f"{_PREFIX}-record.json",
            {
                "schema": HANDOFF_RECORD_SCHEMA,
                "run_identity": recipient_profile.identity,
                "parent": None,
                "warmup_steps": warmup,
                "donor_identity": donor_profile.identity,
                "donor_manifest": donor_profile.manifest,
                "donor_seal_sha256": donor_seal.content_hash.digest,
                "donor_checkpoint_sha256": donor.checkpoint.content_hash.digest,
                "codec_sha256": codec,
                "optimizer_sha256": optimizer,
                "rng_sha256": rng,
            },
            max_bytes=max_artifact_bytes,
        )
        name = f"{_PREFIX}-codec_warmup-{warmup:08d}"
        checkpoint = save_checkpoint(
            root,
            name + ".pt",
            recipient.model,
            recipient.optimizer,
            recipient.sampler,
            phase="codec_warmup",
            step=warmup,
            run_identity=recipient_profile.identity,
            measurement=record.content_hash,
            max_bytes=max_artifact_bytes,
        )
        seal = write_json(
            root,
            name + "-seal.json",
            {
                "schema": HANDOFF_SEAL_SCHEMA,
                "run_identity": recipient_profile.identity,
                "phase": "codec_warmup",
                "step": warmup,
                "checkpoint": artifact_reference(root, checkpoint),
                "journal": artifact_reference(root, record),
            },
            max_bytes=max_artifact_bytes,
        )

    # The seal is proved from disk BEFORE the receipt exists: a refusal here leaves a record, a
    # checkpoint and a seal, but never a receipt a caller could resume from.
    own = artifact_from_mapping(artifact_reference(root, seal))
    verified = admit_codec_pair(
        donor_root,
        donor_seal,
        donor_profile,
        root,
        own,
        recipient_profile,
        source_root,
        **limits,
    )
    receipt = write_progress_receipt(
        root,
        name + "-progress.json",
        recipient_profile,
        profile_artifact,
        status="checkpoint",
        phase="codec_warmup",
        step=warmup,
        last_seal=seal,
        evaluations=[],
        parent_receipt=None,
        inventory=recipient.inventory(),
        max_bytes=max_artifact_bytes,
        binding=binding,
        handoff=record,
    )
    published = {
        key: artifact_from_mapping(artifact_reference(root, value))
        for key, value in (("record", record), ("checkpoint", checkpoint), ("receipt", receipt))
    }
    return CodecHandoff(
        published["record"], published["checkpoint"], own, published["receipt"], verified
    )
