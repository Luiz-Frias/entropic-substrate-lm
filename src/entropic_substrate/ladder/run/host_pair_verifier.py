"""Host-side CPU verifier of a codec hand-off pair, equal to ``admit_codec_pair`` on identical bytes.

FOR-1694 (h1-l4c, the host CUDA pair-verifier note of 2026-11-01). ``admit_codec_pair`` restores
both checkpoints into a ``SentenceExecution`` built on ``profile.device``; for a ``pilot`` or
``confirmatory`` profile that is a CUDA engine (``profile.py`` requires cuda/bf16 for every class
but ``local_tiny``, and a CPU substitute profile cannot exist because ``HProfile.identity``
includes the device), so it is pod-only. This module admits the SAME pair from COLLECTED bytes on
a host with no CUDA device:

* the seal walks (``seals.read_seal``) are CPU-safe by construction (they build their binding on
  the ``meta`` device), so both chains, the hand-off record and the donor's warm-up-final are
  verified exactly as the pod verifier does;
* the three payload digests (codec, optimizer, RNG) are computed with ``codec_handoff``'s OWN
  functions over each checkpoint's STORED payload, CPU-mapped (``torch.load(..., map_location=
  "cpu")``), never over a freshly built engine: ``codec_digest``, ``optimizer_digest`` and
  ``rng_digest`` are handed minimal stand-ins that expose exactly the attributes they read, so
  there is one definition of each digest in the tree;
* the RNG digest covers the stored payload's full CUDA tuple (``rng_digest`` iterates it), so a
  pod-written checkpoint's per-device states are part of the proof. On a CPU fixture the tuple is
  empty, and ``tests/test_h_host_pair_verifier.py`` pins a synthetic non-empty tuple.

``codec_handoff`` is a checkpoint-reader member and is NOT touched here (FOR-1717 owns its
frame replays); this module is not a member and is not in ``_READER_MODULES``. The two private
helpers it mirrors (``_walk`` and ``_same_value``) are copied, not imported, and are named below.

Scope of the claim: equal to ``admit_codec_pair`` on every refusal the pair check names
(``independent_warmup``, ``chained_handoff``, ``foreign_donor``, ``construction_divergence``,
``payload_mismatch``), proven by ``tests/test_h_host_pair_verifier.py`` over L4a's
``ADMIT_HAZARDS`` table. NOT claimed: the full tensor/optimizer-moment admission
(``checkpoint.load_checkpoint``'s finiteness and shape checks), which only a restore performs; the
digest equalities are the guard on those values here. ``read_seal`` reads each checkpoint through
``checkpoint_metadata``, which refuses a checkpoint whose ``reader`` identity differs from the
host's (Python, torch and member bytes): on a real host this verifier admits pod bytes only under
the pinned image regime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import artifact_from_mapping, read_d1_artifact_bytes
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.run.codec_handoff import (
    CodecHandoffDefect,
    VerifiedHandoff,
    codec_digest,
    optimizer_digest,
    require_pair,
    rng_digest,
)
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import artifact_reference, read_json
from entropic_substrate.ladder.run.seals import (
    HANDOFF_SEAL_SCHEMA,
    SealedStep,
    check_handoff_record,
    read_seal,
)

__all__ = ["admit_codec_pair_on_host"]

_SEAL_FIELDS = "schema run_identity phase step checkpoint journal"
_CHECKPOINT_FIELDS = (
    "schema reader binding step source training model optimizer rng rollout measurement"
)


@dataclass(frozen=True)
class _Stored:
    """One checkpoint's stored payload, CPU-mapped, and the binding it carries."""

    binding: dict[str, Any]
    model: dict[str, object]
    optimizer: dict[str, object]
    rng: object


def _load(root: Path, sealed: SealedStep, *, max_bytes: int) -> _Stored:
    """The stored payload of a sealed checkpoint. ``read_seal`` already admitted its hash, schema,
    reader identity, binding and cursor through ``checkpoint_metadata``; this re-reads the same
    bytes for the three payload keys that function does not return."""
    path = artifact_reference(root, sealed.checkpoint)["path"]
    _, payload, digest = read_d1_artifact_bytes(
        root, path, label="H checkpoint", max_bytes=max_bytes
    )
    if digest != sealed.checkpoint.content_hash.digest:
        raise ValueError("checkpoint content hash mismatch")
    saved: object = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    row = recovery_record(saved, _CHECKPOINT_FIELDS)
    binding = json.loads(cast(str, row["binding"]))
    if (
        type(binding) is not dict
        or not isinstance(row["model"], dict)  # torch's state_dict is an OrderedDict
        or not isinstance(row["optimizer"], dict)
    ):
        raise ValueError("checkpoint payload is malformed")
    return _Stored(
        cast(dict[str, Any], binding),
        cast(dict[str, object], row["model"]),
        cast(dict[str, object], row["optimizer"]),
        row["rng"],
    )


def _engine_view(stored: _Stored) -> SentenceExecution:
    """The minimal attributes ``codec_digest`` and ``optimizer_digest`` read, over STORED values:
    ``model.state_dict()``, ``model.named_parameters()`` and ``optimizer.param_groups`` /
    ``optimizer.state_dict()``. The parameter order is the stored binding's own ``groups``."""
    names: list[list[str]] = stored.binding["groups"]
    tokens = {name: object() for group in names for name in group}
    groups = [{"params": [tokens[name] for name in group]} for group in names]
    model = SimpleNamespace(
        state_dict=lambda: stored.model,
        named_parameters=lambda: iter(tokens.items()),
    )
    optimizer = SimpleNamespace(param_groups=groups, state_dict=lambda: stored.optimizer)
    return cast(SentenceExecution, SimpleNamespace(model=model, optimizer=optimizer))


def _walk_root(
    root: Path, cited: ArtifactIdentity, profile: HProfile, max_bytes: int
) -> ArtifactIdentity:
    """The chain ROOT seal's identity, from raw rows (a copy of ``codec_handoff._walk``'s loop:
    that function is private to a member this slice does not touch). UNVERIFIED by itself; the
    caller reads the node through ``read_seal``."""
    found = cited
    cursor: ArtifactIdentity | None = cited
    count = 0
    limit = profile.schedule.warmup_steps + profile.schedule.steps
    while cursor is not None and count < limit:
        row = recovery_record(read_json(root, cursor, max_bytes=max_bytes), _SEAL_FIELDS)
        found, count = cursor, count + 1
        if row["schema"] == HANDOFF_SEAL_SCHEMA:
            break
        journal = read_json(root, artifact_from_mapping(row["journal"]), max_bytes=max_bytes)
        parent = journal.get("parent")
        cursor = None if parent is None else artifact_from_mapping(parent)
    return found


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _same_value(left: object, right: object) -> bool:
    """Bit equality (a copy of ``codec_handoff._same_value``): ``torch.equal`` after dtype and
    shape, canonical JSON for everything else."""
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and bool(torch.equal(left, right))
        )
    return not isinstance(right, torch.Tensor) and _canonical(left) == _canonical(right)


def _without_cell_axes(stored: _Stored) -> dict[str, Any]:
    binding = json.loads(_canonical(stored.binding))
    del binding["run_identity"]
    del binding["spine"]["carrier"]
    return binding


def _guard_construction(donor: _Stored, recipient: _Stored) -> None:
    """``codec_handoff.guard_construction`` over stored payloads: every state key is either
    COPIED (``codec.*``) or GUARDED here."""
    if _without_cell_axes(donor) != _without_cell_axes(recipient):
        raise CodecHandoffDefect(
            "construction_divergence",
            "donor and recipient bindings differ beyond the run identity and spine carrier",
        )
    if donor.model.keys() != recipient.model.keys():
        raise CodecHandoffDefect("construction_divergence", "model inventories differ")
    for name in sorted(k for k in recipient.model if not k.startswith("codec.")):
        if not _same_value(donor.model[name], recipient.model[name]):
            raise CodecHandoffDefect(
                "construction_divergence", f"non-codec state {name} differs from the donor's"
            )


def admit_codec_pair_on_host(
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
    """Prove, on a host without CUDA, that the recipient's codec is the donor's warm-up.

    Same arguments, same result type and the same ``CodecHandoffDefect`` codes as
    :func:`entropic_substrate.ladder.run.codec_handoff.admit_codec_pair`, in the same order:
    the pair predicate, the recipient's chain (an ordinary warm-up step-one root is
    ``independent_warmup``), its hand-off root and record, the donor's chain (a hand-off donor is
    ``chained_handoff``; no warm-up-final is ``foreign_donor``) and the record's claims about it,
    then the construction guard, then the donor's and the recipient's payload digests against the
    record's. Pure reader: writes nothing.
    """
    require_pair(donor_profile, recipient_profile)

    def walk(root: Path, seal: ArtifactIdentity, profile: HProfile) -> SealedStep:
        return read_seal(
            root,
            seal,
            profile,
            source_root,
            max_bytes=max_artifact_bytes,
            max_source_bytes=max_source_bytes,
        )

    recipient = walk(recipient_root, recipient_seal, recipient_profile)
    if recipient.root_kind == "initial":
        raise CodecHandoffDefect(
            "independent_warmup",
            "the recipient's chain starts with an ordinary warm-up step-one seal",
        )
    root_ref = _walk_root(recipient_root, recipient_seal, recipient_profile, max_artifact_bytes)
    handoff = walk(recipient_root, root_ref, recipient_profile)
    if handoff.root_kind != "handoff" or handoff.seal_hashes != recipient.seal_hashes[-1:]:
        raise ValueError("recipient chain root is not its hand-off seal")
    record = check_handoff_record(
        read_json(recipient_root, handoff.journal, max_bytes=max_artifact_bytes),
        recipient_profile,
    )
    cited = walk(donor_root, donor_seal, donor_profile)
    if cited.root_kind == "handoff":
        raise CodecHandoffDefect(
            "chained_handoff",
            "the donor's own codec arrived by hand-off; a donor warms up for itself",
        )
    final_ref = cited.warmup_final
    if final_ref is None:
        raise CodecHandoffDefect("foreign_donor", "the donor's chain holds no warm-up-final seal")
    donor = (
        cited
        if final_ref.content_hash == donor_seal.content_hash
        else walk(donor_root, final_ref, donor_profile)
    )
    if donor.root_kind == "handoff":
        raise CodecHandoffDefect(
            "chained_handoff",
            "the donor's own codec arrived by hand-off; a donor warms up for itself",
        )
    if not (donor.phase == "codec_warmup" and donor.step == donor_profile.schedule.warmup_steps):
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
    stored_donor = _load(donor_root, donor, max_bytes=max_artifact_bytes)
    stored_recipient = _load(recipient_root, handoff, max_bytes=max_artifact_bytes)
    _guard_construction(stored_donor, stored_recipient)
    claimed = (record["codec_sha256"], record["optimizer_sha256"], record["rng_sha256"])
    for who, stored in (("donor", stored_donor), ("recipient", stored_recipient)):
        view = _engine_view(stored)
        if (codec_digest(view), optimizer_digest(view), rng_digest(stored.rng)) != claimed:
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
