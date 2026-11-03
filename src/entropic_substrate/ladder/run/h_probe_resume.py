"""Build the request that resumes a stopped or paused H probe lineage on a fresh pod.

Ruling D4 (2026-10-22): the resume request rides the EXISTING
``h-probe-worker-request-v1`` ``resume`` field; the request schema and its serializer
(:mod:`~entropic_substrate.ladder.run.worker_request`, a reader-tuple member) are not
touched. This module only assembles a :class:`ProbeRequest` from evidence already
collected by :func:`~entropic_substrate.ladder.run.h_probe_collection.collect_probe_terminal`;
``ProbeRequest``'s own construction is the final gate (input namespace, ``max_objects``,
the resume binding). It publishes nothing and spends nothing.

Inputs of the resumed request: the previous request's non-``run/`` inputs (the prepared
source and, remotely, the code bundle) plus every flat ``run/*.json`` / ``run/*.pt``
file of the collected run root, each identified by the digest of its bytes as read
here. The resume receipt is the collection's ``resume_artifact``; it must match those
bytes and be a paused ``h-probe-training-progress-v2`` receipt of the same profile
(FOR-1693 moved it from v1: a v1 receipt is refused by name).

FOR-1694 (h1-l4c, C-L4c-2): :func:`build_pair_resume_request` resumes a PAIR per cell through the
``PairCell.resume`` slot request v3 already carries, from the evidence of a dead, paused, stopped
or failed pod. Three death cases, decided from the receipts the evidence holds (never from a
caller's say-so):

1. the donor died before its training finished: the donor resumes from its newest receipt (a
   warm-up or prediction receipt), and the recipient is materialized from the donor's
   warm-up-final seal as usual;
2. the donor's training finished and the recipient had no receipt of its OWN (it had not started,
   or only its hand-off receipt exists): the donor carries its final receipt (a
   ``training_complete`` receipt resumes instantly, then its generation, diagnostics, capture and
   verification run again from the trained root) and the recipient carries NO resume and names no
   hand-off receipt: it is re-materialized from the inputs under ``run/donor/`` (R3);
3. the recipient trained past its hand-off: the donor carries its final receipt as in case 2 and
   the recipient resumes from its own newest receipt, with the donor's root among the inputs.

The resumed request keeps the block binding, the authority and both profiles, so it re-derives the
same ``pair_id`` and the receipts of the dead pod stay admissible. A collection, when given, must be
v2 and of a RESUMABLE class: a pause or stop capture, or the DH-45 post-training ``execution_defect``
failure that records no exclusion member; a recorded exclusion (DH-47 ``reader_refusal`` inside a
failure snapshot, an ``excluded`` terminal) and a ``scientific_gap`` or ``diagnostic_gap`` failure
are never resumed, because a retry would turn registered negative evidence into a different
outcome and that is a scientific choice the operator has not ruled.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import cast

from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    normalize_d1_artifact_root,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.h_probe_collection import (
    COLLECTION_SCHEMA_V1,
    admit_pair_collection,
)
from entropic_substrate.ladder.run.journal import read_json
from entropic_substrate.ladder.run.runner import (
    PROGRESS_FIELDS,
    PROGRESS_SCHEMA_V1,
    PROGRESS_SCHEMA_V2,
    refuse_progress_version,
)
from entropic_substrate.ladder.run.worker_request import (
    PairCell,
    PairProbeRequest,
    ProbeRequest,
)

__all__ = ["PAIR_RESUME_SOURCES", "build_pair_resume_request", "build_resume_request"]

_COLLECTION_SCHEMA = COLLECTION_SCHEMA_V1
# The two terminal captures that hand back a resumable training receipt
# (h_probe_collection.py ``resume_artifact=``, pause and operator stop).
_RESUMABLE_CAPTURES = ("pause_bytes_verified", "stop_bytes_verified")
_RUN_SUFFIXES = (".json", ".pt")


def _run_inputs(run_root: Path, max_object_bytes: int) -> list[ArtifactIdentity]:
    root = normalize_d1_artifact_root(run_root)
    inputs: list[ArtifactIdentity] = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if not entry.is_file() or entry.suffix not in _RUN_SUFFIXES:
            raise ValueError(f"collected run root holds a non-artifact entry: {entry.name}")
        _, _, digest = read_d1_artifact_bytes(
            root, entry.name, label="resume input", max_bytes=max_object_bytes
        )
        inputs.append(ArtifactIdentity(f"run/{entry.name}", ContentHash("sha256", digest)))
    return inputs


def build_resume_request(
    previous: ProbeRequest,
    collection: Mapping[str, object],
    run_root: Path,
    *,
    nonce: str,
) -> ProbeRequest:
    """The request that continues ``previous``'s lineage from its collected receipt.

    ``collection`` is the terminal collection report of ``previous``; ``run_root`` is
    that collection's ``run/`` directory. Raises :class:`ValueError` for a collection of
    another request or of a non-resumable terminal, a reused nonce, a resume receipt
    whose bytes differ from the collected ones, or any input the request refuses.
    """
    if (
        collection.get("schema") != _COLLECTION_SCHEMA
        or collection.get("request_sha256") != previous.digest
    ):
        raise ValueError("the collection is not a terminal collection of the previous request")
    if collection.get("capture_status") not in _RESUMABLE_CAPTURES:
        raise ValueError(
            f"capture_status {collection.get('capture_status')!r} is not a stop or pause "
            "capture; there is no receipt to resume"
        )
    if nonce == previous.nonce:
        raise ValueError("a resumed request needs a fresh nonce (its digest names its run)")
    resume = artifact_from_mapping(collection.get("resume_artifact"))
    if PurePosixPath(resume.path).parent != PurePosixPath("run"):
        raise ValueError("the collected resume_artifact is not a run/ receipt")
    collected = _run_inputs(run_root, previous.limits.max_object_bytes)
    if resume not in collected:
        raise ValueError(
            f"the resume receipt {resume.path} differs from the collected bytes in {run_root}"
        )
    receipt = read_json(
        normalize_d1_artifact_root(run_root),
        ArtifactIdentity(PurePosixPath(resume.path).name, resume.content_hash),
        max_bytes=previous.limits.max_object_bytes,
    )
    if receipt.get("schema") == PROGRESS_SCHEMA_V1:
        refuse_progress_version(PROGRESS_SCHEMA_V1)
    if (
        receipt.get("schema") != PROGRESS_SCHEMA_V2
        or receipt.get("status") != "paused"
        or receipt.get("run_identity") != previous.profile.identity
    ):
        raise ValueError("the resume artifact is not a paused training receipt of this profile")
    kept = tuple(a for a in previous.inputs if PurePosixPath(a.path).parts[0] != "run")
    return replace(previous, inputs=(*kept, *collected), nonce=nonce, resume=resume, pause=None)


# --------------------------------------------------------------------------- the pair (C-L4c-2)
# The CLOSED set of evidence a pair may be resumed from. Anything else (an excluded terminal, a
# failure that records an H1ExclusionReason, a scientific or diagnostic gap, a pod that never began
# a cell, a complete pair) is refused by name.
PAIR_RESUME_SOURCES = ("pause_bytes_verified", "stop_bytes_verified", "dh45_post_training_failure")


def _pair_resume_source(collection: Mapping[str, object]) -> str:
    status = collection.get("capture_status")
    if status in ("pause_bytes_verified", "stop_bytes_verified"):
        return cast(str, status)
    verdict = collection.get("verdict")
    if (
        collection.get("terminal_name") == "failure"
        and status == "failure_bytes_verified"
        and collection.get("reported_outcome") == "execution_defect"
        # a post-training failure names no verdict (DH-45), or it is a failure AT final_snapshot
        # after result v2 was written, where the collector admits a non-exclusion verdict only
        # (complete or none) and an exclusion stays a recorded outcome
        and (verdict is None or cast(Mapping[str, object], verdict).get("verdict") != "excluded")
    ):
        return "dh45_post_training_failure"
    raise ValueError(
        f"a pair collected as {collection.get('terminal_name')!r} / {status!r} "
        f"(verdict {collection.get('verdict')!r}) is not a resumable source; the resumable "
        f"sources are {', '.join(PAIR_RESUME_SOURCES)}. A recorded exclusion is never overwritten "
        "by a resumed lineage"
    )


def _cell_receipts(
    run_root: Path, request: PairProbeRequest, index: int, max_object_bytes: int
) -> list[tuple[ArtifactIdentity, dict[str, object]]]:
    """Every progress-v2 receipt of THIS cell in its collected root, bound to this pair and cell."""
    cell = request.cells[index]
    found: list[tuple[ArtifactIdentity, dict[str, object]]] = []
    if not run_root.is_dir():
        return found
    for entry in sorted(run_root.iterdir(), key=lambda path: path.name):
        if not entry.name.endswith("-progress.json"):
            continue
        _, _, digest = read_d1_artifact_bytes(
            run_root, entry.name, label="resume input", max_bytes=max_object_bytes
        )
        identity = ArtifactIdentity(entry.name, ContentHash("sha256", digest))
        raw = read_json(run_root, identity, max_bytes=max_object_bytes)
        if raw.get("schema") == PROGRESS_SCHEMA_V1:
            refuse_progress_version(PROGRESS_SCHEMA_V1)
        receipt = recovery_record(raw, PROGRESS_FIELDS)
        if (
            receipt["schema"] != PROGRESS_SCHEMA_V2
            or receipt["run_identity"] != cell.profile.identity
            or (receipt["block"], receipt["pair"], receipt["cell"])
            != (request.binding.block_id, request.binding.pair_id, cell.token)
        ):
            raise ValueError(f"the {cell.token} receipt {entry.name} is not this pair's own")
        found.append((identity, receipt))
    return found


def _is_handoff_receipt(receipt: Mapping[str, object]) -> bool:
    return (
        receipt["handoff"] is not None
        and receipt["own_optimizer_steps"] == 0
        and receipt["phase"] == "codec_warmup"
    )


def _newest(
    receipts: list[tuple[ArtifactIdentity, dict[str, object]]],
) -> tuple[ArtifactIdentity, dict[str, object]] | None:
    return max(
        receipts,
        key=lambda item: cast(int, item[1]["successful_optimizer_steps"]),
        default=None,
    )


def _root_inputs(evidence: Path, token: str, max_object_bytes: int) -> list[ArtifactIdentity]:
    root = normalize_d1_artifact_root(evidence / "run" / token)
    inputs: list[ArtifactIdentity] = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if not entry.is_file() or entry.suffix not in _RUN_SUFFIXES:
            raise ValueError(f"collected run root holds a non-artifact entry: {token}/{entry.name}")
        _, _, digest = read_d1_artifact_bytes(
            root, entry.name, label="resume input", max_bytes=max_object_bytes
        )
        inputs.append(ArtifactIdentity(f"run/{token}/{entry.name}", ContentHash("sha256", digest)))
    return inputs


def build_pair_resume_request(
    previous: PairProbeRequest,
    evidence: Path,
    *,
    nonce: str,
    collection: Mapping[str, object] | None = None,
) -> PairProbeRequest:
    """The pair request that continues ``previous`` from the evidence under ``evidence``.

    ``evidence`` holds ``run/donor/`` and ``run/recipient/`` as a host collected them: a terminal
    collection's destination (give its ``collection`` too, which must be a v2 collection of a
    resumable source) or a materialized checkpoint snapshot of a pod that died without a terminal
    (no collection: nothing was recorded that a retry could overwrite). Publishes nothing and
    spends nothing; ``PairProbeRequest``'s own construction is the final gate.
    """
    if collection is not None:
        admit_pair_collection(collection, previous)
        _pair_resume_source(collection)
    if nonce == previous.nonce:
        raise ValueError("a resumed request needs a fresh nonce (its digest names its run)")
    limit = previous.limits.max_object_bytes
    donor_found = _newest(_cell_receipts(evidence / "run" / "donor", previous, 0, limit))
    recipient_receipts = _cell_receipts(evidence / "run" / "recipient", previous, 1, limit)
    if donor_found is None:
        raise ValueError(
            "the evidence holds no donor receipt: nothing to resume, relaunch the pair"
        )
    donor_receipt, donor_row = donor_found
    if donor_row["phase"] not in ("codec_warmup", "prediction"):
        raise ValueError("a donor resume names a warm-up or prediction receipt")
    donor_cell = PairCell(previous.donor, _receipt_ref(donor_receipt, "donor"), None)
    donor_done = donor_row["status"] == "training_complete"
    recipient_resume: ArtifactIdentity | None = None
    inputs = [*_root_inputs(evidence, "donor", limit)]
    if donor_done:
        # the recipient resumes only from a receipt of ITS OWN training; its hand-off receipt is
        # never a resume target (R3): a recipient with only that is re-materialized from run/donor/
        own = [item for item in recipient_receipts if not _is_handoff_receipt(item[1])]
        newest = _newest(own)
        if newest is not None:
            recipient_resume = _receipt_ref(newest[0], "recipient")
            inputs.extend(_root_inputs(evidence, "recipient", limit))
    elif recipient_receipts:
        raise ValueError(
            "the evidence holds a recipient receipt but the donor's training is not complete"
        )
    kept = tuple(a for a in previous.inputs if PurePosixPath(a.path).parts[0] != "run")
    return replace(
        previous,
        cells=(donor_cell, PairCell(previous.recipient, recipient_resume, None)),
        inputs=(*kept, *inputs),
        nonce=nonce,
    )


def _receipt_ref(receipt: ArtifactIdentity, token: str) -> ArtifactIdentity:
    """A collected receipt, named by the request's own path (``run/<cell>/<name>``)."""
    return ArtifactIdentity(f"run/{token}/{receipt.path}", receipt.content_hash)
