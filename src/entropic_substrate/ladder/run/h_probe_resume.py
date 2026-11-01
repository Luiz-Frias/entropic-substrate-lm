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
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath

from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    normalize_d1_artifact_root,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.journal import read_json
from entropic_substrate.ladder.run.runner import (
    PROGRESS_SCHEMA_V1,
    PROGRESS_SCHEMA_V2,
    refuse_progress_version,
)
from entropic_substrate.ladder.run.worker_request import ProbeRequest

__all__ = ["build_resume_request"]

_COLLECTION_SCHEMA = "h-probe-terminal-collection-v1"
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
