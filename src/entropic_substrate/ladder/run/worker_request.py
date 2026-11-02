"""Hash-bound H probe inputs; a smoke request carries no authority at all.

A smoke (``local_tiny``/``remote_tiny``) request renders ``h-probe-worker-request-v1``
byte-identically to every request written before rung B: ``"none"`` for both
authorities and no ``authority`` key, so every captured ``request_sha256`` still
re-derives. A scientific (``pilot``/``confirmatory``) request renders
``h-probe-worker-request-v2``: it carries its sealed ``RunAuthority`` and names the
authority's digest as both its scientific and its spending authority (operator ruling
2026-10-27, "option B"). The parse is schema-first, and each schema admits only its
own family of classes.

FOR-1693 (h1-l4b, DG-A re-pin row 2). A codec-sharing PAIR is its own frozen type,
:class:`PairProbeRequest`, rendered as ``h-probe-worker-request-v3``: two cells in donor-first
order (the magnitude-spine donor, then the sphere-spine recipient), a block binding and ONE
authority for both. ``h-probe-worker-request-v2`` keeps its bytes and meaning; v1 and v2 still
parse into :class:`ProbeRequest` and never compare equal to a v3. Every single-cell consumer
reads ``request.profile``, which a pair does not have, so the union
(:data:`AnyProbeRequest`) is narrowed by type where it is admitted and refused by name where
it is not.

The ID scheme of the new door (listed in the L4b handoff for the conductor to record):

* the cell token (:func:`cell_token`) is ``donor`` or ``recipient``, one function of the cell
  profile's spine carrier (``magnitude`` donates, ``sphere`` receives; ``H1PreservationBlock``'s
  ``codec_sharing_groups`` pairs index 0 with 2 and 1 with 3, and in both pairs the donor is the
  magnitude-spine arm). It names ``run/<cell>/``, ``capture/<cell>/``, ``verification/<cell>/``,
  event and snapshot names and the stop-marker target, and is never ``profile.identity``;
* ``block_id`` is 64 lowercase hex, DECLARED and digest-bound, never verified here (a worker holds
  two of a block's four profiles): checking it against the block is an L5 obligation;
* ``pair_id`` is 64 lowercase hex and a FUNCTION of the two cells' profile identities
  (:func:`pair_identity`), recomputed at parse and refused on mismatch. It is also the
  ``run_identity`` of every pair-level worker event, never equal to either cell's identity.
"""

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from inspect import signature
from pathlib import PurePosixPath
from typing import NoReturn, cast

from entropic_substrate.core.probe.predictive_probe import ProbeConfig
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    artifact_mapping,
    normalize_d1_relative_artifact_path,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.register.h_preservation import H1ExclusionReason
from entropic_substrate.ladder.run.admission import RUN_ADMISSION, WorkerSurface
from entropic_substrate.ladder.run.codec_handoff import require_pair
from entropic_substrate.ladder.run.run_authority import RunAuthority, parse_run_authority

REQUEST_SCHEMA_V1 = "h-probe-worker-request-v1"
REQUEST_SCHEMA_V2 = "h-probe-worker-request-v2"
REQUEST_SCHEMA_V3 = "h-probe-worker-request-v3"
RESULT_SCHEMA_V1 = "h-probe-worker-result-v1"
RESULT_SCHEMA_V2 = "h-probe-worker-result-v2"

_ID64 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9-]*")
# worker_snapshot's name rule is ``[a-z0-9][a-z0-9-]{0,63}``; the longest name a cell composes
# is ``checkpoint-<cell>-<total:08d>``.
_SNAPSHOT_NAME_LIMIT = 64


def worker_artifact(value: object, section: str) -> ArtifactIdentity:
    """Admit a flat result reference in one declared worker output section."""
    if section not in ("run", "capture", "verification", "worker"):
        raise ValueError("unknown worker result section")
    artifact = artifact_from_mapping(value)
    name = PurePosixPath(artifact.path).name
    if artifact.path != f"{section}/{name}" or PurePosixPath(name).suffix not in (".json", ".pt"):
        raise ValueError("worker result artifact belongs to another section")
    return ArtifactIdentity(name, artifact.content_hash)


def worker_cell_artifact(value: object, section: str, cell: str) -> ArtifactIdentity:
    """Admit a result reference of a pair's CELL: ``<section>/<cell>/<name>`` and nothing else.

    The v3 counterpart of :func:`worker_artifact` (which keeps its two-part rule and messages for
    v1 and v2). Only ``run``, ``capture`` and ``verification`` are per-cell; the pair-level
    ``worker`` section stays flat and is admitted by :func:`worker_artifact`.
    """
    if section not in ("run", "capture", "verification"):
        raise ValueError("unknown worker cell result section")
    if type(cell) is not str or _TOKEN.fullmatch(cell) is None:
        raise ValueError("worker cell token is invalid")
    artifact = artifact_from_mapping(value)
    name = PurePosixPath(artifact.path).name
    if artifact.path != f"{section}/{cell}/{name}" or PurePosixPath(name).suffix not in (
        ".json",
        ".pt",
    ):
        raise ValueError("worker cell artifact belongs to another cell or section")
    return ArtifactIdentity(name, artifact.content_hash)


@dataclass(frozen=True, slots=True)
class TransferLimits:
    max_object_bytes: int
    max_total_bytes: int
    max_objects: int

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("worker transfer limits must be explicit positive integers")


def _check_probe(probe: ProbeConfig) -> None:
    if type(probe) is not ProbeConfig:
        raise TypeError("worker requires an explicit probe fitting envelope")
    probe.__post_init__()
    if (
        tuple(sorted(probe.lags)) != probe.lags
        or min(probe.train_positions, probe.eval_positions) < len(probe.lags)
        or any(type(v) not in (int, float) for v in (probe.learning_rate, probe.weight_decay))
    ):
        raise ValueError("worker probe envelope is invalid")


def _check_remote_proof(
    surface: WorkerSurface | None,
    inputs: tuple[ArtifactIdentity, ...],
    names: set[str],
    code_commit: str | None,
    code_bundle: ArtifactIdentity | None,
    image: str | None,
) -> None:
    if surface == "remote_worker":
        if code_bundle is None or code_commit is None or image is None:
            raise ValueError("remote worker requires immutable image and source proof")
        if (
            code_bundle not in inputs
            or code_bundle.path != "code.bundle"
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", code_commit) is None
            or re.fullmatch(r"ghcr\.io/luiz-frias/entropic-substrate-lm@sha256:[0-9a-f]{64}", image)
            is None
        ):
            raise ValueError("remote worker source proof or project image is invalid")
    elif any(v is not None for v in (code_commit, code_bundle, image)) or "code.bundle" in names:
        raise ValueError("local tiny cannot claim unverified remote source proof")


def _check_pause(pause: tuple[str, int] | None, profile: HProfile) -> None:
    if pause is None:
        return
    if type(pause) is not tuple or len(pause) != 2:
        raise ValueError("worker pause requires an explicit phase/step pair")
    phase, step = pause
    budget = profile.schedule.warmup_steps if phase == "codec_warmup" else profile.schedule.steps
    if (
        phase not in ("codec_warmup", "prediction")
        or type(step) is not int
        or not 1 <= step <= budget
        or not (
            step == budget
            or step % profile.schedule.checkpoint_interval == 0
            or (phase == "prediction" and step % profile.schedule.evaluation_interval == 0)
        )
    ):
        raise ValueError("worker pause must select a declared seal boundary")


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    profile: HProfile
    inputs: tuple[ArtifactIdentity, ...]
    nonce: str
    limits: TransferLimits
    probe: ProbeConfig
    code_commit: str | None
    code_bundle: ArtifactIdentity | None
    image: str | None
    resume: ArtifactIdentity | None
    pause: tuple[str, int] | None
    authority: RunAuthority | None = None

    def __post_init__(self) -> None:
        if type(self.profile) is not HProfile or type(self.limits) is not TransferLimits:
            raise TypeError("worker requires validated profile and transfer limits")
        run_class = self.profile.run_class
        if not RUN_ADMISSION.admits_class(run_class):
            raise ValueError("probe worker has no scientific execution authority")
        refusal = RUN_ADMISSION.authority_refusal(run_class, self.profile.stage, self.authority)
        if refusal is not None:
            raise ValueError(refusal)
        surface = RUN_ADMISSION.worker_surface(run_class)
        if not RUN_ADMISSION.admits(run_class, self.profile.stage, surface, self.authority):
            raise ValueError("remote successor requires its scientific gate adapter")
        ContentHash("sha256", self.nonce)
        _check_probe(self.probe)
        if (
            type(self.inputs) is not tuple
            or not self.inputs
            or len(self.inputs) > self.limits.max_objects
        ):
            raise ValueError("worker inputs must be a bounded explicit tuple")
        names: set[str] = set()
        for item in self.inputs:
            if type(item) is not ArtifactIdentity or item.content_hash.algorithm != "sha256":
                raise ValueError("worker inputs require SHA-256 identities")
            path = normalize_d1_relative_artifact_path(item.path, label="worker input")
            if path != item.path or path in names:
                raise ValueError("worker input paths must be unique and canonical")
            parts = PurePosixPath(path).parts
            is_resume = (
                len(parts) == 2
                and parts[0] == "run"
                and PurePosixPath(path).suffix in (".json", ".pt")
            )
            if path not in ("source/prepared-corpus.json", "code.bundle") and not (
                self.resume is not None and is_resume
            ):
                raise ValueError("worker input is outside the declared input namespace")
            names.add(path)
        source = ArtifactIdentity(
            "source/prepared-corpus.json", ContentHash("sha256", self.profile.source_digest)
        )
        if source not in self.inputs:
            raise ValueError("worker source differs from its profile")
        if self.resume is not None and (
            self.resume not in self.inputs
            or PurePosixPath(self.resume.path).parent != PurePosixPath("run")
            or PurePosixPath(self.resume.path).suffix != ".json"
        ):
            raise ValueError("worker resume must bind its training receipt in run/")
        _check_remote_proof(
            surface, self.inputs, names, self.code_commit, self.code_bundle, self.image
        )
        _check_pause(self.pause, self.profile)

    @property
    def run_identity(self) -> str:
        """The identity a worker event of this request carries: the one profile's."""
        return self.profile.identity

    @property
    def prefix(self) -> str:
        return f"h-probes/v1/{self.nonce}"

    @property
    def manifest(self) -> dict[str, object]:
        mandate = "none" if self.authority is None else self.authority.digest
        row: dict[str, object] = {
            "schema": REQUEST_SCHEMA_V1 if self.authority is None else REQUEST_SCHEMA_V2,
            "scientific_authority": mandate,
            "spending_authority": mandate,
            "profile": self.profile.manifest,
            "inputs": [artifact_mapping(a) for a in self.inputs],
            "nonce": self.nonce,
            "limits": asdict(self.limits),
            "probe": asdict(self.probe),
            "code_commit": self.code_commit,
            "code_bundle": None if self.code_bundle is None else artifact_mapping(self.code_bundle),
            "image": self.image,
            "resume": None if self.resume is None else artifact_mapping(self.resume),
            "pause": self.pause,
        }
        if self.authority is not None:
            row["authority"] = self.authority.manifest
        return row

    @property
    def payload(self) -> bytes:
        return json.dumps(
            self.manifest, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


# ------------------------------------------------------------------------------ the pair (v3)
def cell_token(profile: HProfile) -> str:
    """``donor`` or ``recipient``: one function of the cell profile, never its identity.

    The spine carrier decides: the ``magnitude`` arm donates its warmed codec and the ``sphere``
    arm receives it (the asymmetric donor-first predicate; parity-tested against
    ``H1PreservationBlock.codec_sharing_groups``, which this member never imports).
    """
    carrier = profile.spine.carrier
    token = {"magnitude": "donor", "sphere": "recipient"}.get(carrier)
    if token is None:
        raise ValueError(
            "a pair cell has no role: its spine carrier is neither magnitude nor sphere"
        )
    return token


def pair_identity(donor: HProfile, recipient: HProfile) -> str:
    """64-hex pair identity: a function of the two cells' profile identities, donor first."""
    payload = json.dumps(
        {
            "schema": "h-probe-pair-identity-v1",
            "donor": donor.identity,
            "recipient": recipient.identity,
        },
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class BlockBinding:
    """Where a pair sits: ``block_id`` is declared, ``pair_id``/``source_digest``/``seed`` verified."""

    block_id: str
    pair_id: str
    source_digest: str
    seed: int

    def __post_init__(self) -> None:
        for value in (self.block_id, self.pair_id):
            if type(value) is not str or _ID64.fullmatch(value) is None:
                raise ValueError("block and pair identifiers are 64 lowercase hex digests")
        ContentHash("sha256", self.source_digest)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("block seed must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class PairCell:
    """One cell of a pair with its own pause and (FOR-1694, C-L4c-2) its own resume receipt.

    The ``resume`` slot has been on the wire and in the canonical form since request v3, so
    admitting it changes no byte and no digest of any existing payload; a pod running the
    previous image refuses a resume-bearing request by name, the fail-closed direction.
    """

    profile: HProfile
    resume: ArtifactIdentity | None
    pause: tuple[str, int] | None

    @property
    def token(self) -> str:
        return cell_token(self.profile)


@dataclass(frozen=True, slots=True)
class PairProbeRequest:
    cells: tuple[PairCell, PairCell]
    inputs: tuple[ArtifactIdentity, ...]
    nonce: str
    limits: TransferLimits
    probe: ProbeConfig
    code_commit: str | None
    code_bundle: ArtifactIdentity | None
    image: str | None
    binding: BlockBinding
    authority: RunAuthority | None = None

    def __post_init__(self) -> None:
        if type(self.limits) is not TransferLimits or type(self.binding) is not BlockBinding:
            raise TypeError("pair worker requires validated transfer limits and block binding")
        if (
            type(self.cells) is not tuple
            or len(self.cells) != 2
            or any(type(cell) is not PairCell for cell in self.cells)
            or any(type(cell.profile) is not HProfile for cell in self.cells)
        ):
            raise TypeError("pair worker requires exactly two validated cells")
        donor, recipient = self.cells[0].profile, self.cells[1].profile
        # the pair predicate (codec_handoff's own comparator) and the asymmetric donor-first order
        require_pair(donor, recipient)
        if donor.spine.carrier != "magnitude" or recipient.spine.carrier != "sphere":
            raise ValueError("pair cells are donor first: the magnitude arm, then the sphere arm")
        tokens = (self.cells[0].token, self.cells[1].token)
        if tokens[0] == tokens[1] or any(_TOKEN.fullmatch(t) is None for t in tokens):
            raise ValueError("pair cell tokens must be distinct safe path segments")
        if any(len(f"checkpoint-{t}-00000000") > _SNAPSHOT_NAME_LIMIT for t in tokens):
            raise ValueError("pair cell token overflows the worker snapshot name budget")
        if donor.run_class != recipient.run_class or donor.stage != recipient.stage:
            raise ValueError("pair cells must share one run class and stage")
        run_class = donor.run_class
        if not RUN_ADMISSION.admits_class(run_class):
            raise ValueError("probe worker has no scientific execution authority")
        for profile in (donor, recipient):
            refusal = RUN_ADMISSION.authority_refusal(run_class, profile.stage, self.authority)
            if refusal is not None:
                raise ValueError(refusal)
        surface = RUN_ADMISSION.worker_surface(run_class)
        for profile in (donor, recipient):
            if not RUN_ADMISSION.admits(run_class, profile.stage, surface, self.authority):
                raise ValueError("remote successor requires its scientific gate adapter")
        if (
            self.binding.source_digest != donor.source_digest
            or self.binding.source_digest != recipient.source_digest
            or self.binding.seed != donor.seed
            or self.binding.seed != recipient.seed
        ):
            raise ValueError("pair block binding differs from its cells' source or seed")
        if self.binding.pair_id != pair_identity(donor, recipient):
            raise ValueError("pair identifier is not the identity of its two cells")
        ContentHash("sha256", self.nonce)
        _check_probe(self.probe)
        if (
            type(self.inputs) is not tuple
            or not self.inputs
            or len(self.inputs) > self.limits.max_objects
        ):
            raise ValueError("worker inputs must be a bounded explicit tuple")
        names: set[str] = set()
        resumes = {c.token for c in self.cells if c.resume is not None}
        for item in self.inputs:
            if type(item) is not ArtifactIdentity or item.content_hash.algorithm != "sha256":
                raise ValueError("worker inputs require SHA-256 identities")
            path = normalize_d1_relative_artifact_path(item.path, label="worker input")
            if path != item.path or path in names:
                raise ValueError("worker input paths must be unique and canonical")
            parts = PurePosixPath(path).parts
            owner = (
                parts[1]
                if len(parts) == 3
                and parts[0] == "run"
                and PurePosixPath(path).suffix in (".json", ".pt")
                else None
            )
            # a cell's own ``run/<cell>/`` files are inputs only when THAT cell resumes: a root the
            # worker would never read is a hidden input, refused rather than ignored
            if path not in ("source/prepared-corpus.json", "code.bundle") and owner not in resumes:
                raise ValueError("worker input is outside the declared input namespace")
            names.add(path)
        source = ArtifactIdentity(
            "source/prepared-corpus.json", ContentHash("sha256", donor.source_digest)
        )
        if source not in self.inputs:
            raise ValueError("worker source differs from its profile")
        _check_remote_proof(
            surface, self.inputs, names, self.code_commit, self.code_bundle, self.image
        )
        self._check_resumes()
        for cell in self.cells:
            _check_pause(cell.pause, cell.profile)

    def _check_resumes(self) -> None:
        """Per-cell resume (C-L4c-2). The rules, by NAME (the content of a receipt is read by the
        host builder and by the worker's own receipt admission, never by this gate):

        * a cell's resume is a progress receipt (``*-progress.json``) among the inputs under
          ``run/<its own token>/``;
        * the recipient's resume requires the donor's root among the inputs (so the donor carries
          its own resume) and NEVER names its hand-off receipt (``handoff-*``): R3, a recipient
          that died before its first prediction seal is re-materialized from the donor root, and
          that request carries no recipient resume at all.
        """
        for cell in self.cells:
            resume = cell.resume
            if resume is None:
                continue
            parts = PurePosixPath(resume.path).parts
            if (
                type(resume) is not ArtifactIdentity
                or resume.content_hash.algorithm != "sha256"
                or resume not in self.inputs
                or len(parts) != 3
                or parts[:2] != ("run", cell.token)
                or not parts[2].endswith("-progress.json")
            ):
                raise ValueError(
                    "a pair cell's resume must bind its own training receipt in run/<cell>/"
                )
        donor, recipient = self.cells
        if recipient.resume is not None:
            if PurePosixPath(recipient.resume.path).name.startswith("handoff-"):
                raise ValueError(
                    "a recipient resume never names its hand-off receipt: a recipient that died "
                    "before its first prediction seal is re-materialized from the donor root"
                )
            if donor.resume is None or not any(
                PurePosixPath(item.path).parts[:2] == ("run", donor.token) for item in self.inputs
            ):
                raise ValueError("a recipient resume requires the donor's root among its inputs")

    @property
    def donor(self) -> HProfile:
        return self.cells[0].profile

    @property
    def recipient(self) -> HProfile:
        return self.cells[1].profile

    @property
    def tokens(self) -> tuple[str, str]:
        return (self.cells[0].token, self.cells[1].token)

    @property
    def run_identity(self) -> str:
        """The pair-level identity every pair-level worker event carries (never a cell's)."""
        return self.binding.pair_id

    @property
    def prefix(self) -> str:
        return f"h-probes/v1/{self.nonce}"

    @property
    def manifest(self) -> dict[str, object]:
        mandate = "none" if self.authority is None else self.authority.digest
        row: dict[str, object] = {
            "schema": REQUEST_SCHEMA_V3,
            "scientific_authority": mandate,
            "spending_authority": mandate,
            "cells": [
                {
                    "profile": cell.profile.manifest,
                    "resume": None if cell.resume is None else artifact_mapping(cell.resume),
                    "pause": cell.pause,
                }
                for cell in self.cells
            ],
            "block": asdict(self.binding),
            "inputs": [artifact_mapping(a) for a in self.inputs],
            "nonce": self.nonce,
            "limits": asdict(self.limits),
            "probe": asdict(self.probe),
            "code_commit": self.code_commit,
            "code_bundle": None if self.code_bundle is None else artifact_mapping(self.code_bundle),
            "image": self.image,
        }
        if self.authority is not None:
            row["authority"] = self.authority.manifest
        return row

    @property
    def payload(self) -> bytes:
        return json.dumps(
            self.manifest, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


type AnyProbeRequest = ProbeRequest | PairProbeRequest


def refuse_pair_result_for_single() -> NoReturn:
    """A pair's ``h-probe-worker-result-v2`` met a single-profile request: refused BY NAME.

    The one text every single-profile reader (``object_worker``, ``worker_completion``,
    ``h_probe_collection``) raises, so the three sites cannot drift.
    """
    raise ValueError(
        f"worker result schema {RESULT_SCHEMA_V2!r} names a pair's cells; "
        f"a single-profile request admits {RESULT_SCHEMA_V1}"
    )


def refuse_version(presented: object, minimum: str, what: str) -> NoReturn:
    """Refuse an artifact outside its version ladder, naming the presented schema and the minimum."""
    raise ValueError(
        f"{what} schema {presented!r} is not admitted here; the minimum supported version is {minimum}"
    )


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    row: dict[str, object] = {}
    for key, value in pairs:
        if key in row:
            raise ValueError("duplicate worker request field")
        row[key] = value
    return row


def _construct[T](factory: Callable[..., T], value: object) -> T:
    return factory(**recovery_record(value, " ".join(signature(factory).parameters)))


def _envelope(row: dict[str, object]) -> tuple[TransferLimits, ProbeConfig]:
    limits = recovery_record(row["limits"], " ".join(f.name for f in fields(TransferLimits)))
    probe = recovery_record(row["probe"], " ".join(f.name for f in fields(ProbeConfig))).copy()
    if type(probe["lags"]) is not list:
        raise ValueError("worker lags must be an array")
    probe["lags"] = tuple(probe["lags"])
    return _construct(TransferLimits, limits), _construct(ProbeConfig, probe)


def _parse_pair(decoded: dict[str, object], payload: bytes) -> PairProbeRequest:
    has_authority = "authority" in decoded
    row = recovery_record(
        decoded,
        "schema scientific_authority spending_authority cells block inputs nonce limits probe code_commit code_bundle image"
        + (" authority" if has_authority else ""),
    )
    authority = parse_run_authority(row["authority"]) if has_authority else None
    mandate = "none" if authority is None else authority.digest
    if row["scientific_authority"] != mandate or row["spending_authority"] != mandate:
        raise ValueError("unknown worker request scope or authority")
    if type(row["inputs"]) is not list:
        raise ValueError("worker inputs must be explicit")
    cells_raw = row["cells"]
    if type(cells_raw) is not list or len(cells_raw) != 2:
        raise ValueError("pair request requires exactly two cells")
    cells: list[PairCell] = []
    for raw in cells_raw:
        cell = recovery_record(raw, "profile resume pause")
        pause = cell["pause"]
        if pause is not None and (type(pause) is not list or len(pause) != 2):
            raise ValueError("worker pause must be an explicit pair")
        cells.append(
            PairCell(
                HProfile.from_manifest(cell["profile"]),
                None if cell["resume"] is None else artifact_from_mapping(cell["resume"]),
                None if pause is None else (pause[0], pause[1]),
            )
        )
    limits, probe = _envelope(row)
    binding = _construct(BlockBinding, row["block"])
    result = _construct(
        PairProbeRequest,
        {
            "cells": (cells[0], cells[1]),
            "inputs": tuple(artifact_from_mapping(a) for a in row["inputs"]),
            "nonce": row["nonce"],
            "limits": limits,
            "probe": probe,
            "code_commit": row["code_commit"],
            "code_bundle": None
            if row["code_bundle"] is None
            else artifact_from_mapping(row["code_bundle"]),
            "image": row["image"],
            "binding": binding,
            "authority": authority,
        },
    )
    if result.payload != payload:
        raise ValueError("worker request is not canonical JSON")
    return result


def parse_probe_request(payload: bytes, *, max_bytes: int) -> AnyProbeRequest:
    if (
        type(max_bytes) is not int
        or max_bytes < 1
        or type(payload) is not bytes
        or not 0 < len(payload) <= max_bytes
    ):
        raise ValueError("worker request violates its byte limit")
    decoded = json.loads(payload, object_pairs_hook=_unique)
    schema = decoded.get("schema") if type(decoded) is dict else None
    if schema == REQUEST_SCHEMA_V3:
        return _parse_pair(decoded, payload)
    if schema not in (REQUEST_SCHEMA_V1, REQUEST_SCHEMA_V2):
        raise ValueError("unknown worker request scope or authority")
    row = recovery_record(
        decoded,
        "schema scientific_authority spending_authority profile inputs nonce limits probe code_commit code_bundle image resume pause"
        + (" authority" if schema == REQUEST_SCHEMA_V2 else ""),
    )
    authority = None if schema == REQUEST_SCHEMA_V1 else parse_run_authority(row["authority"])
    mandate = "none" if authority is None else authority.digest
    if row["scientific_authority"] != mandate or row["spending_authority"] != mandate:
        raise ValueError("unknown worker request scope or authority")
    if type(row["inputs"]) is not list:
        raise ValueError("worker inputs must be explicit")
    limits = recovery_record(row["limits"], " ".join(f.name for f in fields(TransferLimits)))
    probe = recovery_record(row["probe"], " ".join(f.name for f in fields(ProbeConfig))).copy()
    if type(probe["lags"]) is not list:
        raise ValueError("worker lags must be an array")
    probe["lags"] = tuple(probe["lags"])
    pause = row["pause"]
    if pause is not None and (type(pause) is not list or len(pause) != 2):
        raise ValueError("worker pause must be an explicit pair")
    result = _construct(
        ProbeRequest,
        {
            "profile": HProfile.from_manifest(row["profile"]),
            "inputs": tuple(artifact_from_mapping(a) for a in row["inputs"]),
            "nonce": row["nonce"],
            "limits": _construct(TransferLimits, limits),
            "probe": _construct(ProbeConfig, probe),
            "code_commit": row["code_commit"],
            "code_bundle": None
            if row["code_bundle"] is None
            else artifact_from_mapping(row["code_bundle"]),
            "image": row["image"],
            "resume": None if row["resume"] is None else artifact_from_mapping(row["resume"]),
            "pause": None if pause is None else tuple(pause),
            "authority": authority,
        },
    )
    if result.payload != payload:
        raise ValueError("worker request is not canonical JSON")
    return result


# ------------------------------------------------------------- the pair result (v2) vocabulary
# FOR-1693 (DH-44, P5-16). ONE closed execution-status set, ONE closed cell-outcome set, and a
# pair verdict that is separate from both. ``worker_execution`` builds the record, ``object_worker``
# re-derives it from disk, ``worker_completion`` re-admits it on the host; all three read this
# structural predicate so none of them can drift from the others.
PAIR_STATUSES = (
    "probe_workload_complete",
    "paused",
    "stopped_by_operator",
    "pair_excluded",
    "execution_defect",
)
CELL_OUTCOMES = (
    "complete",
    "paused",
    "stopped_by_operator",
    "scientific_gap",
    "execution_defect",
    "not_run",
)
_RESULT_FIELDS = (
    "schema request_sha256 run_identity scientific_authority status runtime cells verdict "
    "wall_seconds"
)
_CELL_FIELDS = (
    "cell run_identity outcome successful_optimizer_steps training failure generation "
    "diagnostics capture verification"
)
_SECTIONS = {
    "training": "run",
    "failure": "run",
    "generation": "run",
    "diagnostics": "run",
    "capture": "capture",
    "verification": "verification",
}
_LATER_STAGES = ("generation", "diagnostics", "capture", "verification")


def parse_pair_result(value: object, request: PairProbeRequest) -> dict[str, object]:
    """Admit the STRUCTURE of an ``h-probe-worker-result-v2``; the evidence is re-read elsewhere.

    A v1 result presented for a pair is refused by name with the minimum supported version.
    Every cell entry's reference lives under its own cell (``run/<cell>/...``), its outcome fixes
    which references are present, and the status, the verdict and the outcomes must agree.
    """
    if type(value) is dict and value.get("schema") == RESULT_SCHEMA_V1:
        refuse_version(RESULT_SCHEMA_V1, RESULT_SCHEMA_V2, "pair worker result")
    row = recovery_record(value, _RESULT_FIELDS)
    if (
        row["schema"] != RESULT_SCHEMA_V2
        or row["request_sha256"] != request.digest
        or row["run_identity"] != request.run_identity
        or row["scientific_authority"] != "none"
    ):
        raise ValueError("pair worker result identity mismatch")
    duration = row["wall_seconds"]
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not duration >= 0
        or duration == float("inf")
    ):
        raise ValueError("worker result duration is invalid")
    status = row["status"]
    if status not in PAIR_STATUSES:
        raise ValueError("pair worker result status is outside the closed set")
    entries = row["cells"]
    if type(entries) is not list or len(entries) != 2:
        raise ValueError("pair worker result requires exactly two cell entries, donor first")
    outcomes: list[str] = []
    for raw, cell in zip(entries, request.cells, strict=True):
        entry = recovery_record(raw, _CELL_FIELDS)
        token = cell.token
        outcome = entry["outcome"]
        steps = entry["successful_optimizer_steps"]
        if (
            entry["cell"] != token
            or entry["run_identity"] != cell.profile.identity
            or outcome not in CELL_OUTCOMES
            or type(steps) is not int
            or steps < 0
        ):
            raise ValueError("pair worker result cell entry identity or outcome mismatch")
        for key, section in _SECTIONS.items():
            if entry[key] is not None:
                worker_cell_artifact(entry[key], section, token)
        present = {key for key in _SECTIONS if entry[key] is not None}
        if outcome == "complete":
            required = {"training", *_LATER_STAGES}
            allowed = required
        elif outcome in ("paused", "stopped_by_operator"):
            required = allowed = {"training"}
        elif outcome == "scientific_gap":
            required, allowed = {"failure"}, {"failure", "training"}
        elif outcome == "execution_defect":
            required, allowed = set(), {"failure", "training"}
        else:
            required = allowed = set()
        if not required <= present <= allowed or (outcome == "not_run" and steps != 0):
            raise ValueError("pair worker result cell references differ from its outcome")
        outcomes.append(str(outcome))
    verdict = recovery_record(row["verdict"], "verdict reason cell")
    kind, reason, named = verdict["verdict"], verdict["reason"], verdict["cell"]
    tokens = request.tokens
    if kind in ("complete", "none"):
        if reason is not None or named is not None:
            raise ValueError("a verdict without an exclusion names no reason and no cell")
    elif kind == "excluded":
        if type(reason) is not str or named not in tokens:
            raise ValueError("an excluded verdict names its H1ExclusionReason and its cell")
        H1ExclusionReason(reason)
    else:
        raise ValueError("pair worker result verdict is outside the closed set")
    named_outcome = None if named not in tokens else outcomes[tokens.index(cast(str, named))]
    if status == "probe_workload_complete":
        agrees = outcomes == ["complete", "complete"] and kind == "complete"
    elif status == "paused":
        agrees = kind == "none" and "paused" in outcomes and "stopped_by_operator" not in outcomes
    elif status == "stopped_by_operator":
        agrees = kind == "none" and "stopped_by_operator" in outcomes
    elif status == "pair_excluded":
        agrees = (
            kind == "excluded"
            and reason in ("missing_endpoint", "nonfinite_endpoint")
            and named_outcome == "scientific_gap"
        )
    else:
        agrees = (
            kind == "excluded"
            and reason == "reader_refusal"
            and named_outcome == "execution_defect"
        )
    if not agrees:
        raise ValueError("pair worker result status, verdict and cell outcomes disagree")
    return row
