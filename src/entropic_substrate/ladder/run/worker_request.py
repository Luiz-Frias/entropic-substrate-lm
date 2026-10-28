"""Hash-bound H probe inputs; a smoke request carries no authority at all.

A smoke (``local_tiny``/``remote_tiny``) request renders ``h-probe-worker-request-v1``
byte-identically to every request written before rung B: ``"none"`` for both
authorities and no ``authority`` key, so every captured ``request_sha256`` still
re-derives. A scientific (``pilot``/``confirmatory``) request renders
``h-probe-worker-request-v2``: it carries its sealed ``RunAuthority`` and names the
authority's digest as both its scientific and its spending authority (operator ruling
2026-10-27, "option B"). The parse is schema-first, and each schema admits only its
own family of classes.
"""

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from inspect import signature
from pathlib import PurePosixPath

from entropic_substrate.core.probe.predictive_probe import ProbeConfig
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    artifact_mapping,
    normalize_d1_relative_artifact_path,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.admission import RUN_ADMISSION
from entropic_substrate.ladder.run.run_authority import RunAuthority, parse_run_authority

REQUEST_SCHEMA_V1 = "h-probe-worker-request-v1"
REQUEST_SCHEMA_V2 = "h-probe-worker-request-v2"


def worker_artifact(value: object, section: str) -> ArtifactIdentity:
    """Admit a flat result reference in one declared worker output section."""
    if section not in ("run", "capture", "verification", "worker"):
        raise ValueError("unknown worker result section")
    artifact = artifact_from_mapping(value)
    name = PurePosixPath(artifact.path).name
    if artifact.path != f"{section}/{name}" or PurePosixPath(name).suffix not in (".json", ".pt"):
        raise ValueError("worker result artifact belongs to another section")
    return ArtifactIdentity(name, artifact.content_hash)


@dataclass(frozen=True, slots=True)
class TransferLimits:
    max_object_bytes: int
    max_total_bytes: int
    max_objects: int

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("worker transfer limits must be explicit positive integers")


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
        if type(self.probe) is not ProbeConfig:
            raise TypeError("worker requires an explicit probe fitting envelope")
        self.probe.__post_init__()
        if (
            tuple(sorted(self.probe.lags)) != self.probe.lags
            or min(self.probe.train_positions, self.probe.eval_positions) < len(self.probe.lags)
            or any(
                type(v) not in (int, float)
                for v in (self.probe.learning_rate, self.probe.weight_decay)
            )
        ):
            raise ValueError("worker probe envelope is invalid")
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
        if surface == "remote_worker":
            if self.code_bundle is None or self.code_commit is None or self.image is None:
                raise ValueError("remote worker requires immutable image and source proof")
            if (
                self.code_bundle not in self.inputs
                or self.code_bundle.path != "code.bundle"
                or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.code_commit) is None
                or re.fullmatch(
                    r"ghcr\.io/luiz-frias/entropic-substrate-lm@sha256:[0-9a-f]{64}", self.image
                )
                is None
            ):
                raise ValueError("remote worker source proof or project image is invalid")
        elif (
            any(v is not None for v in (self.code_commit, self.code_bundle, self.image))
            or "code.bundle" in names
        ):
            raise ValueError("local tiny cannot claim unverified remote source proof")
        if self.pause is not None:
            if type(self.pause) is not tuple or len(self.pause) != 2:
                raise ValueError("worker pause requires an explicit phase/step pair")
            phase, step = self.pause
            budget = (
                self.profile.schedule.warmup_steps
                if phase == "codec_warmup"
                else self.profile.schedule.steps
            )
            if (
                phase not in ("codec_warmup", "prediction")
                or type(step) is not int
                or not 1 <= step <= budget
                or not (
                    step == budget
                    or step % self.profile.schedule.checkpoint_interval == 0
                    or (
                        phase == "prediction"
                        and step % self.profile.schedule.evaluation_interval == 0
                    )
                )
            ):
                raise ValueError("worker pause must select a declared seal boundary")

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


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    row: dict[str, object] = {}
    for key, value in pairs:
        if key in row:
            raise ValueError("duplicate worker request field")
        row[key] = value
    return row


def _construct[T](factory: Callable[..., T], value: object) -> T:
    return factory(**recovery_record(value, " ".join(signature(factory).parameters)))


def parse_probe_request(payload: bytes, *, max_bytes: int) -> ProbeRequest:
    if (
        type(max_bytes) is not int
        or max_bytes < 1
        or type(payload) is not bytes
        or not 0 < len(payload) <= max_bytes
    ):
        raise ValueError("worker request violates its byte limit")
    decoded = json.loads(payload, object_pairs_hook=_unique)
    schema = decoded.get("schema") if type(decoded) is dict else None
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
