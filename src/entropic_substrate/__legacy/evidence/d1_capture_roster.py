"""Immutable D1 launch roster and explicitly excluded attempt lanes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Final

from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass

__all__ = [
    "D1AttemptLane",
    "D1DeclaredAttempt",
    "D1ExpectedRun",
    "D1LaunchRoster",
    "D1PairExclusion",
    "D1PairExclusionReason",
]

_ALL_ARMS: Final = set(D1Arm)


def _text(value: str, label: str) -> None:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty trimmed string")


def _hex(value: str, length: int, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != length
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be {length} lowercase hexadecimal characters")


def _commit(value: str, label: str) -> None:
    if type(value) is not str or not 7 <= len(value) <= 64:
        raise ValueError(f"{label} must be a 7-to-64-character commit identity")
    if value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be lowercase hexadecimal")


def _path(value: str, label: str) -> None:
    _text(value, label)
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or path.as_posix() != value or ".." in path.parts:
        raise ValueError(f"{label} must be a normalized repository-relative POSIX path")


@dataclass(frozen=True, slots=True)
class D1ExpectedRun:
    """Predeclared identity of one confirmatory cell and seed."""

    logical_run_id: str
    arm: D1Arm
    seed: int
    pairing_id: str
    family_digest: str
    construction_digest: str
    execution_digest: str
    aggregation_digest: str
    plan_digest: str
    code_commit: str
    repository_digest: str
    image_digest: str
    dataset_digest: str
    split_digest: str
    sampler_construction_digest: str

    def __post_init__(self) -> None:
        _text(self.logical_run_id, "logical_run_id")
        _text(self.pairing_id, "pairing_id")
        if type(self.arm) is not D1Arm:
            raise TypeError("arm must be a D1Arm")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for field in (
            "family_digest",
            "construction_digest",
            "execution_digest",
            "aggregation_digest",
            "plan_digest",
            "repository_digest",
            "dataset_digest",
            "split_digest",
            "sampler_construction_digest",
        ):
            _hex(getattr(self, field), 64, field)
        _commit(self.code_commit, "code_commit")
        if not self.image_digest.startswith("sha256:"):
            raise ValueError("image_digest must be an immutable sha256 OCI digest")
        _hex(self.image_digest.removeprefix("sha256:"), 64, "image_digest")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "aggregation_digest": self.aggregation_digest,
            "arm": self.arm.value,
            "code_commit": self.code_commit,
            "construction_digest": self.construction_digest,
            "dataset_digest": self.dataset_digest,
            "execution_digest": self.execution_digest,
            "family_digest": self.family_digest,
            "image_digest": self.image_digest,
            "logical_run_id": self.logical_run_id,
            "pairing_id": self.pairing_id,
            "plan_digest": self.plan_digest,
            "repository_digest": self.repository_digest,
            "sampler_construction_digest": self.sampler_construction_digest,
            "seed": self.seed,
            "split_digest": self.split_digest,
        }


class D1AttemptLane(StrEnum):
    """Non-success lanes that can never enter the confirmatory capture set."""

    EXCLUDED_PILOT = "excluded_pilot"
    ABORTED = "aborted"
    ABANDONED = "abandoned"


@dataclass(frozen=True, slots=True)
class D1DeclaredAttempt:
    """One visible pilot or unsuccessful launch attempt."""

    logical_run_id: str
    arm: D1Arm
    seed: int
    pairing_id: str
    run_class: RunClass
    lane: D1AttemptLane
    disposition: str

    def __post_init__(self) -> None:
        _text(self.logical_run_id, "attempt logical_run_id")
        _text(self.pairing_id, "attempt pairing_id")
        _text(self.disposition, "attempt disposition")
        if type(self.arm) is not D1Arm:
            raise TypeError("attempt arm must be a D1Arm")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("attempt seed must be a nonnegative integer")
        if type(self.run_class) is not RunClass or type(self.lane) is not D1AttemptLane:
            raise TypeError("attempt requires nominal run_class and lane values")
        if self.lane is D1AttemptLane.EXCLUDED_PILOT:
            if self.run_class is not RunClass.PILOT:
                raise ValueError("excluded pilot lane requires run_class=pilot")
        elif self.run_class is not RunClass.CONFIRMATORY:
            raise ValueError("aborted and abandoned lanes require confirmatory run_class")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "disposition": self.disposition,
            "lane": self.lane.value,
            "logical_run_id": self.logical_run_id,
            "pairing_id": self.pairing_id,
            "run_class": self.run_class.value,
            "seed": self.seed,
        }


class D1PairExclusionReason(StrEnum):
    """Registered D1 pair-exclusion vocabulary; capture does not apply it."""

    NONFINITE_FINAL_LOSS = "nonfinite_final_loss"


@dataclass(frozen=True, slots=True)
class D1PairExclusion:
    """Typed request for later analysis to exclude one complete pairing block."""

    pairing_id: str
    member_run_ids: tuple[str, ...]
    reason: D1PairExclusionReason
    evidence_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.pairing_id, "exclusion pairing_id")
        if type(self.reason) is not D1PairExclusionReason:
            raise TypeError("exclusion reason must be registered")
        if type(self.member_run_ids) is not tuple or not self.member_run_ids:
            raise ValueError("exclusion member_run_ids must be a nonempty tuple")
        if len(set(self.member_run_ids)) != len(self.member_run_ids):
            raise ValueError("exclusion member_run_ids must be unique")
        for value in self.member_run_ids:
            _text(value, "excluded logical_run_id")
        if type(self.evidence_paths) is not tuple or not self.evidence_paths:
            raise ValueError("pair exclusion requires at least one evidence path")
        for value in self.evidence_paths:
            _path(value, "exclusion evidence path")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "evidence_paths": list(self.evidence_paths),
            "member_run_ids": list(self.member_run_ids),
            "pairing_id": self.pairing_id,
            "reason": self.reason.value,
        }


@dataclass(frozen=True, slots=True)
class D1LaunchRoster:
    """Pre-outcome roster proving the complete expected D1 confirmatory set."""

    preregistration_path: str
    preregistration_commit: str
    family_digest: str
    expected_confirmatory: tuple[D1ExpectedRun, ...]
    attempts: tuple[D1DeclaredAttempt, ...]

    def __post_init__(self) -> None:
        _path(self.preregistration_path, "preregistration_path")
        _commit(self.preregistration_commit, "preregistration_commit")
        _hex(self.family_digest, 64, "family_digest")
        if type(self.expected_confirmatory) is not tuple or not self.expected_confirmatory:
            raise ValueError("roster requires a nonempty confirmatory tuple")
        if type(self.attempts) is not tuple:
            raise TypeError("attempts must be a tuple")
        all_ids = [run.logical_run_id for run in self.expected_confirmatory]
        all_ids.extend(attempt.logical_run_id for attempt in self.attempts)
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("logical_run_id must be unique across roster and attempt lanes")
        blocks: dict[str, list[D1ExpectedRun]] = {}
        for run in self.expected_confirmatory:
            if type(run) is not D1ExpectedRun:
                raise TypeError("expected_confirmatory must contain D1ExpectedRun values")
            if run.family_digest != self.family_digest:
                raise ValueError("expected run family_digest differs from roster")
            blocks.setdefault(run.pairing_id, []).append(run)
        for pairing_id, block in blocks.items():
            if len(block) != len(D1Arm) or {run.arm for run in block} != _ALL_ARMS:
                raise ValueError(
                    f"pairing block {pairing_id!r} must contain each D1 arm exactly once"
                )
            if len({run.seed for run in block}) != 1:
                raise ValueError(f"pairing block {pairing_id!r} must share one seed")
        for attempt in self.attempts:
            if type(attempt) is not D1DeclaredAttempt:
                raise TypeError("attempts must contain D1DeclaredAttempt values")

    @property
    def expected_run_ids(self) -> frozenset[str]:
        return frozenset(run.logical_run_id for run in self.expected_confirmatory)

    @property
    def pairing_ids(self) -> tuple[str, ...]:
        return tuple(sorted({run.pairing_id for run in self.expected_confirmatory}))

    def validate_pair_exclusions(
        self, exclusions: tuple[D1PairExclusion, ...]
    ) -> tuple[D1PairExclusion, ...]:
        if type(exclusions) is not tuple:
            raise TypeError("pair exclusions must be a tuple")
        if len({item.pairing_id for item in exclusions}) != len(exclusions):
            raise ValueError("a pairing block may be excluded at most once")
        expected_by_pair = {
            pairing_id: frozenset(
                run.logical_run_id
                for run in self.expected_confirmatory
                if run.pairing_id == pairing_id
            )
            for pairing_id in self.pairing_ids
        }
        for exclusion in exclusions:
            if type(exclusion) is not D1PairExclusion:
                raise TypeError("pair exclusions must contain D1PairExclusion values")
            if expected_by_pair.get(exclusion.pairing_id) != frozenset(exclusion.member_run_ids):
                raise ValueError("pair exclusion must name one whole pairing block")
        return exclusions

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "attempts": [item.manifest for item in self.attempts],
            "expected_confirmatory": [item.manifest for item in self.expected_confirmatory],
            "family_digest": self.family_digest,
            "preregistration_commit": self.preregistration_commit,
            "preregistration_path": self.preregistration_path,
            "schema_version": "d1-launch-roster-v1",
        }
