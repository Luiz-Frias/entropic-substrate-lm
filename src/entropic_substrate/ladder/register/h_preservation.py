"""H1's paired 2x2 numerical design; never a registration or launch authorization."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self, cast

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.records import content_identity
from entropic_substrate.core.sentence.recovery_state import recovery_record

_AXES = (
    ("magnitude", "magnitude"),
    ("sphere", "magnitude"),
    ("magnitude", "sphere"),
    ("sphere", "sphere"),
)
type ContrastTerms = tuple[tuple[str, float], ...]


def _shared_profile(profile: HProfile) -> str:
    value = profile.manifest
    del value["name"]
    del cast(dict[str, object], value["law"])["mode"]
    del cast(dict[str, object], value["spine"])["carrier"]
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def h1_construction_digest(profile: HProfile) -> str:
    """Identity of one H1 construction, invariant across seed blocks and arm names.

    Built from a fresh ``profile.manifest`` with exactly three top-level keys removed:
    ``name``, ``seed`` and ``source_seed``. Nothing nested is removed, so ``law.mode`` and
    ``spine.carrier`` stay and the four arms of a block get four distinct digests.

    Gate ruling (DH-14 reading): ``source_seed`` is removed. It names a replicate's data
    stream, the same category as ``seed``; the corpus itself stays pinned by
    ``source_digest``. Per-arm digests must be invariant across seed blocks, so that a
    construction seen in several blocks is one construction.

    ``sinks.h_config_hash`` is a deliberately different predicate: it keeps
    ``source_seed`` and is not touched. This is also not ``_shared_profile`` (the pair-block
    predicate, which keeps ``seed`` and removes ``name``, ``law.mode`` and
    ``spine.carrier``). The result is the full 64-character lowercase hex digest, in the
    shape of ``HProfile.identity``.
    """
    if type(profile) is not HProfile:
        raise TypeError("H1 construction digest requires an exact numerical H profile")
    value = profile.manifest
    for key in ("name", "seed", "source_seed"):
        del value[key]
    return content_identity((json.dumps(value, sort_keys=True, separators=(",", ":")),))


class H1ExclusionReason(StrEnum):
    """Closed vocabulary for dropping a whole H1 block; a drop is reported, never silent.

    Consumers must type-check with ``type(x) is H1ExclusionReason``: members compare equal
    to their bare strings, so value equality alone would admit an unregistered string.
    """

    NONFINITE_ENDPOINT = "nonfinite_endpoint"
    MISSING_ENDPOINT = "missing_endpoint"
    READER_REFUSAL = "reader_refusal"
    INADMISSIBLE_ESS = "inadmissible_ess"
    INDEPENDENT_WARMUP = "independent_warmup"


_ROW_PHASES = frozenset({"codec_warmup", "prediction"})
_ROW_STATUSES = frozenset({"started", "committed", "scientific_gap", "execution_defect"})


def h1_exposure_bytes(rows: Iterable[Mapping[str, object]]) -> int:
    """Cumulative canonical source bytes over committed prediction-phase journal step rows.

    This is the DH-25 default rule, open for the operator before L8 seals: prediction-phase
    rows only, for every cell, so the donor and the recipient of a codec hand-off are
    counted the same way. Warm-up rows and non-committed rows contribute nothing.

    Measurement path, fails loud: a non-mapping row, an unrecognized ``phase`` or
    ``status``, or a prediction-phase row (any status) whose ``source_bytes`` is not a
    nonnegative ``int`` raises. Persisted journal shards hold only committed rows, so a
    failure-shaped row in the input means attempt records were fed in. The input is consumed
    once, with no dedupe, reorder or repair.
    """
    total = 0
    for row in rows:
        if not isinstance(row, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("exposure row must be a mapping")
        if row.get("phase") not in _ROW_PHASES or row.get("status") not in _ROW_STATUSES:
            raise ValueError("exposure row has an unrecognized phase or status")
        if row["phase"] != "prediction":
            continue
        source_bytes = row.get("source_bytes")
        if type(source_bytes) is not int or source_bytes < 0:
            raise ValueError("prediction row source_bytes must be a nonnegative int")
        if row["status"] == "committed":
            total += source_bytes
    return total


@dataclass(frozen=True, slots=True)
class H1PreservationBlock:
    """One complete paired seed block with no bundled changes outside its axes.

    This validates declared configurations, not observed data, matched active
    parameter counts, artifact sharing, power, or predecessor eligibility.
    """

    profiles: tuple[HProfile, ...]

    def __post_init__(self) -> None:
        if type(self.profiles) is not tuple or len(self.profiles) != 4:
            raise ValueError("H1 requires exactly four profiles in one paired block")
        if any(type(profile) is not HProfile for profile in self.profiles):
            raise TypeError("H1 requires exact numerical H profiles")
        # Reconstruct nested specifications so forged/mutated frozen objects
        # cannot bypass their original validation or remain aliased to callers.
        profiles = tuple(HProfile.from_manifest(profile.manifest) for profile in self.profiles)
        if any(profile.stage != "H1" for profile in profiles):
            raise ValueError("H1 block cannot contain another stage")
        axes = {(profile.law.mode, profile.spine.carrier): profile for profile in profiles}
        if set(axes) != set(_AXES):
            raise ValueError("H1 must contain every sentence/spine preservation combination once")
        if len({profile.name for profile in profiles}) != 4:
            raise ValueError("H1 arm names must be distinct within the seed block")
        ordered = tuple(axes[axis] for axis in _AXES)
        common = _shared_profile(ordered[0])
        if any(_shared_profile(profile) != common for profile in ordered[1:]):
            raise ValueError("H1 paired profiles differ outside the two preservation axes")
        object.__setattr__(self, "profiles", ordered)

    @property
    def factorial_terms(self) -> dict[str, ContrastTerms]:
        """Sphere-minus-magnitude effects; positive means increase, not improvement.

        These are available coefficients, not a deciding family. Registration
        must still declare metrics, directions, margins and multiplicity.
        """
        weights = {
            "sentence_normalization": (-0.5, 0.5, -0.5, 0.5),
            "spine_normalization": (-0.5, -0.5, 0.5, 0.5),
            "interaction": (1.0, -1.0, -1.0, 1.0),
        }
        return {
            name: tuple(
                (profile.name, weight)
                for profile, weight in zip(self.profiles, values, strict=True)
            )
            for name, values in weights.items()
        }

    @property
    def codec_sharing_groups(self) -> tuple[tuple[str, str], tuple[str, str]]:
        """Required same-interface pairings; an executor must bind actual artifacts."""
        names = tuple(profile.name for profile in self.profiles)
        return ((names[0], names[2]), (names[1], names[3]))

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema": "h1-preservation-block-v1",
            "scientific_authority": "none",
            "profiles": [profile.manifest for profile in self.profiles],
        }

    @property
    def identity(self) -> str:
        return content_identity((json.dumps(self.manifest, sort_keys=True, separators=(",", ":")),))

    @classmethod
    def from_manifest(cls, value: object) -> Self:
        row = recovery_record(value, "schema scientific_authority profiles")
        if row["schema"] != "h1-preservation-block-v1" or row["scientific_authority"] != "none":
            raise ValueError("unknown H1 design schema or forbidden authority claim")
        profiles = row["profiles"]
        if type(profiles) is not list or len(profiles) != 4:
            raise ValueError("H1 manifest requires exactly four profiles")
        return cls(tuple(HProfile.from_manifest(profile) for profile in profiles))
