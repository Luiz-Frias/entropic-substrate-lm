"""H1's paired 2x2 numerical design; never a registration or launch authorization."""

import json
from dataclasses import dataclass
from typing import Self, cast

from entropic_substrate.sentence_prediction.profile import HProfile
from entropic_substrate.sentence_prediction.records import content_identity
from entropic_substrate.sentence_prediction.recovery_state import recovery_record

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
