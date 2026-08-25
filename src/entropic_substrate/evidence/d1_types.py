"""D1 verdict vocabulary, canonical factorial construction, and value types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from entropic_substrate.evidence.lifecycle import (
    ContentHash,
    LifecycleContractError,
    SuccessorAuthorization,
)

D1_VERDICT_SCHEMA_VERSION: Final = "d1-verdict/v2"
D1_ARM_IDS: Final[tuple[str, ...]] = ("none", "sg", "ac", "both")
D1_STABILIZATION_ARM_IDS: Final[tuple[str, ...]] = ("sg", "ac", "both")

ROOT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "rung",
        "feasible_arms",
        "preregistration",
        "registered_constants",
        "arms",
        "contrasts",
        "clauses_by_arm",
        "integrity",
        "exclusions",
        "inputs",
        "review",
        "clause_precedence",
        "verdict_vocabulary",
        "verdict",
        "successor",
    }
)
CLAUSE_NAMES: Final[frozenset[str]] = frozenset({"noncollapse", "future_grounding", "language"})
CLAUSE_STATUS: Final[frozenset[str]] = frozenset({"PASS", "FAIL", "INCONCLUSIVE"})
VERDICTS: Final[frozenset[str]] = frozenset(
    {"SUPPORTED", "NOT_SUPPORTED", "PRACTICALLY_NULL", "INCONCLUSIVE", "VALID_NEGATIVE", "RETIRED"}
)
EVIDENCE_CLASSES: Final[frozenset[str]] = frozenset(
    {"MEASURED", "DERIVED_FROM_MEASURED", "MODEL_ESTIMATE", "RATE_ESTIMATE", "PROVIDER_BILLED"}
)
CONTRAST_KINDS: Final[frozenset[str]] = frozenset({"PAIRWISE", "FACTORIAL_INTERACTION"})
CONTRAST_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        "FAVORS_LEFT",
        "FAVORS_RIGHT",
        "PRACTICALLY_NULL",
        "INCONCLUSIVE",
        "INTERACTION_DETECTED",
        "INTERACTION_INCONCLUSIVE",
    }
)
CONTRAST_VOCABULARY: Final = "d1-contrast/v1"

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


class D1VerdictValidationError(ValueError):
    """A missing, malformed, or unregistered D1 verdict value."""

    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = {"path": path, "code": code}
        super().__init__(f"D1 verdict validation {code} at {path}: {detail}")


@dataclass(frozen=True)
class PreregistrationIdentity:
    path: str
    commit: str
    family_digest: str


@dataclass(frozen=True)
class D1FamilyIdentity:
    """The registered full-construction identity a D1 verdict must match.

    This value intentionally contains no approximation of a model recipe. An
    adapter derives it from the already validated ``D1ExperimentFamily``.
    """

    family_digest: str
    arm_construction_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        try:
            ContentHash(algorithm="sha256", digest=self.family_digest)
        except LifecycleContractError as exc:
            msg = "D1 family digest must be a lowercase SHA-256 digest"
            raise ValueError(msg) from exc
        if set(self.arm_construction_digests) != set(D1_ARM_IDS):
            msg = f"D1 family identity requires exact arm identities {D1_ARM_IDS}"
            raise ValueError(msg)
        frozen: dict[str, str] = {}
        for arm in D1_ARM_IDS:
            digest = self.arm_construction_digests[arm]
            try:
                ContentHash(algorithm="sha256", digest=digest)
            except LifecycleContractError as exc:
                msg = f"D1 arm {arm} construction digest must be lowercase SHA-256"
                raise ValueError(msg) from exc
            frozen[arm] = digest
        object.__setattr__(self, "arm_construction_digests", MappingProxyType(frozen))


@dataclass(frozen=True)
class D1ArmEvidence:
    arm_id: str
    construction_digest: str


@dataclass(frozen=True)
class D1ContrastTerm:
    arm_id: str
    coefficient: float


@dataclass(frozen=True)
class D1Endpoint:
    metric: str
    estimator: str
    smoothing_window: int


@dataclass(frozen=True)
class D1Contrast:
    contrast_id: str
    contrast_kind: str
    terms: tuple[D1ContrastTerm, ...]
    endpoint: D1Endpoint
    estimate: float
    sample_sd: float | None
    interval_lower: float | None
    interval_upper: float | None
    confidence_level: float
    multiplicity: str
    margins: Mapping[str, float]
    n: int
    df: int | None
    evidence_class: str
    decision_vocabulary: str
    decision: str
    evidence_hash: ContentHash


@dataclass(frozen=True)
class D1Clause:
    registered_rule: str
    decision_vocabulary: str
    threshold: float | None
    margin: float | None
    status: str
    evidence_hash: ContentHash


@dataclass(frozen=True)
class D1Integrity:
    status: str
    roster_complete: bool
    pairing_verified: bool
    data_order_verified: bool
    completion_verified: bool
    evidence_hash: ContentHash


@dataclass(frozen=True)
class D1InputIdentity:
    """Protocol-compatible input identity with optional stronger hashing."""

    path: str
    md5: ContentHash
    sha256: ContentHash | None


@dataclass(frozen=True)
class D1ReviewFindingDisposition:
    finding_id: str
    status: str
    rationale: str
    evidence_sha256: str | None


@dataclass(frozen=True)
class D1ReviewAdjudication:
    status: str
    major_findings: tuple[str, ...]
    dispositions: Mapping[str, D1ReviewFindingDisposition]
    execution_receipt_sha256: str
    operator_record_sha256: str
    report_sha256: str
    review_packet_sha256: str
    review_request_sha256: str


@dataclass(frozen=True)
class D1Review:
    open_major_findings: tuple[str, ...]
    adjudication: D1ReviewAdjudication


@dataclass(frozen=True)
class D1Successor:
    authorization: SuccessorAuthorization
    eligible_arms: tuple[str, ...]
    eligible_strategies: tuple[str, ...]


@dataclass(frozen=True)
class FiledGateAuthorization:
    rung: str
    eligible_arms: tuple[str, ...]
    eligible_strategies: tuple[str, ...]


@dataclass(frozen=True)
class FiledGateValidNegative:
    rung: str
    verdict: str


@dataclass(frozen=True)
class FiledGateRetirement:
    rung: str


FiledGateOutcome = FiledGateAuthorization | FiledGateValidNegative | FiledGateRetirement
