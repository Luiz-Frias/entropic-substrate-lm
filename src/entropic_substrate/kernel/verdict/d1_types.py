"""D1 verdict vocabulary, canonical factorial construction, and value types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from entropic_substrate.kernel.lifecycle import (
    ContentHash,
    LifecycleContractError,
    SuccessorAuthorization,
)
from entropic_substrate.kernel.verdict.d1_statistical_gaps import D1EstimateGap

D1_VERDICT_SCHEMA_VERSION: Final = "d1-verdict/v2"
D1_ARM_IDS: Final[tuple[str, ...]] = ("none", "sg", "ac", "both")
D1_STABILIZATION_ARM_IDS: Final[tuple[str, ...]] = ("sg", "ac", "both")
D1_OPAQUE_ARM_ALIAS_IDS: Final[tuple[str, ...]] = ("A", "B", "C", "D")

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


class ClauseStatus(StrEnum):
    """Clause and integrity status; members mirror their persisted wire values."""

    PASS = "PASS"  # nosec B105 -- a verdict status wire literal, never a credential
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class VerdictEvidenceClass(StrEnum):
    """Verdict evidence class, UPPERCASE on the wire.

    Distinct from ``kernel.identity.d1_primitives.EvidenceClass``, whose values are
    lowercase: serializing that type's ``.name`` would look byte-identical while
    breaking the persisted contract.
    """

    MEASURED = "MEASURED"
    DERIVED_FROM_MEASURED = "DERIVED_FROM_MEASURED"
    MODEL_ESTIMATE = "MODEL_ESTIMATE"
    RATE_ESTIMATE = "RATE_ESTIMATE"
    PROVIDER_BILLED = "PROVIDER_BILLED"


class ContrastKind(StrEnum):
    PAIRWISE = "PAIRWISE"
    FACTORIAL_INTERACTION = "FACTORIAL_INTERACTION"


class ContrastDecision(StrEnum):
    FAVORS_LEFT = "FAVORS_LEFT"
    FAVORS_RIGHT = "FAVORS_RIGHT"
    PRACTICALLY_NULL = "PRACTICALLY_NULL"
    INCONCLUSIVE = "INCONCLUSIVE"
    INTERACTION_DETECTED = "INTERACTION_DETECTED"
    INTERACTION_INCONCLUSIVE = "INTERACTION_INCONCLUSIVE"


class ReviewAdjudicationStatus(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    NOT_REQUIRED = "NOT_REQUIRED"


class ReviewFindingStatus(StrEnum):
    """A single MAJOR finding is open or resolved, never NOT_REQUIRED."""

    OPEN = "OPEN"
    RESOLVED = "RESOLVED"


# The legacy sets keep their names and hold PLAIN str members: d1_parsing renders
# sorted(CLAUSE_STATUS) into a caller-visible error. Measured 2026-10-06 at this
# commit: sorted(frozenset(ClauseStatus)) prints
# [<ClauseStatus.FAIL: 'FAIL'>, ...] where the plain-str set prints ['FAIL', ...];
# tests/test_d1_verdict_v2.py pins the message.
CLAUSE_STATUS: Final[frozenset[str]] = frozenset(member.value for member in ClauseStatus)
VERDICTS: Final[frozenset[str]] = frozenset(
    {"SUPPORTED", "NOT_SUPPORTED", "PRACTICALLY_NULL", "INCONCLUSIVE", "VALID_NEGATIVE", "RETIRED"}
)
EVIDENCE_CLASSES: Final[frozenset[str]] = frozenset(member.value for member in VerdictEvidenceClass)
CONTRAST_KINDS: Final[frozenset[str]] = frozenset(member.value for member in ContrastKind)
CONTRAST_DECISIONS: Final[frozenset[str]] = frozenset(member.value for member in ContrastDecision)
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


def _closed[Vocabulary: StrEnum](
    vocabulary: type[Vocabulary], value: object, field: str
) -> Vocabulary:
    """Hold ``value`` as a member of its closed vocabulary, or raise at construction.

    The guard is nominal, like every sibling enum field in the repo: a raw ``str``
    (even one whose value is in the vocabulary), ``None`` or a member of another
    enum raises ``TypeError``. The parsers' coded membership checks and their
    ``Vocabulary(value)`` wraps remain the only ``str`` to member mapping.
    """
    if type(value) is not vocabulary:
        msg = f"{field} must be {vocabulary.__name__}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


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
    contrast_kind: ContrastKind
    terms: tuple[D1ContrastTerm, ...]
    endpoint: D1Endpoint
    estimate: float | None
    sample_sd: float | None
    interval_lower: float | None
    interval_upper: float | None
    confidence_level: float
    multiplicity: str
    margins: Mapping[str, float]
    n: int
    df: int | None
    evidence_class: VerdictEvidenceClass
    decision_vocabulary: str
    decision: ContrastDecision
    evidence_hash: ContentHash
    unavailable: D1EstimateGap | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contrast_kind",
            _closed(ContrastKind, self.contrast_kind, "D1Contrast.contrast_kind"),
        )
        object.__setattr__(
            self,
            "evidence_class",
            _closed(VerdictEvidenceClass, self.evidence_class, "D1Contrast.evidence_class"),
        )
        object.__setattr__(
            self, "decision", _closed(ContrastDecision, self.decision, "D1Contrast.decision")
        )


@dataclass(frozen=True)
class D1Clause:
    registered_rule: str
    decision_vocabulary: str
    threshold: float | None
    margin: float | None
    status: ClauseStatus
    evidence_hash: ContentHash

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _closed(ClauseStatus, self.status, "D1Clause.status"))


@dataclass(frozen=True)
class D1Integrity:
    status: ClauseStatus
    roster_complete: bool
    pairing_verified: bool
    data_order_verified: bool
    completion_verified: bool
    evidence_hash: ContentHash

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _closed(ClauseStatus, self.status, "D1Integrity.status"))


@dataclass(frozen=True)
class D1InputIdentity:
    """Protocol-compatible input identity with optional stronger hashing."""

    path: str
    md5: ContentHash
    sha256: ContentHash | None


@dataclass(frozen=True)
class D1ReviewFindingDisposition:
    finding_id: str
    status: ReviewFindingStatus
    rationale: str
    evidence_sha256: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _closed(ReviewFindingStatus, self.status, "D1ReviewFindingDisposition.status"),
        )


@dataclass(frozen=True)
class D1ReviewAdjudication:
    status: ReviewAdjudicationStatus
    major_findings: tuple[str, ...]
    dispositions: Mapping[str, D1ReviewFindingDisposition]
    execution_receipt_sha256: str
    operator_record_sha256: str
    report_sha256: str
    review_packet_sha256: str
    review_request_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _closed(ReviewAdjudicationStatus, self.status, "D1ReviewAdjudication.status"),
        )


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
