"""One-way filed-verdict gate for the D1 ladder."""

from __future__ import annotations

from entropic_substrate.evidence.d1_types import (
    FiledGateAuthorization,
    FiledGateOutcome,
    FiledGateRetirement,
    FiledGateValidNegative,
)
from entropic_substrate.evidence.d1_validator import ValidatedD1Verdict
from entropic_substrate.evidence.lifecycle import SuccessorAuthorization


def decide_filed_verdict(verdict: ValidatedD1Verdict, /) -> FiledGateOutcome:
    """Return the filed human decision without inspecting or recomputing data."""

    match verdict.successor.authorization:
        case SuccessorAuthorization.AUTHORIZED:
            return FiledGateAuthorization(
                rung=verdict.rung,
                eligible_strategies=verdict.successor.eligible_strategies,
            )
        case SuccessorAuthorization.VALID_NEGATIVE:
            return FiledGateValidNegative(rung=verdict.rung, verdict=verdict.verdict)
        case SuccessorAuthorization.RETIRED:
            return FiledGateRetirement(rung=verdict.rung)
