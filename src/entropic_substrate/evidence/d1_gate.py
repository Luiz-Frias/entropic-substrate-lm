"""One-way filed-verdict gate for the D1 ladder."""

from __future__ import annotations

from entropic_substrate.evidence.d1_filing_graph import ValidatedD1Filing
from entropic_substrate.evidence.d1_types import (
    FiledGateAuthorization,
    FiledGateOutcome,
    FiledGateRetirement,
    FiledGateValidNegative,
)
from entropic_substrate.evidence.lifecycle import SuccessorAuthorization


def decide_filed_verdict(filing: ValidatedD1Filing, /) -> FiledGateOutcome:
    """Return a filed decision only after semantic evidence-graph verification."""

    if type(filing) is not ValidatedD1Filing:
        raise TypeError("D1 filed gate requires a verified semantic evidence graph")
    verdict = filing.verdict

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
