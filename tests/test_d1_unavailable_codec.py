"""Gap-bearing machine outputs retain the exact registered population."""

import json

import pytest

from entropic_substrate.ladder.analyze.d1_capture import D1AdmittedAnalysisCapture
from entropic_substrate.ladder.analyze.d1_clause_policy import (
    evaluate_d1_clauses,
    parse_d1_clause_policy,
)
from entropic_substrate.ladder.analyze.d1_factorial import compute_d1_factorial
from entropic_substrate.ladder.analyze.d1_machine_analysis import build_d1_aliased_machine_analysis
from entropic_substrate.ladder.analyze.d1_machine_analysis_codec import (
    decode_d1_machine_analysis_declaration,
)

from .test_d1_aliased_analysis import _aliases, _diagnostics
from .test_d1_factorial_analysis import _authority
from .test_d1_unavailable_factorial import cosine_case

pytestmark = pytest.mark.unit


def gap_packet(gap_step: int = 2500):
    contract, design, cells = cosine_case(gap_step=gap_step)
    factorial = compute_d1_factorial(
        design=design,
        decision_contract=contract,
        cells=cells,
        exclusion_authority=_authority(design),
    )
    clauses = evaluate_d1_clauses(
        policy=parse_d1_clause_policy(contract),
        decision_contract=contract,
        factorial=factorial,
        cells=cells,
    )
    admitted = D1AdmittedAnalysisCapture(
        design.design_digest, design.decision_contract_digest, "e" * 64, cells, (), (), ()
    )
    return build_d1_aliased_machine_analysis(
        design=design,
        decision_contract=contract,
        admitted=admitted,
        factorial=factorial,
        clauses=clauses,
        diagnostics=_diagnostics(),
        aliases=_aliases(),
    )


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def test_v3_gap_results_roundtrip_without_excluding_any_seed() -> None:
    packet = gap_packet()
    decoded = decode_d1_machine_analysis_declaration(packet.canonical_json.encode())
    assert decoded == packet
    value = decoded.manifest
    assert value["schema_version"] == "d1-aliased-machine-analysis-v3"
    assert len(value["included_seeds"]) == 4 and value["excluded_seeds"] == []
    rows = [row for row in value["contrasts"] if "unavailable" in row]
    assert rows and all(
        row["estimate"] is None and row["paired_estimates"][0] is None for row in rows
    )
    value["schema_version"] = "d1-aliased-machine-analysis-v2"
    with pytest.raises(ValueError):
        decode_d1_machine_analysis_declaration(canonical(value))


def test_gap_outside_window_stays_on_historical_finite_wire() -> None:
    packet = gap_packet(0)
    assert packet.manifest["schema_version"] == "d1-aliased-machine-analysis-v2"
    assert decode_d1_machine_analysis_declaration(packet.canonical_json.encode()) == packet


def test_null_estimate_requires_typed_cell_witness() -> None:
    value = gap_packet().manifest
    row = next(row for row in value["contrasts"] if "unavailable" in row)
    del row["unavailable"]
    with pytest.raises(ValueError):
        decode_d1_machine_analysis_declaration(canonical(value))


def test_binding_rejects_gap_witness_reassigned_to_another_seed() -> None:
    from entropic_substrate.ladder.review.d1_machine_analysis_binding import _bind_unavailable

    value = gap_packet().manifest
    row = next(row for row in value["contrasts"] if "unavailable" in row)
    _bind_unavailable(value, row, tuple(value["included_seeds"]))
    row["unavailable"]["cells"][0]["seed"] = value["included_seeds"][1]
    with pytest.raises(ValueError, match="witnesses"):
        _bind_unavailable(value, row, tuple(value["included_seeds"]))
