"""The solver-trace harness observes the production CG solve without changing it.

Every assertion here is about fidelity: tracing returns the untraced result
bitwise, the recorded iterations are the solver's own, a capture leaves training
unchanged, and a moved anchor fails loudly instead of recording nothing.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from tests.test_sentence_execution import engine

from entropic_substrate.core.latent import dynamics
from entropic_substrate.core.latent.dynamics import SolverConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from h_solver_trace import (  # tools/ is not a package
    analyze,
    anchor_lines,
    capture_solves,
    operator_spectrum,
    traced_solve,
)

pytestmark = pytest.mark.unit


type Mapping = Callable[[torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor]


def _linear() -> Mapping:
    def mapping(x: torch.Tensor, values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return values[0].to(x.dtype) @ x

    return mapping


def _problem(scale: float, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    width = 12
    q, _ = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=torch.float64))
    singular = torch.logspace(0, math.log10(scale), width, dtype=torch.float64)
    matrix = (q * singular) @ q.T
    states = torch.randn(3, width, generator=generator)
    residual = torch.randn(3, width, generator=generator)
    return matrix.to(torch.float32), states, residual


def _solve(
    matrix: torch.Tensor, states: torch.Tensor, residual: torch.Tensor
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    mapping = _linear()
    call = (
        lambda x: mapping(x, (matrix,)),
        states,
        residual,
        SolverConfig(),
    )
    options = {"parameterized_mean_map": mapping, "mean_parameters": (matrix,)}
    return call, options


def test_tracing_returns_the_untraced_solution_bitwise() -> None:
    matrix, states, residual = _problem(10.0)
    call, options = _solve(matrix, states, residual)
    plain = dynamics.inverse_fisher_residual(*call, **options)
    traced, trace = traced_solve(dynamics.inverse_fisher_residual, *call, **options)
    assert traced is not None and trace.outcome == "converged"
    torch.testing.assert_close(traced.update, plain.update, atol=0, rtol=0)
    assert trace.final is not None
    assert trace.final["true_residual"] == plain.diagnostics.residual_norm.tolist()
    assert trace.final["iterations"] == plain.diagnostics.iterations.tolist()
    # One record per executed loop iteration: the busiest row's count.
    assert len(trace.iterations) == int(plain.diagnostics.iterations.max())
    assert trace.iterations[0]["active_rows"] == 3
    assert all(item["denominator_all_positive"] for item in trace.iterations)


def test_tracing_reports_the_same_nonconvergence_the_solver_raises() -> None:
    matrix, states, residual = _problem(1e4)
    call, options = _solve(matrix, states, residual)
    with pytest.raises(dynamics.FisherNumericalError, match="solver_nonconvergence"):
        dynamics.inverse_fisher_residual(*call, **options)
    result, trace = traced_solve(dynamics.inverse_fisher_residual, *call, **options)
    assert result is None and trace.outcome == "solver_nonconvergence"
    assert trace.final is not None
    assert any(
        r > t for r, t in zip(trace.final["true_residual"], trace.final["tolerance"], strict=True)
    )


def test_a_function_without_the_anchor_lines_fails_loudly() -> None:
    def impostor(*args: object, **kwargs: object) -> None:
        return None

    with pytest.raises(RuntimeError, match="anchors not found"):
        anchor_lines(impostor)
    with pytest.raises(RuntimeError, match="anchors not found"):
        traced_solve(impostor)
    assert set(anchor_lines(dynamics.inverse_fisher_residual)) == {"iteration", "final"}


def test_spectrum_matches_the_analytic_damped_operator() -> None:
    # damping != 1 so that dropping the damping factor cannot pass unnoticed.
    matrix, states, residual = _problem(30.0)
    damping = 0.5
    rows = operator_spectrum(_linear(), (matrix,), states, residual, SolverConfig(damping=damping))
    singular = torch.linalg.svdvals(matrix.to(torch.float64))
    expected = float((singular.max() ** 2 + damping) / (singular.min() ** 2 + damping))
    assert len(rows) == 3
    for row in rows:
        assert row["lambda_min"] == pytest.approx(float(singular.min() ** 2 + damping))
        assert row["condition"] == pytest.approx(expected, rel=1e-9)
        assert row["fp32_floor_at_exact"] >= 0


def test_reference_cg_matches_the_production_iteration_count_in_float64() -> None:
    """The dense textbook-CG control counts iterations the way the solver does."""
    matrix, states, residual = _problem(30.0)
    config = SolverConfig(max_iterations=64)
    mapping = _linear()
    matrix64 = matrix.to(torch.float64)
    solved = dynamics.inverse_fisher_residual(
        lambda x: mapping(x, (matrix64,)),
        states.to(torch.float64),
        residual.to(torch.float64),
        config,
        parameterized_mean_map=mapping,
        mean_parameters=(matrix64,),
    )
    rows = operator_spectrum(mapping, (matrix,), states, residual, config)
    assert [r["reference_cg_iterations"] for r in rows] == solved.diagnostics.iterations.tolist()
    assert all(r["reference_ratio_at_budget"] <= 1 for r in rows)
    tight = operator_spectrum(mapping, (matrix,), states, residual, SolverConfig(max_iterations=2))
    assert all(r["reference_ratio_at_budget"] > 1 for r in tight)


def test_capture_leaves_training_unchanged_and_replays_the_live_solve(tmp_path: Path) -> None:
    runs = []
    captured = []
    for index in range(2):
        (tmp_path / str(index)).mkdir()
        run = engine(tmp_path / str(index))
        run.advance()
        run.advance()
        run.begin_prediction()
        if index == 0:
            row = run.advance()
        else:
            with capture_solves() as solves:
                row = run.advance()
            captured = solves
        runs.append(row)
    for key in ("objective", "aggregate_norm", "decoder_ce", "future_nll_per_dimension"):
        assert runs[0][key] == runs[1][key], key
    assert runs[0]["gradients"] == runs[1]["gradients"]
    assert captured and all(s.live_outcome == "converged" for s in captured)
    report = analyze(captured[0])
    assert report["carrier"] == "magnitude"
    assert report["fp32"]["outcome"] == "converged" and report["fp64"]["outcome"] == "converged"
    assert len(report["spectrum"]) == len(report["direction_preimage_norm"])
