"""The CPU solver preflight gates a pod launch on the step-1 Fisher solve.

Why (2026-09-28 ruling): the H0 remote-tiny probes died at prediction step 1
with ``solver_nonconvergence`` because the profile budget (32) was below what
finite-precision CG needs at width 32 (43 to 45 FP32 iterations, measured).
The preflight replays the production solver on the profile's own step-1
inputs and refuses any profile whose budget cannot converge every row. The
verdict reads only the FP32 production replay; textbook FP64 CG is a control
column and never decides. No scientific or spending authority.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tests.test_h_tiny_tracer import _foreign_capture, _remote
from tests.test_sentence_profile import profile

from entropic_substrate.core.latent.dynamics import FisherNumericalError, SolverConfig
from entropic_substrate.core.sentence.consumer import SentenceConsumer
from entropic_substrate.core.sentence.head import SentenceLawHead
from entropic_substrate.core.sentence.law import LawSpec
from entropic_substrate.core.sentence.profile import HProfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from h_solver_preflight import REFUSE_EXIT, assess, main  # tools/ is not a package
from h_solver_trace import CapturedSolve, capture_solves

pytestmark = pytest.mark.unit

LIMIT = 10_000_000
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h0-magnitude-w32-step1-solve.json"


def _tensor(entry: dict[str, list[int] | list[float]]) -> torch.Tensor:
    return torch.tensor(entry["values"], dtype=torch.float32).reshape(entry["shape"])


def _captured_width32(budget: int) -> CapturedSolve:
    """Re-drive the REAL consumer on the captured pod-shaped step-1 inputs."""
    data = json.loads(FIXTURE.read_text())
    law = LawSpec(**data["law"])
    head = SentenceLawHead(law.width, law)
    with torch.no_grad():
        for parameter, value in zip(head.metric_parameters(), data["mean_parameters"], strict=True):
            parameter.copy_(_tensor(value))
    states, residual = _tensor(data["states"]), _tensor(data["residual"])
    proposed = states + residual
    # Fixture fidelity: the consumer's own delta reproduces the captured residual.
    assert torch.equal(proposed - states, residual)
    consumer = SentenceConsumer(
        head, data["carrier"], "fisher", SolverConfig(max_iterations=budget)
    )
    with capture_solves() as solves:
        try:
            consumer(states, proposed)
        except FisherNumericalError as error:
            assert error.code == "solver_nonconvergence"
    (solve,) = solves
    return solve


def test_captured_width32_solve_refuses_budget_32_and_passes_budget_64() -> None:
    refused = assess([_captured_width32(32)])
    assert refused["verdict"] == "REFUSE"
    assert refused["budget"] == 32
    assert refused["max_iterations_used"] == 32
    assert refused["worst_residual_ratio"] > 1
    # Measured on this Mac 2026-09-28: 19 of 20 rows above tolerance, ratio ~739.8.
    assert sum(not row["converged"] for row in refused["rows"]) >= 1

    passed = assess([_captured_width32(64)])
    assert passed["verdict"] == "PASS"
    assert passed["budget"] == 64
    # Measured on this Mac 2026-09-28: 45 FP32 production iterations.
    assert 32 < passed["max_iterations_used"] <= 64
    assert passed["worst_residual_ratio"] <= 1
    assert all(row["converged"] for row in passed["rows"])
    # The FP64 textbook control needs more than the width here too (33 to 37 measured).
    assert passed["max_textbook_iterations"] > 32
    assert passed["textbook_unreached_rows"] == 0


def test_rows_report_production_iterations_residual_and_textbook_control() -> None:
    report = assess([_captured_width32(32)])
    assert len(report["rows"]) == 20
    for index, row in enumerate(report["rows"]):
        assert row["solve"] == 0 and row["row"] == index
        assert 1 <= row["iterations"] <= 32
        assert row["residual_ratio"] == pytest.approx(row["true_residual"] / row["tolerance"])
        assert row["converged"] == (row["true_residual"] <= row["tolerance"])
        assert row["live_outcome"] == "solver_nonconvergence"
        assert isinstance(row["textbook_fp64_iterations"], int)
    assert report["worst_residual_ratio"] == max(r["residual_ratio"] for r in report["rows"])
    assert report["max_iterations_used"] == max(r["iterations"] for r in report["rows"])
    assert report["tolerance"] == {"relative": 1e-5, "absolute": 1e-8}


def test_no_observed_solve_is_never_a_pass() -> None:
    report = assess([])
    assert report["verdict"] == "REFUSE"
    assert report["rows"] == [] and report["max_iterations_used"] is None


def _pod_shaped(tmp_path: Path, budget: int | None = None) -> tuple[Path, Path, HProfile]:
    """A remote tiny profile bound to a foreign capture by its source digest."""
    foreign, _ = _foreign_capture(tmp_path)
    value = _remote(profile())
    if budget is not None:
        value = replace(value, solver=replace(value.solver, max_iterations=budget))
    digest = hashlib.sha256(foreign.read_bytes()).hexdigest()
    value = HProfile.from_manifest(value.manifest | {"source_digest": digest})
    path = tmp_path / "attempt" / "profile.json"
    path.parent.mkdir()
    path.write_text(json.dumps(value.manifest, indent=2))
    return path, foreign, value


def _command(path: Path, foreign: Path) -> int:
    return main(
        ["--profile", str(path), "--corpus", str(foreign), "--max-source-bytes", str(LIMIT)]
    )


def test_command_passes_and_files_the_verdict_beside_the_profile(tmp_path: Path) -> None:
    path, foreign, value = _pod_shaped(tmp_path)
    assert _command(path, foreign) == 0
    verdict = json.loads((path.parent / "preflight.json").read_text())
    assert verdict["schema"] == "h-solver-preflight-v1"
    assert verdict["scientific_authority"] == "none"
    assert verdict["spending_authority"] == "none"
    assert verdict["profile_identity"] == value.identity
    assert verdict["profile_name"] == value.name
    assert verdict["local_profile_identity"] != value.identity
    assert verdict["width"] == value.codec.width
    assert verdict["budget"] == value.solver.max_iterations
    assert verdict["tolerance"] == {"relative": 1e-5, "absolute": 1e-8}
    assert verdict["phase"] == "prediction" and verdict["step"] == 1
    assert verdict["step_status"] == "committed"
    assert verdict["verdict"] == "PASS" and verdict["rows"]
    for key in ("max_iterations_used", "max_textbook_iterations", "worst_residual_ratio"):
        assert key in verdict


def test_command_refuses_with_a_distinct_nonzero_exit(tmp_path: Path) -> None:
    path, foreign, _ = _pod_shaped(tmp_path, budget=1)
    assert REFUSE_EXIT not in (0, 1, 2)
    assert _command(path, foreign) == REFUSE_EXIT
    verdict = json.loads((path.parent / "preflight.json").read_text())
    assert verdict["verdict"] == "REFUSE"
    assert verdict["step_status"] == "solver_nonconvergence"
    assert verdict["max_iterations_used"] == 1
    assert verdict["worst_residual_ratio"] > 1


def test_command_rejects_a_corpus_that_is_not_the_profiles_source(tmp_path: Path) -> None:
    path, foreign, value = _pod_shaped(tmp_path)
    path.write_text(json.dumps(value.manifest | {"source_digest": "0" * 64}))
    with pytest.raises(ValueError, match="source digest"):
        _command(path, foreign)
    assert not (path.parent / "preflight.json").exists()
