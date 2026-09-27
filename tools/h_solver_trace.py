"""Observe the H inverse-Fisher CG solve per iteration without changing it.

Diagnostic harness, 2026-09-27 (lane fix/h0-prediction-step-solver). Both
remote_tiny H0 probes raised ``solver_nonconvergence`` at prediction step 1.
This tool answers, for a real failing state, what the solve did and why.

Behaviour is never changed; that is the contract every test here pins:

* ``capture_solves`` wraps the consumer's module-level binding of
  ``inverse_fisher_residual`` with a recorder that clones the inputs and then
  calls the ORIGINAL with the ORIGINAL arguments. Training proceeds (or
  raises) exactly as before.
* ``traced_solve`` re-runs the ORIGINAL ``dynamics.inverse_fisher_residual``
  under ``sys.settrace`` and reads the solver's own locals at two anchor lines,
  located by exact source text. No copy of the loop exists, so the observation
  cannot drift from production; if an anchor line changes, tracing fails loudly
  instead of recording nothing.
* ``operator_spectrum`` builds the dense damped operator ``J^T J + lambda I`` per
  position in float64 (width <= a few dozen, so exact ``eigvalsh`` is cheaper and
  stronger than power iteration) and measures the FP32 residual floor: the
  production true-residual expression evaluated in FP32 at the float64-exact
  solution. A floor above the tolerance means no FP32 solver can pass that row.

Run: ``PYTHONPATH=src uv run --no-sync python tools/h_solver_trace.py --trace-dir
results/<h-tiny-trace dir> --steps 1 --out results/<dir>/solver-trace.json``.
The trace directory is one written by ``just h-tiny-trace`` in THIS regime (its
warm-up seal is resumed through the normal checkpoint reader). No authority.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

import torch

from entropic_substrate.core.latent import dynamics
from entropic_substrate.core.latent.dynamics import SolverConfig
from entropic_substrate.core.latent.metric import FisherNumericalError, fisher_action

ITERATION_ANCHOR = "squared, active = next_squared, next_active"
FINAL_ANCHOR = "if bool((true_residual > tolerance).any()):"
_FP32_EPS = float(torch.finfo(torch.float32).eps)

type ParameterizedMap = Callable[[torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor]


def _rows(values: torch.Tensor) -> list[float]:
    return [float(v) for v in values.detach().reshape(-1).tolist()]


def _norms(values: torch.Tensor) -> list[float]:
    return _rows(torch.linalg.vector_norm(values.detach(), dim=-1))


def anchor_lines(function: Callable[..., Any]) -> dict[str, int]:
    """Absolute line numbers of the two observation anchors, found by exact text."""
    lines, start = inspect.getsourcelines(function)
    wanted = {"iteration": ITERATION_ANCHOR, "final": FINAL_ANCHOR}
    found: dict[str, int] = {}
    for offset, text in enumerate(lines):
        for name, anchor in wanted.items():
            if text.strip() == anchor:
                if name in found:
                    raise RuntimeError(f"solver trace anchor {name!r} is ambiguous")
                found[name] = start + offset
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise RuntimeError(f"solver trace anchors not found in source: {missing}")
    return found


@dataclass
class SolveTrace:
    """Per-iteration and final measurements read from the solver's own locals."""

    iterations: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, Any] | None = None
    outcome: str = "unobserved"


def _iteration_record(index: int, local: dict[str, Any]) -> dict[str, Any]:
    active: torch.Tensor = local["active"]
    denominator: torch.Tensor = local["denominator"]
    candidate: torch.Tensor = local["candidate"]
    next_active: torch.Tensor = local["next_active"]
    stepped = denominator.detach()[active]
    return {
        "iteration": index,
        "active_rows": int(active.sum()),
        "residual_norm": _norms(local["remainder"]),
        "tolerance": _rows(local["tolerance"]),
        "denominator_min_active": None if stepped.numel() == 0 else float(stepped.min()),
        "denominator_all_positive": bool((stepped > 0).all()),
        "measured_restart_rows": int(candidate.sum()),
        "restart_still_active_rows": int((candidate & next_active).sum()),
        "next_active_rows": int(next_active.sum()),
    }


def traced_solve(
    function: Callable[..., dynamics.ResidualSolution],
    *args: Any,
    **kwargs: Any,
) -> tuple[dynamics.ResidualSolution | None, SolveTrace]:
    """Call ``function`` unchanged while observing its anchor lines.

    Only the outermost frame of ``function`` is observed; the solve's own
    re-entrant calls (the zero-residual adjoint) are left untraced.
    """
    lines = anchor_lines(function)
    code = function.__code__
    trace = SolveTrace()
    owner: list[FrameType] = []

    def local_trace(frame: FrameType, event: str, arg: object) -> Any:
        if event == "line" and frame is owner[0]:
            if frame.f_lineno == lines["iteration"]:
                trace.iterations.append(
                    _iteration_record(len(trace.iterations) + 1, frame.f_locals)
                )
            elif frame.f_lineno == lines["final"]:
                local = frame.f_locals
                trace.final = {
                    "true_residual": _rows(local["true_residual"]),
                    "tolerance": _rows(local["tolerance"]),
                    "iterations": [int(v) for v in local["counts"].reshape(-1).tolist()],
                    "update_norm": _norms(local["update"]),
                }
        return local_trace

    def global_trace(frame: FrameType, event: str, arg: object) -> Any:
        if event == "call" and frame.f_code is code and not owner:
            owner.append(frame)
            return local_trace
        return None

    previous = sys.gettrace()
    sys.settrace(global_trace)
    try:
        result = function(*args, **kwargs)
    except FisherNumericalError as error:
        trace.outcome = error.code
        return None, trace
    finally:
        sys.settrace(previous)
    trace.outcome = "converged"
    return result, trace


@dataclass
class CapturedSolve:
    """Cloned inputs of one live consumer solve and what the live call did."""

    states: torch.Tensor
    residual: torch.Tensor
    config: SolverConfig
    parameterized_mean_map: ParameterizedMap
    mean_parameters: tuple[torch.Tensor, ...]
    carrier: str
    live_outcome: str = "unobserved"
    live_residual_norm: list[float] | None = None


def _carrier(mean_map: ParameterizedMap) -> str:
    owner = inspect.getclosurevars(mean_map).nonlocals.get("self")
    carrier = getattr(owner, "carrier", None)
    if carrier not in ("sphere", "magnitude"):
        raise RuntimeError("captured solve does not expose its consumer carrier")
    return str(carrier)


@contextmanager
def capture_solves() -> Generator[list[CapturedSolve]]:
    """Record every Fisher consumer solve; each still runs the original solver."""
    from entropic_substrate.core.sentence import consumer as consumer_module

    original = consumer_module.inverse_fisher_residual
    records: list[CapturedSolve] = []

    def recorder(
        mean_map: Callable[[torch.Tensor], torch.Tensor],
        states: torch.Tensor,
        residual: torch.Tensor,
        config: SolverConfig = dynamics.DEFAULT_SOLVER_CONFIG,
        *,
        parameterized_mean_map: ParameterizedMap | None = None,
        mean_parameters: tuple[torch.Tensor, ...] = (),
    ) -> dynamics.ResidualSolution:
        if parameterized_mean_map is None:
            raise RuntimeError("the H consumer always supplies a parameterized mean map")
        entry = CapturedSolve(
            states.detach().clone(),
            residual.detach().clone(),
            config,
            parameterized_mean_map,
            tuple(p.detach().clone() for p in mean_parameters),
            _carrier(parameterized_mean_map),
        )
        records.append(entry)
        try:
            solved = original(
                mean_map,
                states,
                residual,
                config,
                parameterized_mean_map=parameterized_mean_map,
                mean_parameters=mean_parameters,
            )
        except FisherNumericalError as error:
            entry.live_outcome = error.code
            raise
        entry.live_outcome = "converged"
        entry.live_residual_norm = _rows(solved.diagnostics.residual_norm)
        return solved

    consumer_module.inverse_fisher_residual = recorder
    try:
        yield records
    finally:
        consumer_module.inverse_fisher_residual = original


def replay(
    solve: CapturedSolve, dtype: torch.dtype = torch.float32
) -> tuple[dynamics.ResidualSolution | None, SolveTrace]:
    """Re-run the ORIGINAL solver on the captured inputs, traced, at ``dtype``."""
    parameters = tuple(p.to(dtype) for p in solve.mean_parameters)
    mapping = solve.parameterized_mean_map

    def mean(x: torch.Tensor) -> torch.Tensor:
        return mapping(x, parameters)

    return traced_solve(
        dynamics.inverse_fisher_residual,
        mean,
        solve.states.to(dtype),
        solve.residual.to(dtype),
        solve.config,
        parameterized_mean_map=mapping,
        mean_parameters=parameters,
    )


def reference_cg(
    operator: torch.Tensor, rhs: torch.Tensor, tolerance: float, budget: int, cap: int
) -> tuple[int | None, float]:
    """Textbook dense CG (no restart, no compensation): iterations to reach
    ``tolerance`` on the true residual (None beyond ``cap``), and the true
    residual over tolerance after ``budget`` iterations. A control only."""
    update = torch.zeros_like(rhs)
    remainder, direction = rhs.clone(), rhs.clone()
    squared = remainder @ remainder
    reached: int | None = None
    at_budget = float(torch.linalg.vector_norm(rhs)) / tolerance
    for step in range(1, cap + 1):
        applied = operator @ direction
        alpha = squared / (direction @ applied)
        update = update + alpha * direction
        remainder = remainder - alpha * applied
        true = float(torch.linalg.vector_norm(rhs - operator @ update))
        if step == budget:
            at_budget = true / tolerance
        if reached is None and true <= tolerance:
            reached = step
        if reached is not None and step >= budget:
            break
        following = remainder @ remainder
        if following == 0:  # exact solve: every later residual is this one
            at_budget = true / tolerance if step < budget else at_budget
            break
        direction = remainder + (following / squared) * direction
        squared = following
    return reached, at_budget


def operator_spectrum(
    mapping: ParameterizedMap,
    parameters: tuple[torch.Tensor, ...],
    states: torch.Tensor,
    residual: torch.Tensor,
    config: SolverConfig,
) -> list[dict[str, Any]]:
    """Exact per-position spectrum of ``J^T J + damping I``, the FP32 floor, and
    the textbook-CG iteration count this spectrum needs at the solver tolerance."""
    damping = config.damping
    p64 = tuple(p.detach().to(torch.float64) for p in parameters)
    p32 = tuple(p.detach().to(torch.float32) for p in parameters)
    width = states.shape[-1]
    rows: list[dict[str, Any]] = []
    for state, vector in zip(
        states.detach().reshape(-1, width), residual.detach().reshape(-1, width), strict=True
    ):
        x64 = state.to(torch.float64)
        jacobian = torch.func.jacrev(lambda x: mapping(x, p64))(x64)
        operator = jacobian.T @ jacobian + damping * torch.eye(width, dtype=torch.float64)
        values = torch.linalg.eigvalsh(operator)
        rhs64 = (1 + damping) * vector.to(torch.float64)
        exact = torch.linalg.solve(operator, rhs64)
        x32, exact32 = x64.to(torch.float32), exact.to(torch.float32)
        rhs32 = (1 + damping) * vector.to(torch.float32)
        applied = fisher_action(lambda x: mapping(x, p32), x32, exact32)
        floor = torch.linalg.vector_norm(rhs32 - applied - damping * exact32)
        tolerance = config.absolute_tolerance + config.relative_tolerance * float(
            torch.linalg.vector_norm(rhs64)
        )
        reached, at_budget = reference_cg(
            operator, rhs64, tolerance, config.max_iterations, cap=8 * width
        )
        rows.append(
            {
                "lambda_min": float(values[0]),
                "eigenvalues_above_damping_1pct": int((values > 1.01 * damping).sum()),
                "reference_cg_iterations": reached,
                "reference_ratio_at_budget": at_budget,
                "lambda_max": float(values[-1]),
                "condition": float(values[-1] / values[0]),
                "rhs_norm": float(torch.linalg.vector_norm(rhs64)),
                "exact_update_norm": float(torch.linalg.vector_norm(exact)),
                "fp32_floor_at_exact": float(floor),
                "fp32_bound_estimate": _FP32_EPS
                * float(
                    torch.linalg.vector_norm(rhs64) + values[-1] * torch.linalg.vector_norm(exact)
                ),
            }
        )
    return rows


def direction_preimage_norms(solve: CapturedSolve) -> list[float]:
    """``||W x + b||`` feeding the direction normalization (``x`` unit for sphere)."""
    weight, bias = solve.mean_parameters[0], solve.mean_parameters[1]
    states = solve.states.detach().to(torch.float64)
    if solve.carrier == "sphere":
        states = states / torch.linalg.vector_norm(states, dim=-1, keepdim=True)
    raw = states @ weight.detach().to(torch.float64).T + bias.detach().to(torch.float64)
    return _norms(raw)


def _worst(trace: SolveTrace) -> dict[str, Any]:
    final = trace.final
    if final is None:
        return {"row": None, "ratio": None}
    ratios = [r / t for r, t in zip(final["true_residual"], final["tolerance"], strict=True)]
    row = max(range(len(ratios)), key=ratios.__getitem__)
    return {
        "row": row,
        "ratio": ratios[row],
        "true_residual": final["true_residual"][row],
        "tolerance": final["tolerance"][row],
        "iterations": final["iterations"][row],
        "failing_rows": sum(r > 1 for r in ratios),
        "rows": len(ratios),
    }


def analyze(solve: CapturedSolve) -> dict[str, Any]:
    """Replay FP32 (fidelity-checked) and FP64, plus spectrum and floor, per position."""
    fp32, trace32 = replay(solve, torch.float32)
    if trace32.outcome != solve.live_outcome:
        raise RuntimeError(
            f"replay outcome {trace32.outcome!r} differs from live {solve.live_outcome!r}"
        )
    if fp32 is not None and _rows(fp32.diagnostics.residual_norm) != solve.live_residual_norm:
        raise RuntimeError("replay residuals differ bitwise from the live solve")
    _, trace64 = replay(solve, torch.float64)
    spectrum = operator_spectrum(
        solve.parameterized_mean_map,
        solve.mean_parameters,
        solve.states,
        solve.residual,
        solve.config,
    )
    return {
        "carrier": solve.carrier,
        "live_outcome": solve.live_outcome,
        "config": {
            "damping": solve.config.damping,
            "max_iterations": solve.config.max_iterations,
            "relative_tolerance": solve.config.relative_tolerance,
            "absolute_tolerance": solve.config.absolute_tolerance,
        },
        "fp32": {
            "outcome": trace32.outcome,
            "worst": _worst(trace32),
            "final": trace32.final,
            "iterations": trace32.iterations,
        },
        "fp64": {
            "outcome": trace64.outcome,
            "worst": _worst(trace64),
            "final": trace64.final,
            "iterations": trace64.iterations,
        },
        "spectrum": spectrum,
        "direction_preimage_norm": direction_preimage_norms(solve),
    }


def _seal(run: Path, phase: str, step: int) -> Any:
    from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash

    (path,) = sorted(run.glob(f"*-{phase}-{step:08d}-seal.json"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ArtifactIdentity(path.name, ContentHash("sha256", digest))


def trace_prediction(trace_dir: Path, steps: int, *, max_bytes: int) -> list[dict[str, Any]]:
    """Resume the trace's warm-up seal and observe the first ``steps`` prediction steps."""
    from entropic_substrate.core.sentence.profile import HProfile
    from entropic_substrate.ladder.run.execution import SentenceExecution
    from entropic_substrate.ladder.run.seals import resume_seal

    run = trace_dir / "run"
    profile = HProfile.from_manifest(json.loads((run / "profile.json").read_text()))
    engine = SentenceExecution(profile, trace_dir / "source", max_source_bytes=max_bytes)
    resume_seal(
        run, _seal(run, "codec_warmup", profile.schedule.warmup_steps), engine, max_bytes=max_bytes
    )
    engine.begin_prediction()
    reports: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        with capture_solves() as solves:
            try:
                row = engine.advance()
                status, batch = "committed", row["source_batch"]
            except FisherNumericalError as error:
                status = error.code
                batch = None if engine.last_attempt is None else engine.last_attempt["source_batch"]
        for index, solve in enumerate(solves):
            reports.append(
                {
                    "profile": profile.name,
                    "width": profile.codec.width,
                    "phase": "prediction",
                    "step": step,
                    "solve": index,
                    "step_status": status,
                    "source_batch": batch,
                    **analyze(solve),
                }
            )
        if status != "committed":
            break
    return reports


def table(reports: list[dict[str, Any]]) -> str:
    header = (
        "| profile | phase | step | solve | dtype | iterations (worst row) | true residual "
        "| tolerance | kappa max | fp32 floor/tol max | textbook-CG iters needed (max) | verdict |"
    )
    lines = [header, "|" + "---|" * 12]
    for report in reports:
        kappa = max(r["condition"] for r in report["spectrum"])
        floor = max(
            r["fp32_floor_at_exact"] / t
            for r, t in zip(report["spectrum"], report["fp32"]["final"]["tolerance"], strict=True)
        )
        needed = [r["reference_cg_iterations"] for r in report["spectrum"]]
        reference = ">cap" if None in needed else str(max(needed))
        for dtype in ("fp32", "fp64"):
            worst = report[dtype]["worst"]
            lines.append(
                f"| {report['profile']} | {report['phase']} | {report['step']} | {report['solve']}"
                f" | {dtype} | {worst.get('iterations')} | {worst.get('true_residual'):.3e}"
                f" | {worst.get('tolerance'):.3e} | {kappa:.3e} | {floor:.3e}"
                f" | {reference} | {report[dtype]['outcome']} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="h-solver-trace")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=67_108_864)
    args = parser.parse_args(argv)
    reports = trace_prediction(args.trace_dir, args.steps, max_bytes=args.max_bytes)
    args.out.write_text(json.dumps(reports, indent=1, allow_nan=False))
    print(table(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
