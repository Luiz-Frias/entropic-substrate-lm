"""CPU preflight: can this profile's solver budget converge its step-1 Fisher solve?

Why (ruling 2026-09-28, lane fix/h0-prediction-step-solver): both H0 remote
tiny probes raised ``solver_nonconvergence`` at prediction step 1. Measured
root cause: the budget ``max_iterations: 32`` equals the width, while
finite-precision CG on the damped Fisher operator at width 32 needs 43 to 45
FP32 iterations. The operator ruled a wave-level budget ladder and that a CPU
preflight gates every pod launch. This is that gate.

What it does, on this host's CPU:

1. Binds the corpus to the profile: the capture's sha256 must equal the
   profile's ``source_digest`` (the pod corpus), else it refuses to run.
2. Reseals that capture in this regime and localizes the profile exactly as
   ``just h-tiny-trace`` does (execution-regime fields only; precision kept).
3. Builds the engine at the profile's seed, runs every declared codec warm-up
   step, begins prediction, and wraps prediction step 1 in ``capture_solves``.
   The live step runs the production solver unchanged, at the profile budget.
4. For every captured solve and row: the production solver's FP32 iterations
   used and true residual vs tolerance (a traced replay that must reproduce the
   live outcome), and the textbook FP64 CG count as a control column.

Verdict: PASS only when at least one solve was observed, every live solve
converged, and every row's true residual is within tolerance at the budget.
REFUSE otherwise. The FP64 control never decides. The verdict file carries no
scientific and no spending authority; a PASS is necessary for a launch, never
sufficient. Exit 0 on PASS, ``REFUSE_EXIT`` on REFUSE, 1 on a tool error.

Run: ``just h-solver-preflight --profile <attempt>/profile.json --corpus
<pod capture>/prepared-corpus.json`` (verdict defaults to
``<attempt>/preflight.json``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from h_solver_trace import CapturedSolve, capture_solves, checked_replay, operator_spectrum

from entropic_substrate.composition.wiring.h_tiny_tracer import (
    localize_profile,
    reseal_foreign_corpus,
)
from entropic_substrate.core.latent.metric import FisherNumericalError
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.kernel.artifacts import normalize_d1_artifact_root
from entropic_substrate.kernel.runtime.logging import configure_logging, logger
from entropic_substrate.ladder.run.execution import SentenceExecution

SCHEMA = "h-solver-preflight-v1"
REFUSE_EXIT = 3
_DEFAULT_LIMIT = 67_108_864


def _decide(outcomes: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    """PASS needs an observed solve, converged live solves, and every row in tolerance."""
    if not rows or any(outcome != "converged" for outcome in outcomes):
        return "REFUSE"
    if not all(row["converged"] for row in rows):
        return "REFUSE"
    return "PASS"


def assess(solves: Sequence[CapturedSolve]) -> dict[str, Any]:
    """Per-row production-solver evidence and the verdict for captured solves."""
    rows: list[dict[str, Any]] = []
    outcomes: list[str] = []
    for index, solve in enumerate(solves):
        trace = checked_replay(solve)
        outcomes.append(solve.live_outcome)
        if trace.final is None:  # raised before the true-residual check
            continue
        final = trace.final
        spectrum = operator_spectrum(
            solve.parameterized_mean_map,
            solve.mean_parameters,
            solve.states,
            solve.residual,
            solve.config,
        )
        for position, (iterations, residual, tolerance, control) in enumerate(
            zip(
                final["iterations"],
                final["true_residual"],
                final["tolerance"],
                spectrum,
                strict=True,
            )
        ):
            rows.append(
                {
                    "solve": index,
                    "row": position,
                    "live_outcome": solve.live_outcome,
                    "iterations": iterations,
                    "true_residual": residual,
                    "tolerance": tolerance,
                    "residual_ratio": residual / tolerance,
                    "converged": residual <= tolerance,
                    "textbook_fp64_iterations": control["reference_cg_iterations"],
                    "condition": control["condition"],
                }
            )
    configs = {solve.config for solve in solves}
    if len(configs) > 1:
        raise RuntimeError("captured solves disagree on the solver configuration")
    config = next(iter(configs), None)
    reached = [r["textbook_fp64_iterations"] for r in rows if r["textbook_fp64_iterations"]]
    return {
        "budget": None if config is None else config.max_iterations,
        "tolerance": None
        if config is None
        else {"relative": config.relative_tolerance, "absolute": config.absolute_tolerance},
        "damping": None if config is None else config.damping,
        "solves_observed": len(solves),
        "solve_outcomes": outcomes,
        "rows": rows,
        "verdict": _decide(outcomes, rows),
        "max_iterations_used": max((r["iterations"] for r in rows), default=None),
        "max_textbook_iterations": max(reached, default=None),
        "textbook_unreached_rows": sum(r["textbook_fp64_iterations"] is None for r in rows),
        "worst_residual_ratio": max((r["residual_ratio"] for r in rows), default=None),
    }


def step_one_solves(
    profile: HProfile, source_root: Path, *, max_source_bytes: int
) -> tuple[list[CapturedSolve], str, object]:
    """Warm up as declared, then capture every Fisher solve of prediction step 1."""
    engine = SentenceExecution(profile, source_root, max_source_bytes=max_source_bytes)
    warmup = profile.schedule.warmup_steps
    for step in range(1, warmup + 1):
        engine.advance()
        if step % 250 == 0 or step == warmup:
            logger.info(f"h-solver-preflight warm-up {step}/{warmup}")
    engine.begin_prediction()
    with capture_solves() as solves:
        try:
            row = engine.advance()
            status, batch = "committed", row["source_batch"]
        except FisherNumericalError as error:
            status = error.code
            batch = None if engine.last_attempt is None else engine.last_attempt["source_batch"]
    return list(solves), status, batch


def run_preflight(
    profile_path: Path, corpus_path: Path, *, max_source_bytes: int
) -> dict[str, Any]:
    profile_bytes = profile_path.read_bytes()
    remote = HProfile.from_manifest(json.loads(profile_bytes))
    corpus_digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    if corpus_digest != remote.source_digest:
        raise ValueError("corpus capture does not match the profile's source digest")
    with tempfile.TemporaryDirectory(prefix="h-solver-preflight-") as work:
        # The D1 readers refuse symlinked roots; macOS temp paths sit under one.
        source = normalize_d1_artifact_root(Path(work).resolve()) / "source"
        source.mkdir()
        artifact = reseal_foreign_corpus(corpus_path, source, max_bytes=max_source_bytes)
        local = localize_profile(remote, source_digest=artifact.content_hash.digest)
        solves, status, batch = step_one_solves(local, source, max_source_bytes=max_source_bytes)
    if any(solve.config != remote.solver for solve in solves):
        raise RuntimeError("a captured solve ran with a solver other than the profile's")
    report = assess(solves)
    solver = remote.solver
    return {
        "schema": SCHEMA,
        "scientific_authority": "none",
        "spending_authority": "none",
        "profile_identity": remote.identity,
        "profile_name": remote.name,
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "local_profile_identity": local.identity,
        "corpus_sha256": corpus_digest,
        "device": local.device,
        "precision": local.precision,
        "seed": local.seed,
        "carrier": local.spine.carrier,
        "law_mode": local.law.mode,
        "width": remote.codec.width,
        "warmup_steps": local.schedule.warmup_steps,
        "phase": "prediction",
        "step": 1,
        "step_status": status,
        "source_batch": batch,
        **report,
        "budget": solver.max_iterations,
        "tolerance": {
            "relative": solver.relative_tolerance,
            "absolute": solver.absolute_tolerance,
        },
        "damping": solver.damping,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="h-solver-preflight", description="gate a pod launch on the step-1 Fisher solve"
    )
    parser.add_argument("--profile", type=Path, required=True, help="attempt profile.json")
    parser.add_argument("--corpus", type=Path, required=True, help="pod prepared-corpus.json")
    parser.add_argument("--verdict", type=Path, default=None, help="default: beside the profile")
    parser.add_argument("--max-source-bytes", type=int, default=_DEFAULT_LIMIT)
    args = parser.parse_args(argv)
    profile_path = cast(Path, args.profile)
    verdict_path = cast(Path | None, args.verdict) or profile_path.parent / "preflight.json"
    verdict = run_preflight(profile_path, args.corpus, max_source_bytes=args.max_source_bytes)
    verdict_path.write_text(json.dumps(verdict, indent=1, allow_nan=False) + "\n")
    print(
        f"{verdict['verdict']} {verdict['profile_name']} budget={verdict['budget']}"
        f" max_iterations_used={verdict['max_iterations_used']}"
        f" max_textbook={verdict['max_textbook_iterations']}"
        f" worst_ratio={verdict['worst_residual_ratio']} -> {verdict_path}"
    )
    return 0 if verdict["verdict"] == "PASS" else REFUSE_EXIT


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
