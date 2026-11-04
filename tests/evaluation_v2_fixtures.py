"""Hand-built ``h-predictive-evaluation-v2`` events (FOR-1704, L4d).

Plain dicts, no torch and no reader member: the analysis-side tests (extractor, E-LM
projection, aggregate loader) need a v2 record whose arithmetic is self-consistent but
whose bytes do not depend on the engine that writes it. The v2 shape is the one the door
ruling fixes: the q rows stay at their v1 keys, the prior and the mixture live under
``proposals`` (named from the closed vocabulary), the diagnostics and the latent-usage
rows sit beside them, and the record names its latent-usage operand.
"""

from __future__ import annotations

import math
from typing import Any

V1 = "h-predictive-evaluation-v1"
V2 = "h-predictive-evaluation-v2"
PROPOSALS = ("q", "prior", "mixture")
OPERANDS = ("posterior_mean", "draw_average")
ZERO_MASS_GAP = "zero_importance_weight_mass"


def row(log_estimate: float, ess: float, mcse: float = 0.1) -> dict[str, Any]:
    return {
        "log_estimate": log_estimate,
        "effective_sample_size": ess,
        "likelihood_relative_mcse": mcse,
        "gap": None,
    }


def gap_row() -> dict[str, Any]:
    return {
        "log_estimate": None,
        "effective_sample_size": None,
        "likelihood_relative_mcse": None,
        "gap": ZERO_MASS_GAP,
    }


def summary(rows_by_k: dict[int, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {str(k): rows for k, rows in rows_by_k.items()}


def example(
    *,
    source_bytes: list[int],
    q: dict[int, list[dict[str, Any]]],
    prior: dict[int, list[dict[str, Any]]] | None = None,
    mixture: dict[int, list[dict[str, Any]]] | None = None,
    sd: dict[str, dict[int, list[float | None]]] | None = None,
    prior_log_radius: list[float] | None = None,
    q_log_radius: list[float] | None = None,
    cosine: list[float] | None = None,
    usage: tuple[list[float], list[float]] | None = None,
) -> dict[str, Any]:
    """One example. ``proposals``/``diagnostics``/``latent_usage`` appear only when given."""
    out: dict[str, Any] = {
        "source_batch": "batch",
        "source_examples": [],
        "source_bytes": source_bytes,
        "boundary_symbols": [1] * len(source_bytes),
        "by_sample_count": summary(q),
        "log_weights": [],
        "decoder_log_prob": [],
        "prior_log_prob": [],
        "proposal_log_prob": [],
    }
    if prior is not None or mixture is not None:
        proposals: dict[str, Any] = {}
        if prior is not None:
            proposals["prior"] = {
                "proposal": "prior",
                "by_sample_count": summary(prior),
                "log_weights": [],
                "decoder_log_prob": [],
                "prior_log_prob": [],
            }
        if mixture is not None:
            proposals["mixture"] = {
                "proposal": "mixture",
                "allocation": "bernoulli_half",
                "component": [],
                "by_sample_count": summary(mixture),
                "log_weights": [],
                "decoder_log_prob": [],
                "q_log_prob": [],
                "prior_log_prob": [],
            }
        out["proposals"] = proposals
    if sd is not None or cosine is not None or prior_log_radius is not None:
        out["diagnostics"] = {
            "prior_log_radius_mean": prior_log_radius,
            "q_log_radius_mean": q_log_radius,
            "prior_q_cosine": cosine,
            "log_weight_sd": {
                name: {str(k): values for k, values in by_k.items()}
                for name, by_k in (sd or {}).items()
            },
        }
    if usage is not None:
        out["latent_usage"] = {"q_decoder_log_prob": usage[0], "null_decoder_log_prob": usage[1]}
    return out


def event(
    examples: list[dict[str, Any]],
    *,
    schema: str = V2,
    step: int = 1000,
    operand: str | None = "posterior_mean",
    wall_seconds: float = 4.5,
    cuda_seconds: float | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": schema,
        "meaning": "importance_weighted_log_marginal_estimate",
        "run_identity": "x" * 64,
        "step": step,
        "examples": examples,
        "wall_seconds": wall_seconds,
        "cuda_seconds": cuda_seconds,
    }
    if operand is not None:
        out["latent_operand"] = operand
    return out


def finite_log_weight_sd(values: list[float]) -> float | None:
    """The unbiased (ddof = 1) sample sd of one k-prefix; None when any weight is -inf."""
    if any(not math.isfinite(v) for v in values):
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
