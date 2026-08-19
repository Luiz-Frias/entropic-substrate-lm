"""E1 paired-gap Pareto chart — marginal gap vs cumulative advantage share.

Descriptive addendum to the filed E1 verdict (operator ask 2026-08-18): the
marginal curve is the mean paired gap g(t) = mean_s(random_s − fisher_s)
nats/byte at each eval step; the cumulative curve is the share of total
gap-area banked by step t. The cumulative curve's flattening is the
diminishing-returns readout — it names the step by which most of the init
advantage has accrued. Post-filing: this chart decides nothing.

Reads exclusively through entropic_substrate.telemetry. Colorblind
discipline: Okabe-Ito, linestyle + marker redundancy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from entropic_substrate.telemetry import read_eval_series

OKABE_BLUE = "#0072B2"
OKABE_ORANGE = "#E69F00"


def cumulative_share(gaps: list[float]) -> list[float]:
    """Fraction of the total gap-area accrued at each index (Pareto curve)."""
    if not gaps:
        msg = "empty gap series"
        raise ValueError(msg)
    total = sum(gaps)
    if total == 0:
        msg = "zero total gap — no advantage to apportion"
        raise ValueError(msg)
    shares: list[float] = []
    running = 0.0
    for g in gaps:
        running += g
        shares.append(running / total)
    return shares


def paired_gap_series(
    pairs: list[tuple[str | Path, str | Path]],
) -> tuple[list[int], list[float]]:
    """(eval steps, mean paired gap per step) across (fisher, random) pairs."""
    per_pair: list[dict[int, float]] = []
    for fisher_path, random_path in pairs:
        fisher = {r.step: r.nats_per_byte for r in read_eval_series(fisher_path)}
        random_ = {r.step: r.nats_per_byte for r in read_eval_series(random_path)}
        common = sorted(set(fisher) & set(random_))
        per_pair.append({s: random_[s] - fisher[s] for s in common})
    steps = sorted(set.intersection(*(set(d) for d in per_pair)))
    gaps = [sum(d[s] for d in per_pair) / len(per_pair) for s in steps]
    return steps, gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captured", type=Path, default=Path("results/captured"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--out", type=Path, default=Path("docs/verdicts/assets/e1-gap-pareto.png"))
    args = parser.parse_args()

    pairs: list[tuple[str | Path, str | Path]] = []
    for seed in args.seeds:
        fisher = sorted(args.captured.glob(f"e1-fisher-*-s{seed}.jsonl"))
        random_ = sorted(args.captured.glob(f"e1-random-*-s{seed}.jsonl"))
        if len(fisher) != 1 or len(random_) != 1:
            msg = f"seed {seed}: expected exactly one capture per arm, got {fisher} / {random_}"
            raise FileNotFoundError(msg)
        pairs.append((fisher[0], random_[0]))

    steps, gaps = paired_gap_series(pairs)
    shares = cumulative_share(gaps)
    half_step = next(s for s, share in zip(steps, shares, strict=True) if share >= 0.5)

    fig, ax_marginal = plt.subplots(figsize=(8.5, 5))
    ax_marginal.plot(
        steps,
        [g * 1000 for g in gaps],
        color=OKABE_BLUE,
        linestyle="-",
        marker="o",
        markersize=4,
        label="marginal: mean paired gap (milli-nats/byte)",
    )
    ax_marginal.set_xlabel("training step")
    ax_marginal.set_ylabel("mean paired gap, random − fisher (milli-nats/byte)")
    ax_marginal.grid(True, alpha=0.3)

    ax_share = ax_marginal.twinx()
    ax_share.plot(
        steps,
        [s * 100 for s in shares],
        color=OKABE_ORANGE,
        linestyle="--",
        marker="s",
        markersize=4,
        label="cumulative: share of total gap-area (%)",
    )
    ax_share.set_ylabel("cumulative share of gap-area (%)")
    ax_share.axvline(half_step, color=OKABE_ORANGE, alpha=0.4, linestyle=":")
    ax_share.annotate(
        f"50% of cumulative advantage\nby step {half_step}",
        xy=(half_step, 50),
        xytext=(half_step * 1.6, 38),
        arrowprops={"arrowstyle": "->", "alpha": 0.6},
    )

    lines = ax_marginal.get_lines() + ax_share.get_lines()[:1]
    ax_marginal.legend(lines, [str(ln.get_label()) for ln in lines], loc="upper right")
    ax_marginal.set_title(
        f"E1 stage 1: init advantage is front-loaded (seeds {min(args.seeds)}–{max(args.seeds)})"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out} | 50% share at step {half_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
