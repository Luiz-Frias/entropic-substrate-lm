"""E0 baselines re-plotted on a BYTES-SEEN x-axis (nats/byte vs bytes read).

At matched steps both arms consume batch x seq TOKENS per step, but a GPT-2
BPE token covers a MEASURED bytes/token ratio (3.377739 on the 2026-08-18
enwik8 eval split — never the folklore 4.0), so the BPE arm reads ~3.38x the
bytes of the byte arm over the same 20k steps. This plot is the control for
the obvious rebuttal to any matched-step comparison: it answers "did the
byte arm win only because bytes-seen was uncontrolled?" by showing both
curves against the bytes each actually read.

Reads exclusively through entropic_substrate.telemetry (the admissibility
gate); a file whose final eval record lacks bytes_per_token is a refusal,
never a defaulted 1.0 — both E0 arms are measured to carry the field.

Colorblind discipline (operator is red-green colorblind): Okabe-Ito blue and
orange, plus linestyle AND marker as redundant non-color channels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from entropic_substrate.telemetry import TelemetryGap, final_eval_record, read_eval_series

OKABE_BLUE = "#0072B2"
OKABE_ORANGE = "#E69F00"


def bytes_seen(step: int, tokens_per_step: int, bytes_per_token: float) -> float:
    """Cumulative source bytes consumed by ``step`` optimizer steps."""
    return step * tokens_per_step * bytes_per_token


def build_series(
    path: str | Path, *, tokens_per_step: int
) -> tuple[list[float], list[float], float]:
    """(bytes-seen xs, nats/byte ys, measured bytes/token) for one run."""
    final = final_eval_record(path)
    ratio = final.get("bytes_per_token")
    if not isinstance(ratio, (int, float)):
        msg = f"{path}: final eval record carries no numeric bytes_per_token"
        raise TelemetryGap(msg)
    ratio = float(ratio)
    series = read_eval_series(path)
    xs = [bytes_seen(r.step, tokens_per_step, ratio) for r in series]
    ys = [r.nats_per_byte for r in series]
    return xs, ys, ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--byte", type=Path, required=True, help="e0-byte results JSONL")
    parser.add_argument("--bpe", type=Path, required=True, help="e0-bpe results JSONL")
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=16 * 1024,
        help="batch_size x seq_len from the E0 configs (both arms: 16 x 1024)",
    )
    parser.add_argument("--out", type=Path, default=Path("results/charts/e0-bytes-seen.png"))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for path, label, color, style, marker in (
        (args.byte, "byte-level", OKABE_BLUE, "-", "o"),
        (args.bpe, "GPT-2 BPE", OKABE_ORANGE, "--", "s"),
    ):
        xs, ys, ratio = build_series(path, tokens_per_step=args.tokens_per_step)
        ax.plot(
            xs,
            ys,
            label=f"{label} (measured {ratio:.3f} bytes/token, final {ys[-1]:.4f})",
            color=color,
            linestyle=style,
            marker=marker,
            markevery=max(1, len(xs) // 12),
            markersize=4,
        )
    ax.set_xlabel("source bytes seen")
    ax.set_ylabel("nats/byte (held-out)")
    ax.set_title("E0 baselines at equal BYTES SEEN, not equal steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
