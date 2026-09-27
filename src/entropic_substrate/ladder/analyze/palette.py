"""Okabe–Ito colorblind-safe palette + redundant non-color channels.

Luiz (the operator) is red-green colorblind (deutan/protan-type): every
semantic distinction on a chart in this repo carries a channel beyond hue —
linestyle, marker, or a direct label — never color alone, and luminance
contrast (not hue) is the primary distinguisher.

Palette: Okabe, M. & Ito, K. (2008), "Color Universal Design (CUD)" — the
standard 8-color set verified safe under protanopia/deuteranopia/tritanopia.

RETRACTION (this docstring, prior revision): claimed blue ``#0072B2`` and
vermillion ``#D55E00`` sit at "relative luminance ≈0.10 vs ≈0.16" — MEASURED
WRONG by the named formula (WCAG 2.x relative luminance: linearize sRGB per
channel, ``L = 0.2126R + 0.7152G + 0.0722B``). Computed directly from the hex
values: blue ``L≈0.1525``, vermillion ``L≈0.2215`` — a WCAG contrast ratio of
``(0.2215+0.05)/(0.1525+0.05)≈1.34:1``, well under the 3:1 non-text minimum.
Luminance alone does NOT make this pairing dichromacy-safe; what does is the
redundant non-color channel every ``*_style()`` helper below attaches
(``fisher_style``: solid line, circle marker; ``control_style``: dashed line,
square marker) plus the direct legend label — the pairing reads under a
grayscale/dichromacy simulation because of linestyle+marker+label, not because
of a luminance gap this palette does not actually have.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "OKABE_ITO",
    "SeriesStyle",
    "control_style",
    "e2_style",
    "e3_style",
    "fisher_style",
    "head_style",
]

OKABE_ITO: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Cycled for per-head trajectories (chart 5/6): color, linestyle, and marker
# all advance so a figure with more heads than distinct Okabe-Ito colors
# comfortably tell apart still keeps every head visually distinct.
#
# RETRACTION (prior revision of this comment): claimed the three cycles
# "advance together... in lockstep... so the pair stays a meaningful
# redundant channel" — MEASURED WRONG. With color len 8, linestyle len 4,
# and (pre-fix) marker len 8, the combined (color, linestyle, marker) triple
# had period lcm(8,4,8)=8, not the "more heads than 8 colors" case the
# comment claimed to cover: head_style(8) reproduced head_style(0)'s triple
# EXACTLY (color, linestyle, AND marker all identical — see
# tests/test_analysis_palette.py::test_head_style_stays_distinct_across_the_real_e3_head_count).
# This is not a hypothetical scale case — every real E1/E2/E3 config
# (configs/e0-*.yaml, e1-*.yaml, e2.yaml, e3*.yaml) pins n_heads: 12, so
# heads 8-11 were complete visual clones of heads 0-3 on every real E3 run.
#
# Fix: the marker cycle is length 7 (one dropped), coprime with the color
# cycle's length 8 (gcd(7,8)=1) — the combined triple's period becomes
# lcm(8,4,7)=56, so no two of the real 12 heads collide on every channel at
# once. Individual channels still repeat sooner (color repeats every 8,
# e.g.) but at least one of {color, linestyle, marker} always differs
# between any two of the 12 real heads, which is what "redundant channel"
# requires — no single channel needs to be unique alone. Concretely: heads 0
# and 8 STILL share both color (index 8%8=0) and linestyle (index 8%4=0)
# post-fix — the marker (index 8%7=1) is the one channel telling them apart,
# by design, not an oversight.
_HEAD_COLOR_CYCLE: tuple[str, ...] = (
    "blue",
    "vermillion",
    "bluish_green",
    "reddish_purple",
    "orange",
    "sky_blue",
    "yellow",
    "black",
)
_HEAD_LINESTYLE_CYCLE: tuple[str, ...] = ("-", "--", "-.", ":")
_HEAD_MARKER_CYCLE: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X")


@dataclass(frozen=True, slots=True)
class SeriesStyle:
    """Color + redundant non-color channels for one plotted series."""

    color: str
    linestyle: str
    marker: str
    label: str


def fisher_style(label: str = "fisher") -> SeriesStyle:
    """The √p Fisher–Rao init arm — solid, blue, circle marker."""
    return SeriesStyle(color=OKABE_ITO["blue"], linestyle="-", marker="o", label=label)


def control_style(label: str = "control") -> SeriesStyle:
    """The uniform-random-unit-vector control arm — dashed, vermillion, square."""
    return SeriesStyle(color=OKABE_ITO["vermillion"], linestyle="--", marker="s", label=label)


def e2_style(label: str = "e2") -> SeriesStyle:
    """The nGPT-backbone (κ=0) baseline — solid, blue, circle marker."""
    return SeriesStyle(color=OKABE_ITO["blue"], linestyle="-", marker="o", label=label)


def e3_style(label: str = "e3") -> SeriesStyle:
    """The κ-attention arm — dashed, orange, triangle marker."""
    return SeriesStyle(color=OKABE_ITO["orange"], linestyle="--", marker="^", label=label)


def head_style(index: int, *, label: str | None = None) -> SeriesStyle:
    """Style for per-head trajectory line ``index`` (chart 5/6)."""
    color_key = _HEAD_COLOR_CYCLE[index % len(_HEAD_COLOR_CYCLE)]
    linestyle = _HEAD_LINESTYLE_CYCLE[index % len(_HEAD_LINESTYLE_CYCLE)]
    marker = _HEAD_MARKER_CYCLE[index % len(_HEAD_MARKER_CYCLE)]
    return SeriesStyle(
        color=OKABE_ITO[color_key],
        linestyle=linestyle,
        marker=marker,
        label=label if label is not None else f"head {index}",
    )
