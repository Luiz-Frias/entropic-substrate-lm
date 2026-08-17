"""Okabe–Ito colorblind-safe palette + redundant non-color channels.

Luiz (the operator) is red-green colorblind (deutan/protan-type): every
semantic distinction on a chart in this repo carries a channel beyond hue —
linestyle, marker, or a direct label — never color alone, and luminance
contrast (not hue) is the primary distinguisher.

Palette: Okabe, M. & Ito, K. (2008), "Color Universal Design (CUD)" — the
standard 8-color set verified safe under protanopia/deuteranopia/tritanopia.
The two colors this module pairs by default, blue ``#0072B2`` and vermillion
``#D55E00``, sit at markedly different luminance (relative luminance ≈0.10 vs
≈0.16 by WCAG's coefficients) as well as different hue, so the pairing reads
under a grayscale/dichromacy simulation and not only in full color.
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
# all advance together so two adjacent heads never differ by hue alone even
# when a figure has more heads than distinct Okabe-Ito colors are comfortable
# to tell apart at a glance — the linestyle/marker cycle repeats in lockstep
# with color at that point, so the pair (color, linestyle+marker) stays a
# meaningful redundant channel rather than color carrying it alone.
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
_HEAD_MARKER_CYCLE: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")


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
