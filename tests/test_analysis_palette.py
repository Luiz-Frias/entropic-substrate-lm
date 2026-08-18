"""Literal-pin guard for the Okabe-Ito palette hex values.

Every consuming test (tests/test_analysis_charts.py) asserts a plotted color
against ``entropic_substrate.analysis.palette.OKABE_ITO`` imported from the
SAME module under test — self-referential, so it would pass even if the hex
codes drifted or were typo'd, as long as ``charts.py`` and the test both read
the same (possibly wrong) constant. This file is the one place the dict is
checked against literals that do NOT come from the module: the eight
published Okabe, M. & Ito, K. (2008) Color Universal Design (CUD) hex codes,
transcribed by hand from the palette's own docstring citation.

Mirrors the literal-pin idiom in tests/test_analysis_series.py
(``test_paired_verdict_stage1_matches_the_preregistered_t_value``): asserting
against ``series.T_0_9875[4]`` would be tautological; asserting against the
literal ``3.4954`` from the prereg doc is not. Same shape here.

Guard-teeth (measured, not asserted): with ``OKABE_ITO["blue"]`` swapped for
the unsafe pure red ``#FF0000`` (the FLIP #6 mutation from the review that
flagged this gap), `pytest tests/test_analysis_palette.py -m unit` goes RED
on this file while the rest of the previously-green 625-test suite reported
no other failures tied to the swap — i.e. this is the ONLY test in the repo
that would have caught it. Verified 2026-08-17, reverted before commit.
"""

from __future__ import annotations

import pytest

from entropic_substrate.analysis.palette import OKABE_ITO, head_style

pytestmark = pytest.mark.unit

# configs/e0-byte.yaml, e0-bpe.yaml, e1-fisher.yaml, e1-random.yaml, e2.yaml,
# e3.yaml, e3-anchor.yaml, e3-vantage.yaml all pin n_heads: 12 — only
# tiny.yaml (a smoke-test config, not a real arm) uses fewer. head_style()'s
# distinctness guarantee is evaluated against THIS number, not an arbitrary
# round one, because chart 5/6 (fig_e3_kappa_trajectories /
# fig_e3_kappa_divergence_vs_loss_gap) render exactly this many per-head
# lines on every real E3 run.
_REAL_E3_N_HEADS = 12

# The eight published Okabe-Ito CUD hex codes, transcribed independently of
# palette.py — do NOT import these from entropic_substrate.analysis.palette.
_PUBLISHED_CUD_HEX: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}


def test_okabe_ito_matches_the_published_cud_hex_codes_exactly() -> None:
    """Pins every key AND value — a drifted entry, a swapped color, or a
    silently added/removed key must all fail this test."""
    assert OKABE_ITO == _PUBLISHED_CUD_HEX


def test_okabe_ito_has_no_red_or_green_pure_hues() -> None:
    """Sanity companion to the literal pin above: the colorblind rule this
    whole module exists for is 'never red-vs-green alone', so guard against
    the two unsafe pure hues landing in the palette by any future edit."""
    assert "#FF0000" not in OKABE_ITO.values()
    assert "#00FF00" not in OKABE_ITO.values()


def test_head_style_stays_distinct_across_the_real_e3_head_count() -> None:
    """MEASURED regression this pins: at the repo's actual n_heads=12,
    ``head_style(8)`` was a COMPLETE clone of ``head_style(0)`` — identical
    color AND linestyle AND marker, not merely a color collision. The color
    cycle (len 8) and the marker cycle (len 8, pre-fix) share a period, so
    every 8-index step realigned all three channels at once; the colorblind
    operator would have seen literally no way to tell head 0 from head 8 on
    fig_e3_kappa_trajectories for every real E1/E2/E3 config (all pin
    n_heads: 12).

    Verified directly: ``head_style(0) == SeriesStyle(color='#0072B2',
    linestyle='-', marker='o', label='head 0')`` and pre-fix
    ``head_style(8)`` reproduced that exact triple (color, linestyle,
    marker) — only the label differed. Asserts distinctness on the
    STYLE triple alone (excluding the always-distinct numeric label), for
    every pair of the 12 real heads.
    """
    styles = [head_style(i) for i in range(_REAL_E3_N_HEADS)]
    triples = [(s.color, s.linestyle, s.marker) for s in styles]
    assert len(set(triples)) == _REAL_E3_N_HEADS, (
        f"head_style() produces only {len(set(triples))} distinct "
        f"(color, linestyle, marker) triples across the real {_REAL_E3_N_HEADS} "
        f"E3 heads — two or more heads would render as complete visual clones"
    )
