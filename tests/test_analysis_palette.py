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

from entropic_substrate.analysis.palette import OKABE_ITO

pytestmark = pytest.mark.unit

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
