"""Blinding contract for the review-packet assembler.

The packet's blinding is only as good as the scrub: ``run_start`` carries
``embedding_init`` and EVERY record carries ``run``, so filename relabeling
alone is theater (measured on the 2026-08-18 captures — see
scripts/assemble_review_packet.py). These tests pin the transform and prove
the banned-substring gate actually fires on a planted leak.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SPEC = importlib.util.spec_from_file_location(
    "assemble_review_packet",
    Path(__file__).resolve().parents[1] / "scripts" / "assemble_review_packet.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["assemble_review_packet"] = _MOD
_SPEC.loader.exec_module(_MOD)


def test_scrub_maps_every_identifying_field() -> None:
    record = {
        "event": "run_start",
        "run": "e1-fisher",
        "config": {
            "name": "e1-fisher",
            "model": {"embedding_init": "fisher_sqrt", "fisher_init_path": "data/f.pt"},
        },
    }
    out = _MOD.scrub_record(record, "A")
    assert out["run"] == "arm-A"
    assert out["config"]["name"] == "arm-A"
    assert out["config"]["model"]["embedding_init"] == "init-A"
    assert "fisher_init_path" not in out["config"]["model"]
    # the original is not mutated — capture files stay canonical
    assert record["config"]["model"]["embedding_init"] == "fisher_sqrt"


def test_scrub_touches_only_arm_fields() -> None:
    record = {"event": "train", "run": "e1-random", "step": 40, "data_order_digest": "abc"}
    out = _MOD.scrub_record(record, "B")
    assert out == {"event": "train", "run": "arm-B", "step": 40, "data_order_digest": "abc"}


def test_banned_scan_fires_on_a_planted_leak(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    clean.write_text('{"run": "arm-A"}\n', encoding="utf-8")
    assert _MOD.banned_scan(clean) == []
    leaky = tmp_path / "leaky.jsonl"
    leaky.write_text('{"run": "arm-A", "note": "Fisher init"}\n', encoding="utf-8")
    assert "fisher" in _MOD.banned_scan(leaky)


def test_build_reader_strips_prose_and_shims_logging() -> None:
    source = (
        '"""Module docstring naming the e3 ladder."""\n'
        "from entropic_substrate.logging import configure_logging, logger\n\n\n"
        "def f() -> int:\n"
        '    """Docstring naming fisher."""\n'
        "    return 1\n"
    )
    reader = _MOD.build_reader(source)
    lowered = reader.lower()
    assert "e3" not in lowered.replace("prose", "")  # header prose is neutral
    assert "fisher" not in lowered
    assert "entropic_substrate" not in reader
    assert "def configure_logging" in reader
    assert "return 1" in reader
