"""Pins for the E1 gap Pareto/diminishing-returns transform.

The verdict's gap trajectory (front-loaded: +0.0174 at step 500 decaying to
~+0.005 at 20k) invites the cumulative question: how much of the total
paired advantage has accrued by step t? ``cumulative_share`` answers it —
the Pareto curve whose flattening IS the diminishing-returns readout.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from itertools import pairwise
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SPEC = importlib.util.spec_from_file_location(
    "e1_gap_pareto_plot",
    Path(__file__).resolve().parents[1] / "scripts" / "e1_gap_pareto_plot.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["e1_gap_pareto_plot"] = _MOD
_SPEC.loader.exec_module(_MOD)


def test_cumulative_share_is_monotone_and_ends_at_one() -> None:
    shares = _MOD.cumulative_share([4.0, 3.0, 2.0, 1.0])
    assert shares == [0.4, 0.7, 0.9, 1.0]
    assert all(b >= a for a, b in pairwise(shares))


def test_cumulative_share_refuses_empty_and_zero_total() -> None:
    with pytest.raises(ValueError, match="empty"):
        _MOD.cumulative_share([])
    with pytest.raises(ValueError, match="zero"):
        _MOD.cumulative_share([0.0, 0.0])


def _write_run(path: Path, offsets: list[float]) -> Path:
    records: list[dict[str, object]] = [{"event": "run_start", "run": "x", "step": 0}]
    for i, off in enumerate(offsets, start=1):
        records.append(
            {
                "event": "eval",
                "run": "x",
                "step": 500 * i,
                "loss": 1.0,
                "nats_per_byte": 1.0 + off,
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_paired_gap_series_aligns_on_eval_steps(tmp_path: Path) -> None:
    fisher = _write_run(tmp_path / "f.jsonl", [0.0, 0.0])
    random = _write_run(tmp_path / "r.jsonl", [0.02, 0.01])
    steps, gaps = _MOD.paired_gap_series([(fisher, random)])
    assert steps == [500, 1000]
    assert gaps == [pytest.approx(0.02), pytest.approx(0.01)]
