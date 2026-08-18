"""Pins for the E0 bytes-seen re-plot transform.

The E0 comparison at matched STEPS is not matched at BYTES SEEN: both arms
step over batch x seq TOKENS, and a BPE token covers a measured 3.377739
bytes on the enwik8 eval split (2026-08-18 run) — so the BPE arm reads
~3.38x the bytes in the same 20k steps. The obvious rebuttal to "byte-level
wins" is "but did you control bytes seen?" — this transform is the control,
and the x-axis it produces must come from each file's OWN measured ratio,
never a shared constant.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from entropic_substrate.telemetry import TelemetryGap

pytestmark = pytest.mark.unit

_SPEC = importlib.util.spec_from_file_location(
    "e0_bytes_seen_plot",
    Path(__file__).resolve().parents[1] / "scripts" / "e0_bytes_seen_plot.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["e0_bytes_seen_plot"] = _MOD
_SPEC.loader.exec_module(_MOD)


def _write_run(path: Path, bytes_per_token: float | None, steps: list[int]) -> Path:
    records: list[dict[str, object]] = [{"event": "run_start", "run": "x", "step": 0}]
    for step in steps:
        record: dict[str, object] = {
            "event": "eval",
            "run": "x",
            "step": step,
            "loss": 1.0,
            "nats_per_byte": 1.0 / step,
        }
        if bytes_per_token is not None:
            record["bytes_per_token"] = bytes_per_token
        records.append(record)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_bytes_seen_scales_by_the_measured_ratio() -> None:
    assert _MOD.bytes_seen(20_000, 16_384, 1.0) == 327_680_000.0
    assert _MOD.bytes_seen(20_000, 16_384, 3.377739) == pytest.approx(1_106_817_515.52, rel=1e-9)


def test_build_series_uses_the_files_own_ratio(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "bpe.jsonl", 2.0, [500, 1000])
    xs, ys, ratio = _MOD.build_series(run, tokens_per_step=100)
    assert ratio == 2.0
    assert xs == [500 * 100 * 2.0, 1000 * 100 * 2.0]
    assert ys == [1.0 / 500, 1.0 / 1000]


def test_missing_ratio_is_a_refusal_not_a_default(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "no-ratio.jsonl", None, [500])
    with pytest.raises(TelemetryGap):
        _MOD.build_series(run, tokens_per_step=100)
