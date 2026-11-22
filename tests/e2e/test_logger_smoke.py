"""Logger smoke: every active series path records what it emits.

Lives in ``tests/e2e`` so the pre-push stage runs it with no extra hook: one
event through the real logging substrate, one row on the tape, one human line
on stderr. ``scripts/logger_smoke.py`` owns the registry and the probe.
"""

from __future__ import annotations

from pathlib import Path

import logger_smoke as smoke


def test_every_active_series_path_records_what_it_emits(tmp_path: Path) -> None:
    active = smoke.select("active")
    assert active, "no active series path: the smoke would examine nothing"
    for entry in active:
        result = smoke.probe(entry, tmp_path / f"{entry.name}.jsonl")
        assert result.ok, f"{entry.name}: {result.detail}"
