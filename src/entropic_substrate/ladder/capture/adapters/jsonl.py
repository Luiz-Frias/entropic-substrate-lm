"""``JsonlSink`` -- the AUTHORITATIVE :class:`~entropic_substrate.ladder.capture.ports.metric_sink.MetricSink` adapter.

Landed at p3b from ``entropic_substrate.sinks:233`` (quarantined
``entropic_substrate.__legacy.sinks``). Owns the exact byte-for-byte
serialisation of ``results/*.jsonl`` that :mod:`entropic_substrate.telemetry`
(the E3 execution gate) parses -- same dict, same
``json.dumps(record, sort_keys=True) + "\\n"``, same append-only file as the
pre-port implementation, so a run through this sink is byte-identical to one
that never went through a port. A write failure here (disk full,
permissions) is a defect, not a degraded-mode event: it propagates, never
gets softened.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["JsonlSink"]


class JsonlSink:
    """AUTHORITATIVE. Byte-identical to ``Trainer.log()``'s pre-port serialisation."""

    __slots__ = ("path",)

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        # DVC owns checkpoint artifacts (`dvc add`, downstream of training);
        # the JSONL sink carries metrics only.
        return None

    def finish(self) -> None:
        return None
