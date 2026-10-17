"""``JsonlSink`` -- the AUTHORITATIVE :class:`~entropic_substrate.ladder.capture.ports.metric_sink.MetricSink` adapter.

Landed at p3b from ``entropic_substrate.sinks:233`` (residue now at
``entropic_substrate.ladder.capture.reader.sinks`` since p8). Owns the exact byte-for-byte
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
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from entropic_substrate.kernel.runtime.logging import logger

__all__ = ["JsonlSink"]


class JsonlSink:
    """AUTHORITATIVE. Byte-identical to ``Trainer.log()``'s pre-port serialisation."""

    __slots__ = ("_finished", "path")

    def __init__(self, path: str | Path) -> None:
        self._finished = False
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
        # FOR-1637: the authoritative sink narrates its own finish (once per
        # instance; CompositeSink finishes it first, so its `.end` is already in
        # the worker log the MLflow sink uploads). Literal event names: this
        # adapter may not import the worker's vocabulary module (layer edge).
        if self._finished:
            return
        self._finished = True
        started = time.perf_counter()
        logger.bind(event="sink.finish.start", sink="jsonl").info("sink.finish.start sink=jsonl")
        seconds = time.perf_counter() - started
        logger.bind(
            event="sink.finish.end", sink="jsonl", settled=True, stranded=0, seconds=seconds
        ).info(f"sink.finish.end sink=jsonl settled=True stranded=0 seconds={seconds:.3f}")
