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

# DN-1643-PORT (2026-10-15): the closed vocabulary of ``finish(status=...)`` and the
# record keys ``annotate`` may not claim. Both are validated by EVERY adapter at method
# entry, before any side effect, by the two helpers below (this module is the lowest
# adapter, so the sibling adapters import them without a cycle).
FINISH_STATUSES = ("FINISHED", "FAILED")
DEFAULT_FINISH_STATUS = "FINISHED"
_RESERVED_ANNOTATION_KEYS = frozenset({"event", "schema"})


def require_finish_status(status: object) -> str:
    """Return ``status`` if it is ``"FINISHED"`` or ``"FAILED"``; refuse anything else.

    Raises ``TypeError`` for a non-string and ``ValueError`` for any other string. A
    refused call is a caller defect, so it raises in every adapter, best-effort or not.
    """
    if type(status) is not str:
        msg = f"finish status must be a str, got {type(status).__name__}"
        raise TypeError(msg)
    if status not in FINISH_STATUSES:
        msg = f"finish status must be one of {FINISH_STATUSES}, got {status!r}"
        raise ValueError(msg)
    return status


def require_annotation(tags: object) -> dict[str, str]:
    """Return a validated copy of an ``annotate`` mapping; refuse anything else.

    Every key and value must be a ``str`` (a non-string raises ``TypeError``), keys must
    be non-blank, and ``event`` / ``schema`` are reserved for the record stream
    (``ValueError``): an annotate row must never be read as a lifecycle or step record.
    """
    if not isinstance(tags, Mapping):
        msg = f"annotate tags must be a mapping, got {type(tags).__name__}"
        raise TypeError(msg)
    checked: dict[str, str] = {}
    for key, value in tags.items():
        if type(key) is not str or type(value) is not str:
            msg = f"annotate tags must map str to str, got {key!r}: {value!r}"
            raise TypeError(msg)
        if not key.strip() or key in _RESERVED_ANNOTATION_KEYS:
            msg = f"annotate tag key {key!r} is blank or reserved ({sorted(_RESERVED_ANNOTATION_KEYS)})"
            raise ValueError(msg)
        checked[key] = value
    return checked


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

    def annotate(self, tags: Mapping[str, str]) -> None:
        """Append one ``{"event": "annotate", **tags}`` record (DN-1643-PORT).

        The record carries no ``schema`` key, so no schema-keyed reader (the H step
        extractor, ``final_h_run_end``) counts it. An empty mapping appends nothing.
        """
        checked = require_annotation(tags)
        if checked:
            self.log({"event": "annotate", **checked})

    def finish(self, *, status: str = DEFAULT_FINISH_STATUS) -> None:
        # DN-1643-PORT: validated at entry, BEFORE the idempotence flag, so a refused
        # call changes no state and a later valid finish still narrates. The status
        # itself is ignored: the authoritative file carries no run status.
        require_finish_status(status)
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
