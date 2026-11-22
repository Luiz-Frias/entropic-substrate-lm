"""Logger smoke: did the tape record what the code emitted?

One registry (``SERIES_PATHS``) names every emission path a series owns. The
probe pushes one representative event through the real logging substrate and
checks both faces of it:

* the JSON tape holds exactly one row carrying the event, every bound field
  and the rendered sentence as its message;
* the human stderr sink printed that same sentence once. On a pod that stderr
  line is what the provider's system log shows.

A new series joins by adding entries here and flipping the previous series'
entries to ``retired``. Active entries run at pre-push
(``tests/e2e/test_logger_smoke.py``); retired ones run in the ship-it post step
(the ``slow`` test in ``tests/repo/test_logger_smoke.py``). ``just
logger-smoke-image`` runs the active set inside the pinned image.

MEASURED 2026-11-22, one ``host.monitor.tick`` event, current source: host
in-process (arm64, Python 3.12.15) 1 row; pinned amd64 image under qemu
(x86_64, Python 3.12.3) 1 row, identical apart from the timestamp.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

Status = Literal["active", "retired"]
Selection = Literal["active", "retired", "all"]

EXIT_FAILED = 1
EXIT_NOTHING_EXAMINED = 2

DEPLOY_DOC = Path(__file__).resolve().parents[1] / "docs" / "deploy-h100.md"
_PIN = re.compile(r"^(ghcr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64})\s*$", re.MULTILINE)

# Fixed probe payload. The float exercises the vocabulary's millisecond
# rendering; ``pod_name`` leads because ``render`` lists fields in keyword order.
PROBE_FIELDS: dict[str, object] = {"pod_name": "smoke-pod", "step": 7, "elapsed": 1.5}


class Emit(Protocol):
    def __call__(self, level: str, event: str, /, **fields: object) -> None: ...


class Render(Protocol):
    def __call__(self, event: str, /, **fields: object) -> str: ...


@dataclass(frozen=True, slots=True)
class SeriesPath:
    """One emission path: where its vocabulary lives and how it emits.

    ``emitter`` is ``module:function`` for a path with its own helper, or
    ``None`` for a path whose call sites write
    ``logger.bind(event=..., **fields).info(render(...))`` inline.
    """

    series: str
    name: str
    status: Status
    vocabulary: str
    event: str
    emitter: str | None = None


SERIES_PATHS: tuple[SeriesPath, ...] = (
    # Host side: the controller, the stop command and the wave monitor narrate
    # what they do TO a pod, from the operator's machine.
    SeriesPath(
        series="h",
        name="h-host",
        status="active",
        vocabulary="entropic_substrate.ladder.run.worker.h_probe_log_events",
        event="HOST_MONITOR_TICK",
        emitter="entropic_substrate.ladder.run.worker.h_probe_host_log:emit",
    ),
    # In-pod side: ``ladder/run/worker_command.py`` inside the container. Its
    # stderr is the pod's system log; its tape is ``worker-log.jsonl``.
    SeriesPath(
        series="h",
        name="h-pod",
        status="active",
        vocabulary="entropic_substrate.ladder.run.worker_log_events",
        event="WORKER_START",
    ),
)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    name: str
    event: str
    ok: bool
    detail: str


def select(selection: Selection, *, paths: Sequence[SeriesPath] = SERIES_PATHS) -> list[SeriesPath]:
    return [entry for entry in paths if selection in ("all", entry.status)]


def event_of(entry: SeriesPath) -> str:
    value: object = getattr(importlib.import_module(entry.vocabulary), entry.event)
    if not isinstance(value, str):
        raise TypeError(f"{entry.vocabulary}.{entry.event} is not an event name")
    return value


def render_of(entry: SeriesPath) -> Render:
    # cast: a module attribute is untyped at this boundary; the probe's message
    # comparison is the runtime guard on what it returns.
    return cast("Render", importlib.import_module(entry.vocabulary).render)


def emitter_of(entry: SeriesPath) -> Emit:
    if entry.emitter is not None:
        module, _, function = entry.emitter.partition(":")
        # cast: same untyped module boundary; the tape comparison is the guard.
        return cast("Emit", getattr(importlib.import_module(module), function))
    render = render_of(entry)

    def inline(level: str, event: str, /, **fields: object) -> None:
        from entropic_substrate.kernel.runtime.logging import logger

        getattr(logger.bind(event=event, **fields), level)(render(event, **fields))

    return inline


def _mismatch(rows: list[dict[str, object]], stderr: str, event: str, sentence: str) -> str:
    hits = [row for row in rows if row.get("event") == event]
    if len(hits) != 1:
        return f"tape holds {len(hits)} rows for {event}, expected 1"
    row = hits[0]
    lost = sorted(key for key, value in PROBE_FIELDS.items() if row.get(key) != value)
    if lost:
        return f"field(s) missing or changed on the tape: {lost}"
    if row.get("message") != sentence:
        return f"tape message is {row.get('message')!r}, expected {sentence!r}"
    if stderr.count(sentence) != 1:
        return f"stderr shows the sentence {stderr.count(sentence)} times, expected 1"
    return ""


def probe(entry: SeriesPath, tape: Path, *, emit: Emit | None = None) -> ProbeResult:
    """Emit one event through the real sinks and compare both faces of it."""
    from entropic_substrate.kernel.runtime.logging import configure_logging

    event = event_of(entry)
    sentence = render_of(entry)(event, **PROBE_FIELDS)
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            configure_logging(json_path=tape)
            (emit or emitter_of(entry))("info", event, **PROBE_FIELDS)
    finally:
        # The sinks above hold ``captured`` and ``tape``; leave none behind.
        configure_logging(json_path=None)
    lines = tape.read_text(encoding="utf-8").splitlines() if tape.exists() else []
    rows = [cast("dict[str, object]", json.loads(line)) for line in lines]
    detail = _mismatch(rows, captured.getvalue(), event, sentence)
    return ProbeResult(entry.name, event, not detail, detail)


def run(selection: Selection, *, paths: Sequence[SeriesPath] = SERIES_PATHS) -> int:
    chosen = select(selection, paths=paths)
    if not chosen:
        if selection == "retired":
            print("logger smoke: no retired paths")
            return 0
        print(f"logger smoke: nothing examined for {selection!r}", file=sys.stderr)
        return EXIT_NOTHING_EXAMINED
    failed = False
    with tempfile.TemporaryDirectory(prefix="logger-smoke-") as scratch:
        for entry in chosen:
            result = probe(entry, Path(scratch) / f"{entry.name}.jsonl")
            verdict = "PASS" if result.ok else "FAIL"
            print(f"logger smoke: {verdict} {result.name} {result.event} {result.detail}".rstrip())
            failed = failed or not result.ok
    return EXIT_FAILED if failed else 0


def image_ref(doc: Path = DEPLOY_DOC) -> str:
    """The digest-pinned image the deploy doc names; a tag is refused."""
    found = _PIN.findall(doc.read_text(encoding="utf-8"))
    if len(set(found)) != 1:
        raise ValueError(f"{doc} names {len(set(found))} digest-pinned images, expected 1")
    return found[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Logger smoke over the series registry.")
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="probe the selected series paths")
    runner.add_argument("--status", choices=("active", "retired", "all"), default="active")
    commands.add_parser("image-ref", help="print the digest-pinned image from the deploy doc")
    args = parser.parse_args(argv)
    if args.command == "image-ref":
        print(image_ref())
        return 0
    return run(cast("Selection", args.status))


if __name__ == "__main__":
    raise SystemExit(main())
