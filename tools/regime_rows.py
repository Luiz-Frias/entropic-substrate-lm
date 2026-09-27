#!/usr/bin/env python3
"""Compute the H composed identity rows for a given regime.

Conductor tooling (p0 ruling 2026-09-19). The two composed H identities hash a
regime key next to regime-independent file digests:

- ``h-checkpoint-reader-v2`` keys on ``(python_version, torch.__version__)``
  (``ladder/run/checkpoint.py::_reader_identity``);
- ``h-source-reader-v3`` keys on ``(python_version, pysbd version)``
  (``ladder/capture/capture.py::_reader_identity``).

``content_identity`` is ``sha256(json.dumps(parts))``, a pure function of the
strings, so a row for a regime this machine cannot run (CI's Linux torch build)
is computable here from MEASURED regime strings plus the file digests at this
tip. The regime strings are inputs, never guessed by this tool: measure them
(a dry-run resolution on CI's platform, or CI's own log) and pass them in.

Control: fed the local regime strings, both functions must reproduce the repo's
own identity functions exactly (pinned by tests/test_regime_rows.py). Run from
the repo root with ``PYTHONPATH=src`` or ``uv run --no-sync``:

    uv run --no-sync python tools/regime_rows.py --python 3.12.14 --torch 2.13.0+cpu --pysbd 0.3.4

p4 (h-tracer) RE-PIN (ruling Q1, 2026-09-22): ``checkpoint_row`` used to
hand-copy checkpoint's own glob + explicit-extra-paths logic. That glob
RAISES once the reader lives under ``ladder/run`` (ruling "checkpoint glob",
brief p4-h-tracer.md), so p4 rewrote the reader to an explicit
``_READER_MODULES`` tuple -- the same idiom ``capture.py`` already used.
``checkpoint_row`` now mirrors that tuple dynamically, the same way
``source_row`` already mirrors ``capture._READER_MODULES``: one shape, no
second hand-copied member list to drift out of sync with the reader it
mirrors.
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

from entropic_substrate.core.sentence.records import content_identity
from entropic_substrate.kernel.artifacts import read_d1_artifact_bytes
from entropic_substrate.ladder.capture import capture
from entropic_substrate.ladder.run import checkpoint


def _digests(reader_modules: tuple[str, ...], *, label: str) -> list[str]:
    digests = []
    for name in reader_modules:
        module = import_module(f"entropic_substrate.{name}")
        if module.__file__ is None:
            raise RuntimeError(f"{label} cannot bind module {name}")
        source = Path(module.__file__)
        _, _, digest = read_d1_artifact_bytes(source.parent, source.name, label=label)
        digests.append(f"{name}:{digest}")
    return digests


def checkpoint_row(python: str, torch_version: str) -> str:
    """Mirror of checkpoint._reader_identity() with the regime strings injected."""
    digests = _digests(
        checkpoint._READER_MODULES,  # pyright: ignore[reportPrivateUsage] -- mirrors the private identity on purpose
        label="checkpoint reader",
    )
    return content_identity(("h-checkpoint-reader-v2", python, torch_version, *digests))


def source_row(python: str, pysbd_version: str) -> str:
    """Mirror of capture._reader_identity() with the regime strings injected."""
    digests = _digests(
        capture._READER_MODULES,  # pyright: ignore[reportPrivateUsage] -- mirrors the private identity on purpose
        label="H source reader",
    )
    return content_identity(("h-source-reader-v3", python, pysbd_version, *digests))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--python", required=True, help="platform.python_version() string of the target regime"
    )
    parser.add_argument(
        "--torch", required=True, help="torch.__version__ string of the target regime"
    )
    parser.add_argument(
        "--pysbd", required=True, help="pysbd distribution version of the target regime"
    )
    args = parser.parse_args(argv)
    print(
        f"_CHECKPOINT_REGIME_TABLE[({args.python!r}, {args.torch!r})] = {checkpoint_row(args.python, args.torch)!r}"
    )
    print(
        f"_SOURCE_REGIME_TABLE[({args.python!r}, {args.pysbd!r})] = {source_row(args.python, args.pysbd)!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
