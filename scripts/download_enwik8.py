#!/usr/bin/env python3
"""Download enwik8 into ``data/enwik8/`` ready for ``dvc add data/enwik8``.

Usage::

    python scripts/download_enwik8.py [--dest data/enwik8] [--keep-zip]
    dvc add data/enwik8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from entropic_substrate.data.enwik8 import download_enwik8  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Fetch and unpack enwik8; prints the resulting path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "data" / "enwik8")
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="keep enwik8.zip (default: delete it so DVC only tracks the raw file)",
    )
    args = parser.parse_args(argv)

    raw = download_enwik8(args.dest)
    archive = Path(args.dest) / "enwik8.zip"
    if archive.exists() and not args.keep_zip:
        archive.unlink()

    print(f"enwik8 ready at {raw} ({raw.stat().st_size} bytes)")
    print(f"next: dvc add {Path(args.dest).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
