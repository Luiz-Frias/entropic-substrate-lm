"""Allow ``python -m entropic_substrate ...`` as a shorthand for the CLI."""

from __future__ import annotations

from entropic_substrate.composition.entrypoints.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
