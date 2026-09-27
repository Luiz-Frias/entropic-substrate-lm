"""Allow ``python -m entropic_substrate ...`` as a shorthand for the CLI.

P13-ROOT-MAIN: this is a REAL module, not a generator-emitted shim -- the
generator retires with the eleven old-namespace directories and __legacy/ in
this commit. No ``sys.modules`` rebind, no ``__legacy`` involvement. This is
the one root-level file the namespace scan admits (P13-NAMESPACE-RULE:
``__main__`` is an accepted segment) and the one root-level file this phase
does NOT delete.
"""

from __future__ import annotations

from entropic_substrate.composition.entrypoints.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
