"""Layout-invariant repo-root and source-root resolvers (E-3/DN-8).

The five depth-derived ``Path(__file__).resolve().parents[n]`` anchors this
module replaces do not share one semantic: two resolve to the REPO ROOT
(the directory containing both ``src/`` and ``configs/``) and one resolves
to the SOURCE ROOT (the directory that directly contains the
``entropic_substrate`` package -- one level deeper). This module exposes
both, named, rather than one resolver misapplied at the wrong depth.

Anchored on the package ``__init__``, never on this module's own ``__file__``
depth: ``paths.py`` itself relocates under ``entropic_substrate.kernel.runtime``
at a later phase (``tools/mapping.json``), and a self-depth anchor would
silently return the wrong root the moment that move happens -- reintroducing
the exact defect class this module exists to retire, one level of
indirection later. The import below is a real, static, grimp-visible
``import entropic_substrate`` line (never a string-only ``importlib`` form),
consistent with the ratified shim-form ruling.

Both resolvers are strict: they raise when the package cannot be located and
carry no CWD, environment, or heuristic fallback. Any such fallback belongs
at the calling site (``config.configs_dir()``, ``sinks._default_enwik8_dvc_pointer()``),
never inside these primitives -- a fallback here would let a fail-closed
guard silently accept an unbound tree whose CWD happens to match the bound
repo root.
"""

from __future__ import annotations

from pathlib import Path

import entropic_substrate


class RepoLayoutError(RuntimeError):
    """Raised when the installed package's on-disk layout cannot be resolved."""


def _package_dir() -> Path:
    module_file = entropic_substrate.__file__
    if not module_file:
        raise RepoLayoutError("entropic_substrate.__file__ is unset; cannot resolve package layout")
    package_dir = Path(module_file).resolve().parent
    if package_dir.is_symlink() or not package_dir.is_dir():
        raise RepoLayoutError(
            f"entropic_substrate package directory is not a real directory: {package_dir}"
        )
    return package_dir


def source_root() -> Path:
    """The directory that directly contains the ``entropic_substrate`` package.

    This is the anchor ``analysis/d1_reader_source_closure.py`` requires:
    ``_source_root()`` hard-requires ``<root>/entropic_substrate`` to be a
    real, non-symlink directory. Passing the repo root there raises.
    """
    return _package_dir().parent


def repo_root() -> Path:
    """The repository root: the directory containing ``src/`` and ``configs/``.

    Assumes the standard ``<repo_root>/src/entropic_substrate`` source
    layout; an installed package sitting elsewhere (not under a ``src/``
    directory) is out of scope for this resolver -- callers that must
    tolerate that case keep their own CWD fallback.
    """
    return source_root().parent
