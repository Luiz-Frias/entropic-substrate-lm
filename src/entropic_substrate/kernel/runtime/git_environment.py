"""Process-local Git context must not override an explicitly selected repository."""

from collections.abc import Mapping


def isolated_git_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Return a fresh child environment following the existing exit-gate convention.

    Keep non-Git settings, but not caller repository, index, object-store or
    injected Git configuration. Never mutate the caller's environment.
    """
    return {key: value for key, value in environ.items() if not key.startswith("GIT_")}
