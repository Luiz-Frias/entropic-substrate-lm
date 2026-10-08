"""D1 command-envelope application boundary.

The legacy CLI owns spelling and display.  This module owns the D1 command
contract: one strict JSON manifest, one injected application service, and one
machine-readable outcome.  It deliberately has no environment-based service
discovery or scientific override path.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, NoReturn, Protocol, cast

from entropic_substrate.kernel.codec.strict_json import strict_loads

__all__ = [
    "D1_CLI_MANIFEST_SCHEMA",
    "D1CliApplication",
    "D1CliCommand",
    "D1CliCommandService",
    "D1CliExitCode",
    "D1CliInvalidEvidence",
    "D1CliInvocation",
    "D1CliOperatorGap",
    "D1CliOutcome",
    "D1CliServiceUnavailable",
]

D1_CLI_MANIFEST_SCHEMA: Final[str] = "d1-cli-command-v1"
_MANIFEST_KEYS: Final = frozenset({"command", "payload", "schema_version"})


class D1CliCommand(StrEnum):
    RUN_FRESH = "d1-run-fresh"
    RUN_RESUME = "d1-run-resume"
    CAMPAIGN_RUN = "d1-campaign-run"
    CAPTURE_PREPARE = "d1-capture-prepare"
    CAPTURE_FINALIZE = "d1-capture-finalize"
    INSPECT = "d1-inspect"


class D1CliExitCode(IntEnum):
    SUCCESS = 0
    OPERATOR_GAP = 20
    INVALID_EVIDENCE = 21
    UNEXPECTED_DEFECT = 70


class D1CliOperatorGap(RuntimeError):
    """Expected missing operator capability or external evidence."""


class D1CliInvalidEvidence(ValueError):
    """Input bytes or a domain service failed evidence admission."""


@dataclass(frozen=True, slots=True)
class D1CliInvocation:
    command: D1CliCommand
    manifest_path: Path
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.command) is not D1CliCommand:
            raise TypeError("D1 CLI invocation command must be nominal")
        if not isinstance(cast(object, self.manifest_path), Path):
            raise TypeError("D1 CLI invocation manifest_path must be a Path")
        if not isinstance(cast(object, self.payload), Mapping):
            raise TypeError("D1 CLI invocation payload must be a mapping")


class D1CliCommandService(Protocol):
    """One application service; live infrastructure belongs behind this port."""

    def execute(self, invocation: D1CliInvocation) -> Mapping[str, object]: ...


class D1CliServiceUnavailable:
    """Explicit placeholder that can never turn missing work into success."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason.strip():
            raise ValueError("unavailable D1 service requires a reason")
        self._reason = reason

    def execute(self, invocation: D1CliInvocation) -> Mapping[str, object]:
        del invocation
        raise D1CliOperatorGap(self._reason)


@dataclass(frozen=True, slots=True)
class D1CliOutcome:
    command: D1CliCommand
    exit_code: D1CliExitCode
    status: str
    payload: Mapping[str, object] | None
    error_type: str | None
    detail: str | None

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "command": self.command.value,
            "detail": self.detail,
            "error_type": self.error_type,
            "exit_code": int(self.exit_code),
            "payload": None if self.payload is None else dict(self.payload),
            "status": self.status,
        }


def _invalid(command: D1CliCommand, error: Exception) -> D1CliOutcome:
    return D1CliOutcome(
        command,
        D1CliExitCode.INVALID_EVIDENCE,
        "invalid_evidence",
        None,
        type(error).__name__,
        str(error),
    )


def _reject(reason: str) -> NoReturn:
    raise D1CliInvalidEvidence(f"D1 command manifest {reason}")


def _read_manifest(command: D1CliCommand, path: Path) -> D1CliInvocation:
    if not isinstance(cast(object, path), Path):
        raise D1CliInvalidEvidence("D1 command manifest path must be a Path")
    lexical = Path(os.path.abspath(path))
    if lexical.is_symlink() or not lexical.is_file():
        raise D1CliInvalidEvidence("D1 command manifest must be a regular non-symlink file")
    try:
        raw = lexical.read_text(encoding="utf-8")
        value = strict_loads(raw, fail=_reject)
    except D1CliInvalidEvidence:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise D1CliInvalidEvidence(
            f"D1 command manifest is not valid UTF-8 JSON: {error}"
        ) from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise D1CliInvalidEvidence("D1 command manifest top level must be a JSON object")
    manifest = cast(dict[str, object], value)
    if frozenset(manifest) != _MANIFEST_KEYS:
        raise D1CliInvalidEvidence("D1 command manifest keys must be exact")
    if manifest["schema_version"] != D1_CLI_MANIFEST_SCHEMA:
        raise D1CliInvalidEvidence(
            f"D1 command manifest schema_version must be {D1_CLI_MANIFEST_SCHEMA!r}"
        )
    if manifest["command"] != command.value:
        raise D1CliInvalidEvidence("D1 command manifest command does not match the subcommand")
    payload = manifest["payload"]
    if type(payload) is not dict or any(type(key) is not str for key in payload):
        raise D1CliInvalidEvidence("D1 command manifest payload must be a JSON object")
    return D1CliInvocation(command, lexical, MappingProxyType(cast(dict[str, object], payload)))


def _checked_payload(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise TypeError("D1 command service must return a string-keyed mapping")
    payload = dict(cast(Mapping[str, object], value))
    json.dumps(payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return MappingProxyType(payload)


class D1CliApplication:
    """Total dispatcher over the six core D1 application services."""

    def __init__(self, services: Mapping[D1CliCommand, D1CliCommandService]) -> None:
        if not isinstance(cast(object, services), Mapping):
            raise TypeError("D1 command services must be a mapping")
        expected = frozenset(D1CliCommand)
        actual = frozenset(services)
        if actual != expected:
            missing = sorted(command.value for command in expected - actual)
            extra = sorted(str(command) for command in actual - expected)
            raise ValueError(f"missing D1 command services: {missing}; unexpected: {extra}")
        self._services = MappingProxyType(dict(services))

    def execute(self, command: D1CliCommand, manifest_path: Path) -> D1CliOutcome:
        if type(command) is not D1CliCommand:
            raise TypeError("D1 command must be a D1CliCommand")
        try:
            invocation = _read_manifest(command, manifest_path)
        except D1CliInvalidEvidence as error:
            return _invalid(command, error)
        try:
            payload = _checked_payload(self._services[command].execute(invocation))
            return D1CliOutcome(
                command,
                D1CliExitCode.SUCCESS,
                "success",
                payload,
                None,
                None,
            )
        except D1CliOperatorGap as error:
            return D1CliOutcome(
                command,
                D1CliExitCode.OPERATOR_GAP,
                "operator_gap",
                None,
                type(error).__name__,
                str(error),
            )
        except D1CliInvalidEvidence as error:
            return _invalid(command, error)
        except Exception as error:
            return D1CliOutcome(
                command,
                D1CliExitCode.UNEXPECTED_DEFECT,
                "unexpected_defect",
                None,
                type(error).__name__,
                str(error) or repr(error),
            )
