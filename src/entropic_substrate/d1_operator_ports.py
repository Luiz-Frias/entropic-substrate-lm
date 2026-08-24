"""Ports shared by the D1 cloud-build and accelerator operator adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from entropic_substrate.d1_operator_values import (
    D1ImageBuildRequest,
    D1ImmutableImage,
)


class D1OperatorTransportError(RuntimeError):
    """Fail-loud external command or provider response defect."""

    def __init__(self, *, operation: str, subject: str, detail: str) -> None:
        self.operation = operation
        self.subject = subject
        self.context = MappingProxyType({"operation": operation, "subject": subject})
        super().__init__(f"D1 operator {operation} for {subject}: {detail}")


@dataclass(frozen=True, slots=True)
class D1CommandResult:
    """Captured text streams from one shell-free argv invocation."""

    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        if type(self.stdout) is not str or type(self.stderr) is not str:
            raise TypeError("D1 command result streams must be strings")


class D1CommandRunner(Protocol):
    """Injected process boundary; implementations must execute argv without a shell."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        stdin: str | None,
    ) -> D1CommandResult: ...


class D1BuildCloudPort(Protocol):
    def build_and_push(self, request: D1ImageBuildRequest) -> D1ImmutableImage: ...
