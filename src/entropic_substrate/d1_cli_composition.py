"""Production composition root for the public D1 trainer commands."""

from __future__ import annotations

from dataclasses import dataclass

from entropic_substrate.d1_cli import (
    D1CliApplication,
    D1CliCommand,
    D1CliCommandService,
    D1CliInvocation,
    D1CliServiceUnavailable,
)
from entropic_substrate.d1_inspect_service import D1InspectCliService
from entropic_substrate.d1_observability import AuthoritativeJournalOnlyObserver
from entropic_substrate.d1_trainer_cli_services import (
    D1FreshTrainerCliService,
    D1ResumeTrainerCliService,
)
from entropic_substrate.experiments.d1_config import load_d1_profile_catalog

__all__ = ["build_default_d1_cli_application"]


@dataclass(frozen=True, slots=True)
class _LazyTrainerCliService:
    command: D1CliCommand

    def execute(self, invocation: D1CliInvocation) -> dict[str, object]:
        if invocation.command is not self.command:
            raise ValueError("lazy D1 trainer service received the wrong command")
        catalog = (*load_d1_profile_catalog(), *load_d1_profile_catalog(tiny=True))
        if self.command is D1CliCommand.RUN_FRESH:
            return dict(D1FreshTrainerCliService(catalog=catalog).execute(invocation))
        if self.command is D1CliCommand.RUN_RESUME:
            return dict(
                D1ResumeTrainerCliService(
                    catalog=catalog,
                    observer=AuthoritativeJournalOnlyObserver(),
                ).execute(invocation)
            )
        raise ValueError("lazy D1 trainer service command is unsupported")


def build_default_d1_cli_application() -> D1CliApplication:
    """Compose real local trainers and loud gaps for unconnected services."""
    services: dict[D1CliCommand, D1CliCommandService] = {
        command: D1CliServiceUnavailable(
            f"{command.value} composition service is unavailable; "
            "the production adapter has not been connected"
        )
        for command in D1CliCommand
    }
    services[D1CliCommand.RUN_FRESH] = _LazyTrainerCliService(D1CliCommand.RUN_FRESH)
    services[D1CliCommand.RUN_RESUME] = _LazyTrainerCliService(D1CliCommand.RUN_RESUME)
    services[D1CliCommand.INSPECT] = D1InspectCliService()
    return D1CliApplication(services)
