"""Production composition root for the public D1 trainer commands."""

from __future__ import annotations

from entropic_substrate.d1_cli import (
    D1CliApplication,
    D1CliCommand,
    D1CliCommandService,
    D1CliServiceUnavailable,
)
from entropic_substrate.d1_observability import AuthoritativeJournalOnlyObserver
from entropic_substrate.d1_trainer_cli_services import (
    D1FreshTrainerCliService,
    D1ResumeTrainerCliService,
)
from entropic_substrate.experiments.d1_config import load_d1_profile_catalog

__all__ = ["build_default_d1_cli_application"]


def build_default_d1_cli_application() -> D1CliApplication:
    """Compose real local trainers and loud gaps for unconnected services."""
    catalog = (*load_d1_profile_catalog(), *load_d1_profile_catalog(tiny=True))
    services: dict[D1CliCommand, D1CliCommandService] = {
        command: D1CliServiceUnavailable(
            f"{command.value} composition service is unavailable; "
            "the production adapter has not been connected"
        )
        for command in D1CliCommand
    }
    services[D1CliCommand.RUN_FRESH] = D1FreshTrainerCliService(catalog=catalog)
    services[D1CliCommand.RUN_RESUME] = D1ResumeTrainerCliService(
        catalog=catalog,
        observer=AuthoritativeJournalOnlyObserver(),
    )
    return D1CliApplication(services)
