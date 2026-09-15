"""D1 tiny convenience dispatch without inventing its runtime provenance."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping

from entropic_substrate.d1_cli import (
    D1CliApplication,
    D1CliCommand,
    D1CliInvocation,
    D1CliServiceUnavailable,
)
from entropic_substrate.d1_trainer_cli_services import D1FreshTrainerCliService
from entropic_substrate.d1_trainer_request_codec import (
    D1FreshTrainerRequest,
    parse_d1_trainer_request,
)
from entropic_substrate.experiments.d1_config import load_d1_profile_catalog
from entropic_substrate.experiments.d1_primitives import RunClass, canonical_json


class _TinyFreshService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def execute(self, invocation: D1CliInvocation) -> Mapping[str, object]:
        catalog = load_d1_profile_catalog(tiny=True)
        request = parse_d1_trainer_request(
            canonical_json(invocation.payload).encode(), catalog=catalog
        )
        if (
            type(request) is not D1FreshTrainerRequest
            or request.plan.run.run_class is not RunClass.SMOKE
        ):
            raise ValueError("D1 probe requires an excluded fresh smoke manifest")
        if request.plan.profile.name != self.args.exp:
            raise ValueError("D1 probe experiment differs from its manifest")
        if self.args.seed is not None and self.args.seed != request.plan.seed:
            raise ValueError("D1 probe seed differs from its manifest")
        steps = request.plan.profile.draft.execution_protocol.optimization.max_steps
        if self.args.steps is not None and self.args.steps != steps:
            raise ValueError("D1 probe step budget differs from its manifest")
        service = D1FreshTrainerCliService(catalog=catalog, artifact_root=self.args.out_dir)
        return service.execute_request(request, manifest_path=invocation.manifest_path)


def run_d1_probe(args: argparse.Namespace) -> int:
    """Validate one manifest once, then execute the existing real D1 trainer."""
    if args.manifest is None:
        raise ValueError("D1 probe requires --manifest with actual runtime provenance")
    if args.resume is not None:
        raise ValueError("D1 resume uses d1-run-resume and its explicit resume manifest")
    services = {
        command: _TinyFreshService(args)
        if command is D1CliCommand.RUN_FRESH
        else D1CliServiceUnavailable("not selected by this probe")
        for command in D1CliCommand
    }
    outcome = D1CliApplication(services).execute(D1CliCommand.RUN_FRESH, args.manifest)
    print(json.dumps(outcome.manifest, sort_keys=True, allow_nan=False))
    return outcome.exit_code
