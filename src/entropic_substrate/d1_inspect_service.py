"""Application service for bounded inspection of one local D1 journal artifact."""

from __future__ import annotations

from collections.abc import Mapping

from entropic_substrate.d1_artifact_snapshot import D1ArtifactSnapshotError
from entropic_substrate.d1_cli import (
    D1CliCommand,
    D1CliInvalidEvidence,
    D1CliInvocation,
)
from entropic_substrate.d1_inspect_codec import (
    D1InspectCodecError,
    d1_inspect_result_to_mapping,
)
from entropic_substrate.d1_inspect_projection import compose_d1_inspect_view
from entropic_substrate.d1_inspect_request import d1_inspect_request_from_mapping
from entropic_substrate.d1_journal_snapshot import read_d1_local_journal_snapshot
from entropic_substrate.evidence.d1_telemetry_values import D1TelemetryGap

__all__ = ["D1InspectCliService"]


def _invalid(error: Exception) -> D1CliInvalidEvidence:
    return D1CliInvalidEvidence(f"D1 inspect evidence was refused: {error}")


class D1InspectCliService:
    """Resolve, snapshot, project, and encode one operator-visible regular file."""

    def execute(self, invocation: D1CliInvocation) -> Mapping[str, object]:
        if type(invocation) is not D1CliInvocation:
            raise TypeError("D1 inspect service requires D1CliInvocation")
        if invocation.command is not D1CliCommand.INSPECT:
            error = ValueError("D1 inspect service received the wrong command")
            raise _invalid(error) from error
        try:
            request = d1_inspect_request_from_mapping(dict(invocation.payload))
        except (TypeError, ValueError) as error:
            raise _invalid(error) from error
        target = request.journal_path
        if not target.is_absolute():
            target = invocation.manifest_path.parent / target
        try:
            snapshot = read_d1_local_journal_snapshot(
                target,
                max_bytes=request.max_journal_bytes,
                expected_logical_run_id=request.expected_logical_run_id,
                expected_family_digest=request.expected_family_digest,
                expected_run_class=request.expected_run_class,
                expected_plan_digest=request.expected_plan_digest,
                allow_incomplete=request.allow_incomplete,
            )
            view = compose_d1_inspect_view(
                snapshot,
                evaluation_limit=request.evaluation_limit,
            )
            return d1_inspect_result_to_mapping(view)
        except (
            D1ArtifactSnapshotError,
            D1InspectCodecError,
            D1TelemetryGap,
            TypeError,
            ValueError,
        ) as error:
            raise _invalid(error) from error
