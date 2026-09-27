"""A neutral provider launch preserves the existing allocation/cleanup transport."""

import json

import pytest

from entropic_substrate.ladder.run.provider.provider_launch import PodLaunchRequest

from .test_d1_runpod_adapter import _adapter, _create_payload, _intent, _result, _Runner

pytestmark = pytest.mark.unit


def test_neutral_ladder_worker_uses_same_provider_parser_and_deadline():
    intent = _intent()
    runner = _Runner([_result(_create_payload())])
    provider = _adapter(runner)
    request = PodLaunchRequest(
        intent.plan.logical_run_id,
        intent.image,
        intent.resource,
        intent.terminate_after_seconds,
        ("python", "-m", "entropic_substrate.historical.d_ladder.d_ladder_worker_command", "serve"),
        (("D_LADDER_CODE_COMMIT", intent.image.source_commit),),
    )
    result = provider.create_pod(request)
    assert result.pod_id
    argv = runner.calls[0][0]
    assert "--terminate-after" in argv
    assert (
        "entropic_substrate.historical.d_ladder.d_ladder_worker_command serve"
        in argv[argv.index("--docker-args") + 1]
    )
    environment = json.loads(argv[argv.index("--env") + 1])
    assert environment["D_LADDER_CODE_COMMIT"] == intent.image.source_commit
    assert {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"} <= set(environment)


@pytest.mark.parametrize(
    "deadline, command, message",
    [
        (
            0,
            ("python", "-m", "entropic_substrate.historical.d_ladder.d_ladder_worker_command"),
            "positive termination",
        ),
        (60, ("bash",), "pinned Python worker"),
    ],
)
def test_neutral_provider_refuses_unbounded_or_shell_command(deadline, command, message):
    intent = _intent()
    with pytest.raises(ValueError, match=message):
        PodLaunchRequest(
            intent.plan.logical_run_id, intent.image, intent.resource, deadline, command
        )
