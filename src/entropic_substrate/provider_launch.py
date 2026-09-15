"""Scientific-plan-independent provider allocation values."""

from dataclasses import dataclass

from entropic_substrate.d1_operator_values import D1A100ResourceRequest, D1ImmutableImage
from entropic_substrate.experiments.d1_primitives import require_nonempty_string


@dataclass(frozen=True, slots=True)
class PodLaunchRequest:
    """Immutable worker command and bounded placement; never scientific authority."""

    logical_run_id: str
    image: D1ImmutableImage
    resource: D1A100ResourceRequest
    terminate_after_seconds: int
    worker_command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        require_nonempty_string(self.logical_run_id, label="provider logical run")
        if (
            type(self.image) is not D1ImmutableImage
            or type(self.resource) is not D1A100ResourceRequest
        ):
            raise TypeError("provider launch requires typed immutable image and resource")
        if type(self.terminate_after_seconds) is not int or self.terminate_after_seconds < 1:
            raise ValueError("provider launch requires a positive termination deadline")
        command = self.worker_command
        if type(command) is not tuple or len(command) < 3 or command[:2] != ("python", "-m"):
            raise ValueError("provider command must invoke a pinned Python worker module")
        if command[2] not in {
            "entropic_substrate.d1_trainer_mailbox_command",
            "entropic_substrate.d_ladder_worker_command",
        }:
            raise ValueError("provider command must select a known scientific worker")
        if any(
            type(item) is not str
            or not item
            or any(not (c.isalnum() or c in "-_.:/=") for c in item)
            for item in command
        ):
            raise ValueError("worker command contains unsafe shell tokens")
        if type(self.environment) is not tuple or len(dict(self.environment)) != len(
            self.environment
        ):
            raise ValueError("worker environment pins must be unique")
        for key, value in self.environment:
            if key not in {
                "D_LADDER_CODE_COMMIT",
                "D_LADDER_IMAGE_DIGEST",
                "D_LADDER_SOURCE_GIT_BUNDLE_KEY",
            }:
                raise ValueError("only non-secret worker provenance pins are allowed")
            if type(value) is not str or not value or any(c.isspace() for c in value):
                raise ValueError("worker provenance pin is invalid")
