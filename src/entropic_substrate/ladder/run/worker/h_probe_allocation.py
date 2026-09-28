"""One allocation submission with an attributed, durable watchdog handoff."""

import hashlib
from typing import Protocol

from entropic_substrate.ladder.run.campaign.d1_operator_values import require_d1_pod_id
from entropic_substrate.ladder.run.h_probe_launch import H0ProbeLaunch
from entropic_substrate.ladder.run.h_probe_process import ProbeWatchdogProcess
from entropic_substrate.ladder.run.h_runpod_transport import (
    ProbeAccount,
    ProbePod,
    ProbeProviderError,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker_transport import (
    read_published_probe_request,
)


class ProbeAllocationProvider(Protocol):
    def account(self) -> ProbeAccount: ...
    def check_template(self, launch: H0ProbeLaunch) -> None: ...
    def create_once(self, launch: H0ProbeLaunch) -> ProbePod: ...


class ProbeAllocationHandoffError(RuntimeError):
    """Safe supervision failed; the caller must reconcile this exact attempt.

    Retain the safe response ID for emergency ownership-checked cleanup. Neither
    this exception nor the surviving guardian proves that cleanup succeeded.
    """

    def __init__(self, pod_id: str | None, provider_error: BaseException | None) -> None:
        self.pod_id = pod_id
        self.provider_error = provider_error
        super().__init__("allocation handoff failed; attributed cleanup remains unresolved")


_PROVIDER_DETAIL_MAX_BYTES = 512


def provider_error_code(error: BaseException | None) -> str | None:
    """The stable failure code for provider-failure telemetry, never a free-text message."""
    if error is None:
        return None
    if isinstance(error, ProbeProviderError):
        return error.code
    return type(error).__name__


def _bounded_provider_detail(text: str) -> tuple[str, str]:
    """Bound the stored detail to 512 bytes; the sha256 always covers the full text."""
    encoded = text.encode("utf-8", errors="replace")
    digest = hashlib.sha256(encoded).hexdigest()
    if len(encoded) <= _PROVIDER_DETAIL_MAX_BYTES:
        return text, digest
    return encoded[:_PROVIDER_DETAIL_MAX_BYTES].decode("utf-8", errors="ignore"), digest


def _handoff(
    guard: ProbeWatchdogProcess,
    *,
    pod_id: str | None,
    response_sha256: str | None,
    provider_error: BaseException | None,
) -> None:
    code = provider_error_code(provider_error)
    allocation_uncertain = False
    possible_pod_id: str | None = None
    provider_detail: str | None = None
    provider_detail_sha256: str | None = None
    if provider_error is not None:
        allocation_uncertain = (
            provider_error.allocation_uncertain
            if isinstance(provider_error, ProbeProviderError)
            else True
        )
        possible_pod_id = pod_id
        provider_detail, provider_detail_sha256 = _bounded_provider_detail(str(provider_error))
    try:
        guard.control.write(
            "allocation",
            {
                "schema": "h-probe-allocation-v2",
                "guard_pid": guard.pid,
                "outcome": "created" if provider_error is None else "uncertain",
                "pod_id": pod_id,
                "response_sha256": response_sha256,
                "provider_error_code": code,
                "allocation_uncertain": allocation_uncertain,
                "possible_pod_id": possible_pod_id,
                "provider_detail": provider_detail,
                "provider_detail_sha256": provider_detail_sha256,
            },
            max_bytes=guard.config.max_control_bytes,
        )
    except Exception as error:
        raise ProbeAllocationHandoffError(pod_id, provider_error) from error


def allocate_probe_once(
    guard: ProbeWatchdogProcess,
    provider: ProbeAllocationProvider,
    objects: D1TrainerMailboxObjectPort,
) -> ProbePod:
    """Use only inside an operator-authorized, supervised H0 tiny workflow.

    The caller supplies spending approval and publishes the input graph before
    entering. This rechecks its request commit marker, account, template and live
    guard; it does not grant authority, retry create, or claim workload readiness.
    An uncertain response is handed off conservatively even for an unexpected
    exception. A repeated call cannot pass the immutable create-intent claim.
    """
    if type(guard) is not ProbeWatchdogProcess:
        raise TypeError("allocation requires a supervised watchdog process")
    guard.require_alive()
    launch = guard.launch
    request = launch.request
    marker = D1TrainerMailboxObjectIdentity(
        f"{request.prefix}/request.json", len(request.payload), request.digest
    )
    received = read_published_probe_request(
        objects, marker, max_request_bytes=request.limits.max_object_bytes
    )
    if received.payload != request.payload:
        raise ValueError("published request differs from the supervised launch")
    if guard.ready is None or provider.account().account_id != guard.ready["account_id"]:
        raise ValueError("allocation account differs from the live guardian")
    provider.check_template(launch)
    guard.record_create_intent()
    try:
        pod = provider.create_once(launch)
    except BaseException as error:
        possible_id = error.possible_pod_id if isinstance(error, ProbeProviderError) else None
        if possible_id is not None:
            possible_id = require_d1_pod_id(possible_id)
        _handoff(guard, pod_id=possible_id, response_sha256=None, provider_error=error)
        raise
    _handoff(guard, pod_id=pod.pod_id, response_sha256=pod.response_sha256, provider_error=None)
    try:
        guard.require_alive()
    except Exception as error:
        raise ProbeAllocationHandoffError(pod.pod_id, None) from error
    return pod
