"""One verified trainer stage/start transaction shared by operator lanes."""

from __future__ import annotations

from entropic_substrate.__legacy.d1_operator_evidence import (
    D1StagedTrainerRequestReceipt,
    D1TrainerStartReceipt,
    d1_runtime_observation_sha256,
)
from entropic_substrate.__legacy.d1_operator_values import D1RuntimeObservation
from entropic_substrate.kernel.evidence_class.resource_provenance import RunStartProvenance
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.register.d1_plan import D1RunPlan
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    build_d1_trainer_request,
    d1_trainer_command_sha256,
    d1_trainer_request_sha256,
    serialize_d1_trainer_request,
)
from entropic_substrate.ladder.run.values.d1_operator_ports import D1TrainerEvidencePort


def stage_and_start_d1_trainer(
    trainer: D1TrainerEvidencePort,
    pod_id: str,
    plan: D1RunPlan,
    start: RunStartProvenance,
    runtime: D1RuntimeObservation,
    attempt_sha256: str,
) -> ArtifactIdentity:
    runtime_sha256 = d1_runtime_observation_sha256(runtime)
    origin = D1TrainerWallOrigin(runtime.trainer_started_at, runtime.trainer_monotonic_origin_ns)
    request = build_d1_trainer_request(plan, start, origin)
    request_bytes = serialize_d1_trainer_request(request)
    request_sha256 = d1_trainer_request_sha256(request)
    live = request
    request_identity = type(live), id(live.start), id(live.origin), id(live.locations)
    staged = trainer.stage_inputs(pod_id, request=request)
    if d1_runtime_observation_sha256(runtime) != runtime_sha256:
        raise ValueError("trainer runtime changed while staging its request")
    if type(staged) is not D1StagedTrainerRequestReceipt:
        raise TypeError("staged trainer request must be factory-attested")
    staged.require_factory_attested()
    if (
        (type(live), id(live.start), id(live.origin), id(live.locations)) != request_identity
        or serialize_d1_trainer_request(request) != request_bytes
        or d1_trainer_request_sha256(request) != request_sha256
        or staged.request is not request
        or staged.request_sha256 != request_sha256
        or staged.artifact.content_hash.algorithm != "sha256"
        or staged.artifact.content_hash.digest != request_sha256
    ):
        raise ValueError("staged trainer request differs from controller request")
    start_snapshot = (
        pod_id,
        attempt_sha256,
        staged.request_sha256,
        origin.started_at,
        runtime_sha256,
        d1_trainer_command_sha256(request),
    )
    started = trainer.start_trainer(pod_id)
    if type(started) is not D1TrainerStartReceipt:
        raise TypeError("trainer start must return a factory-attested receipt")
    started.require_factory_attested()
    if (
        d1_runtime_observation_sha256(runtime) != runtime_sha256
        or serialize_d1_trainer_request(request) != request_bytes
        or (
            started.pod_id,
            started.attempt_sha256,
            started.request_sha256,
            started.trainer_started_at,
            started.runtime_sha256,
            started.command_sha256,
        )
        != start_snapshot
        or not started.process_id.strip()
        or started.process_id.strip() != started.process_id
    ):
        raise ValueError("trainer start receipt differs from staged request or runtime")
    return staged.artifact
