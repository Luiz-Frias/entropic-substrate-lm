"""TDD contract for D1 Build Cloud and RunPod operator values."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from d1_launch_admission_fixtures import IMAGE_DIGEST, plan, profiles, start
from entropic_substrate.d1_operator_ports import D1RunPodPort, D1TrainerEvidencePort
from entropic_substrate.d1_operator_values import (
    D1A100FormFactor,
    D1A100ResourceRequest,
    D1ImageBuildRequest,
    D1ImmutableImage,
    D1OperatorLaunchIntent,
    D1PodPlacement,
    D1RuntimeObservation,
    attest_d1_run_start,
)

pytestmark = pytest.mark.unit

_COMMIT = "c" * 40


def _image(tmp_path: Path) -> D1ImmutableImage:
    return D1ImmutableImage(
        repository="registry.example/entropic-substrate-lm",
        digest=IMAGE_DIGEST,
        source_commit=_COMMIT,
    )


def _resource() -> D1A100ResourceRequest:
    return D1A100ResourceRequest(
        accelerator_form_factor=D1A100FormFactor.SXM,
        provider_gpu_type_id="NVIDIA A100-SXM4-80GB",
        accelerator_count=1,
        provider="runpod",
        provider_sku="a100-sxm-80gb",
        region="US-KS-2",
        market="secure-cloud",
        purchase_option=start(plan(profiles()[0], 101)).purchase_option,
    )


def test_build_request_derives_commit_tag_but_launch_identity_is_digest_only(
    tmp_path: Path,
) -> None:
    context = tmp_path / "repo"
    context.mkdir()
    dockerfile = context / "Containerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    request = D1ImageBuildRequest(
        builder="cloud-builder",
        repository="registry.example/entropic-substrate-lm",
        source_commit=_COMMIT,
        context=context,
        dockerfile=dockerfile,
    )
    image = _image(tmp_path)

    assert request.platform == "linux/amd64"
    assert request.build_tag == f"registry.example/entropic-substrate-lm:git-{_COMMIT}"
    assert image.reference == f"registry.example/entropic-substrate-lm@{IMAGE_DIGEST}"
    assert ":git-" not in image.reference


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repository", "registry.example/model:latest", "tag-free"),
        ("repository", "registry.example/model@sha256:" + "0" * 64, "tag-free"),
        ("source_commit", "abcdef0", "full lowercase"),
        ("platform", "linux/arm64", "linux/amd64"),
    ],
)
def test_build_request_refuses_mutable_or_non_target_identity(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    context = tmp_path / "repo"
    context.mkdir()
    dockerfile = context / "Containerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    kwargs: dict[str, object] = {
        "builder": "cloud-builder",
        "repository": "registry.example/model",
        "source_commit": _COMMIT,
        "context": context,
        "dockerfile": dockerfile,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        D1ImageBuildRequest(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("form", "model"),
    [
        (D1A100FormFactor.PCIE, "NVIDIA A100 80GB PCIe"),
        (D1A100FormFactor.SXM, "NVIDIA A100-SXM4-80GB"),
    ],
)
def test_only_explicit_a100_pcie_or_sxm_requests_are_admitted(
    form: D1A100FormFactor, model: str
) -> None:
    assert (
        replace(
            _resource(), accelerator_form_factor=form, provider_gpu_type_id=model
        ).accelerator_form_factor
        is form
    )

    with pytest.raises(ValueError, match="A100"):
        replace(_resource(), provider_gpu_type_id="NVIDIA H100 PCIe")
    with pytest.raises(ValueError, match="form factor"):
        replace(
            _resource(),
            accelerator_form_factor=D1A100FormFactor.PCIE,
            provider_gpu_type_id="NVIDIA A100-SXM4-80GB",
        )


def test_launch_intent_binds_registered_plan_image_quote_and_termination_guard(
    tmp_path: Path,
) -> None:
    run = plan(profiles()[0], 101)
    intent = D1OperatorLaunchIntent(
        plan=run,
        registered_design_digest="f" * 64,
        image=_image(tmp_path),
        resource=_resource(),
        quoted_rate=start(run).quoted_rate,
        resource_activation=start(run).resource_activation,
        execution_segment_index=0,
        resume_checkpoint_sha256=None,
        terminate_after_seconds=7200,
    )

    assert intent.manifest["plan"] == run.plan_manifest
    assert intent.manifest["image_reference"] == _image(tmp_path).reference
    assert intent.manifest["terminate_after_seconds"] == 7200
    assert "secret" not in repr(intent.manifest).lower()


def test_resume_is_an_explicit_next_segment_with_checkpoint_identity(tmp_path: Path) -> None:
    run = plan(profiles()[0], 101)
    base = D1OperatorLaunchIntent(
        plan=run,
        registered_design_digest="f" * 64,
        image=_image(tmp_path),
        resource=_resource(),
        quoted_rate=start(run).quoted_rate,
        resource_activation=start(run).resource_activation,
        execution_segment_index=1,
        resume_checkpoint_sha256="9" * 64,
        terminate_after_seconds=7200,
    )

    assert base.segment_start_mode.value == "resume"
    with pytest.raises(ValueError, match="resume checkpoint"):
        replace(base, resume_checkpoint_sha256=None)
    with pytest.raises(ValueError, match="fresh segment"):
        replace(base, execution_segment_index=0)


def test_measured_placement_and_runtime_construct_exact_start_provenance(
    tmp_path: Path,
) -> None:
    run = plan(profiles()[0], 101)
    expected = start(run)
    intent = D1OperatorLaunchIntent(
        plan=run,
        registered_design_digest="f" * 64,
        image=_image(tmp_path),
        resource=_resource(),
        quoted_rate=expected.quoted_rate,
        resource_activation=expected.resource_activation,
        execution_segment_index=0,
        resume_checkpoint_sha256=None,
        terminate_after_seconds=7200,
    )
    placement = D1PodPlacement(
        pod_id="pod-123",
        accelerator_form_factor=D1A100FormFactor.SXM,
        accelerator_model="NVIDIA A100-SXM4-80GB",
        accelerator_count=1,
        provider="runpod",
        provider_sku="a100-sxm-80gb",
        region="US-KS-2",
        market="secure-cloud",
        purchase_option=expected.purchase_option,
        pod_created_at=expected.pod_created_at,
    )
    runtime = D1RuntimeObservation(
        pod_id="pod-123",
        trainer_started_at=expected.trainer_started_at,
        cuda_version="12.6",
        driver_version="560.35.03",
        pytorch_version="2.13.0",
        compiler_enabled=False,
        compiler_version="disabled",
        trainer_monotonic_origin_ns=123,
    )

    observed = attest_d1_run_start(intent=intent, placement=placement, runtime=runtime)

    assert observed.manifest == expected.manifest
    assert runtime.trainer_monotonic_origin_ns == 123
    with pytest.raises(ValueError, match="monotonic"):
        replace(runtime, trainer_monotonic_origin_ns=-1)
    with pytest.raises(ValueError, match="resource placement"):
        attest_d1_run_start(
            intent=intent,
            placement=replace(placement, provider_sku="unexpected-sku"),
            runtime=runtime,
        )
    with pytest.raises(ValueError, match="pod identity"):
        attest_d1_run_start(
            intent=intent,
            placement=placement,
            runtime=replace(runtime, pod_id="another-pod"),
        )


def test_trainer_runtime_observation_belongs_to_the_trainer_port() -> None:
    assert "runtime" not in D1RunPodPort.__dict__
    assert "observe_runtime" in D1TrainerEvidencePort.__dict__
