"""Shared values for D1 registered-design and launch-admission tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from d1_verdict_v2_fixtures import decision_contract
from entropic_substrate.kernel.evidence_class.resource_provenance import (
    RESOURCE_TELEMETRY_SCHEMA_VERSION,
    BillingRoundingPolicy,
    BillingUnit,
    Money,
    PurchaseOption,
    QuotedRate,
    RateScope,
    RoundingMode,
    RunStartProvenance,
    SegmentStartMode,
    execution_segment_id,
    unavailable_resource_activation_registration,
)
from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass
from entropic_substrate.kernel.verdict.d1_contract import D1DecisionContract
from entropic_substrate.kernel.verdict.d1_types import D1FamilyIdentity
from entropic_substrate.ladder.register.d1_config import (
    D1LaunchProfile,
    load_d1_profile_catalog,
)
from entropic_substrate.ladder.register.d1_family import D1ExperimentFamily
from entropic_substrate.ladder.register.d1_plan import (
    ConfirmatoryD1RunPlan,
    parse_d1_confirmatory_plan,
)
from entropic_substrate.ladder.register.d1_provenance import D1_TELEMETRY_SCHEMA_VERSION

SEEDS = (101, 103, 107, 109)
PREREG_PATH = "docs/d1-preregistration.md"
PREREG_COMMIT = "a" * 40
PREREG_BLOB = "b" * 64
CODE_COMMIT = "c" * 40
REPOSITORY_DIGEST = "d" * 64
IMAGE_DIGEST = "sha256:" + "e" * 64
DATASET_DIGEST = "1" * 64
SPLIT_DIGEST = "2" * 64
SAMPLER_DIGEST = "3" * 64
STARTED_AT = datetime(2026, 8, 24, 16, 0, tzinfo=UTC)


def profiles() -> tuple[D1LaunchProfile, ...]:
    return load_d1_profile_catalog()


def family() -> D1ExperimentFamily:
    return profiles()[0].family


def contract(
    experiment_family: D1ExperimentFamily | None = None,
    *,
    seeds: tuple[int, ...] = SEEDS,
) -> D1DecisionContract:
    selected = family() if experiment_family is None else experiment_family
    identity = D1FamilyIdentity(
        family_digest=selected.family_digest,
        arm_construction_digests={
            arm.value: selected.draft_for(arm).construction_digest for arm in selected.arms
        },
    )
    base = decision_contract()
    return replace(
        base,
        family_identity=identity,
        registered_constants={"seeds": list(seeds), "primary": {"alpha": 0.025}},
    )


def plan(profile: D1LaunchProfile, seed: int, **overrides: object) -> ConfirmatoryD1RunPlan:
    payload: dict[str, object] = {
        "schema_version": "d1-run-plan-v1",
        "run_class": "confirmatory",
        "profile_name": profile.name,
        "arm": profile.arm.value,
        "family_digest": profile.family.family_digest,
        "construction_digest": profile.draft.construction_digest,
        "logical_run_id": f"d1-confirmatory-{profile.arm.value}-s{seed}",
        "seed": seed,
        "pairing_id": f"d1-pair-s{seed}",
        "code": {"commit": CODE_COMMIT, "repository_digest": REPOSITORY_DIGEST},
        "container": {"image_digest": IMAGE_DIGEST},
        "data": {
            "dataset_name": "enwik8",
            "dataset_digest": DATASET_DIGEST,
            "split_digest": SPLIT_DIGEST,
            "sampler_construction_digest": SAMPLER_DIGEST,
        },
        "evidence": {
            "checkpoint_schema_version": 1,
            "dvc_capture_required": True,
            "required_classes": ["derived_from_measured", "measured"],
            "schema_version": "d1-evidence-v1",
            "telemetry_schema_version": D1_TELEMETRY_SCHEMA_VERSION,
        },
        "preregistration": {
            "path": PREREG_PATH,
            "commit": PREREG_COMMIT,
            "family_digest": profile.family.family_digest,
        },
    }
    payload.update(overrides)
    return parse_d1_confirmatory_plan(profile, payload)


def start(run: ConfirmatoryD1RunPlan, **overrides: object) -> RunStartProvenance:
    rate = QuotedRate(
        amount=Money(amount=Decimal("1.99"), currency="USD"),
        billing_unit=BillingUnit.HOUR,
        rounding_policy=BillingRoundingPolicy(
            increment=1,
            mode=RoundingMode.CEILING,
            minimum_units=1,
        ),
        rate_scope=RateScope.PER_ACCELERATOR,
        source="provider-price-api",
        quote_id="a100-sxm-20260824",
        observed_at=STARTED_AT - timedelta(minutes=10),
        evidence_class=EvidenceClass.MEASURED,
    )
    payload: dict[str, object] = {
        "schema_version": RESOURCE_TELEMETRY_SCHEMA_VERSION,
        "logical_run_id": run.logical_run_id,
        "execution_segment_index": 0,
        "execution_segment_id": execution_segment_id(run.logical_run_id, 0),
        "segment_start_mode": SegmentStartMode.FRESH,
        "resume_checkpoint_sha256": None,
        "accelerator_model": "NVIDIA A100-SXM4-80GB",
        "accelerator_count": 1,
        "provider": "runpod",
        "provider_sku": "a100-sxm-80gb",
        "region": "US-KS-2",
        "market": "secure-cloud",
        "purchase_option": PurchaseOption.ON_DEMAND,
        "image_digest": IMAGE_DIGEST,
        "cuda_version": "12.6",
        "driver_version": "560.35.03",
        "pytorch_version": "2.13.0",
        "compiler_enabled": False,
        "compiler_version": "disabled",
        "resource_activation": unavailable_resource_activation_registration(),
        "quoted_rate": rate,
        "pod_created_at": STARTED_AT - timedelta(minutes=2),
        "trainer_started_at": STARTED_AT,
    }
    payload.update(overrides)
    return RunStartProvenance(**payload)  # type: ignore[arg-type]
