"""Small, fully typed D1 plans and clocks for fresh-trainer integration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from entropic_substrate.config import merge_dicts
from entropic_substrate.evidence.resource_provenance import (
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
from entropic_substrate.experiments.d1_config import d1_configs_dir
from entropic_substrate.experiments.d1_config_profile import (
    D1LaunchScale,
    parse_d1_profile_payload,
    read_d1_profile_mapping,
)
from entropic_substrate.experiments.d1_plan import SmokeD1RunPlan, create_d1_smoke_plan
from entropic_substrate.experiments.d1_primitives import D1Arm, EvidenceClass
from entropic_substrate.experiments.d1_provenance import EvidenceEnvelope

STARTED_AT = datetime(2026, 8, 23, 20, 1, tzinfo=UTC)
ORIGIN_NS = 10_000


class DeterministicTrainerClock:
    """Strictly advancing clock without coupling UTC and monotonic call counts."""

    def __init__(self) -> None:
        self._utc_ticks = 0
        self._monotonic_ticks = 0

    def utc_now(self) -> datetime:
        self._utc_ticks += 1
        return STARTED_AT + timedelta(seconds=self._utc_ticks)

    def monotonic_ns(self) -> int:
        self._monotonic_ticks += 1
        return ORIGIN_NS + self._monotonic_ticks * 1_000_000_000


def micro_plan(
    arm: D1Arm,
    *,
    seed: int = 17,
    save_interval: int = 1,
) -> SmokeD1RunPlan:
    path = d1_configs_dir() / f"d1-{arm.value}.yaml"
    payload = read_d1_profile_mapping(path)
    profile = parse_d1_profile_payload(
        merge_dicts(
            payload,
            {
                "encoder": {
                    "latent_width": 8,
                    "n_layers": 1,
                    "n_heads": 2,
                    "block_size": 8,
                },
                "future_law": {"predictor_hidden_width": 8},
                "data": {
                    "source": "synthetic",
                    "synthetic_bytes": 2048,
                    "batch_size": 1,
                    "seq_len": 4,
                    "val_fraction": 0.2,
                    "num_eval_batches": 1,
                },
                "train": {
                    "max_steps": 2,
                    "warmup_steps": 0,
                    "eval_interval": 1,
                    "log_interval": 1,
                    "save_interval": save_interval,
                    "amp_dtype": "off",
                    "device": "cpu",
                },
            },
        ),
        scale=D1LaunchScale.TINY,
    )
    return create_d1_smoke_plan(
        profile,
        logical_run_id=f"d1-smoke-{arm.value}-seed-{seed}",
        seed=seed,
        pairing_id=f"d1-smoke-pair-{seed}",
        evidence=EvidenceEnvelope(
            schema_version="d1-evidence-v1",
            required_classes=frozenset({EvidenceClass.MEASURED}),
        ),
    )


def run_start(plan: SmokeD1RunPlan) -> RunStartProvenance:
    logical = plan.logical_run_id
    created = STARTED_AT - timedelta(minutes=1)
    return RunStartProvenance(
        schema_version=RESOURCE_TELEMETRY_SCHEMA_VERSION,
        logical_run_id=logical,
        execution_segment_index=0,
        execution_segment_id=execution_segment_id(logical, 0),
        segment_start_mode=SegmentStartMode.FRESH,
        resume_checkpoint_sha256=None,
        accelerator_model="CPU test double",
        accelerator_count=1,
        provider="local-test",
        provider_sku="cpu",
        region="local",
        market="test",
        purchase_option=PurchaseOption.ON_DEMAND,
        image_digest="sha256:" + "a" * 64,
        cuda_version="not-applicable",
        driver_version="not-applicable",
        pytorch_version="test-runtime",
        compiler_enabled=False,
        compiler_version="disabled",
        resource_activation=unavailable_resource_activation_registration(),
        quoted_rate=QuotedRate(
            amount=Money(amount=Decimal("0"), currency="USD"),
            billing_unit=BillingUnit.SECOND,
            rounding_policy=BillingRoundingPolicy(
                increment=1,
                mode=RoundingMode.CEILING,
                minimum_units=1,
            ),
            rate_scope=RateScope.PER_INSTANCE,
            source="local test fixture",
            quote_id="local-zero-rate",
            observed_at=created,
            evidence_class=EvidenceClass.MEASURED,
        ),
        pod_created_at=created,
        trainer_started_at=STARTED_AT,
    )


def output_paths(tmp_path: Path, arm: D1Arm) -> tuple[Path, Path]:
    return tmp_path / f"{arm.value}.jsonl", tmp_path / f"{arm.value}.pt"
