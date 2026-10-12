"""Pure RunPod billing rows map to the shared provider evidence algebra."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from d1_launch_admission_fixtures import plan, profiles, start
from entropic_substrate.ladder.capture.resource.resource_billing import (
    ProviderBillingPartial,
    ProviderBillingPending,
)
from entropic_substrate.ladder.run.closure.d1_runpod_billing_parser import (
    RunpodBillingSchemaError,
    parse_runpodctl_billing_observation,
)

pytestmark = pytest.mark.unit


def _start():
    return start(plan(profiles()[0], 101))


def test_parser_preserves_exact_row_bytes_and_provider_partial_fields() -> None:
    raw = json.dumps(
        [
            {
                "amount": 0.0017170833889394999,
                "diskSpaceBilledGb": 20,
                "podId": "pod-123",
                "time": "2026-08-26 17:00:00",
                "timeBilledMs": 24726,
            }
        ]
    )

    observed = parse_runpodctl_billing_observation(
        raw,
        pod_id="pod-123",
        start=_start(),
        checked_at=datetime(2026, 8, 26, 18, tzinfo=UTC),
    )

    assert type(observed) is ProviderBillingPartial
    assert observed.billed_duration_seconds == Decimal("24.726")
    assert observed.billed_amount.amount == Decimal("0.0017170833889394999")
    assert observed.observation_sha256 == hashlib.sha256(raw.encode()).hexdigest()


def test_empty_and_invalid_rows_remain_distinct_provider_outcomes() -> None:
    pending = parse_runpodctl_billing_observation(
        "[]",
        pod_id="pod-123",
        start=_start(),
        checked_at=datetime(2026, 8, 26, 18, tzinfo=UTC),
    )
    assert type(pending) is ProviderBillingPending

    with pytest.raises(RunpodBillingSchemaError, match="calendar"):
        parse_runpodctl_billing_observation(
            '[{"amount":0.1,"podId":"pod-123","time":"2026-99-99 99:99:99","timeBilledMs":1}]',
            pod_id="pod-123",
            start=_start(),
            checked_at=datetime(2026, 8, 26, 18, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    "amount_token",
    ["1e9999999999999999999999", "-1e9999999999999999999999", "1e-9999999999999999999999"],
)
def test_decimal_signal_from_an_out_of_range_amount_token_is_a_schema_error(
    amount_token: str,
) -> None:
    raw = (
        f'[{{"amount":{amount_token},"podId":"pod-123",'
        '"time":"2026-08-26 17:00:00","timeBilledMs":1000}]'
    )
    with pytest.raises(RunpodBillingSchemaError, match="not a valid decimal"):
        parse_runpodctl_billing_observation(
            raw,
            pod_id="pod-123",
            start=_start(),
            checked_at=datetime(2026, 8, 26, 18, tzinfo=UTC),
        )
