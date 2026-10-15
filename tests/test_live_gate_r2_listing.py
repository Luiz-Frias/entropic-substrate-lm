"""Teeth for the live gate's R2 key probe (``tests/test_mlflow_live.py``).

MEASURED 2026-10-15 (slice-4 validator, FOR-1653): the gate's venue prefix
``mlflow-live-gate/`` held 1089 keys, so one unpaginated
``list_objects_v2(Prefix="mlflow-live-gate/")`` returned page 1 (1000 keys,
``IsTruncated=True``) and the run under test sat on page 2; the live gate went
red with ``keys=[]`` while the artifact was in R2. These unit tests pin the two
properties that end that failure class: the probe is scoped to ONE run's own
prefix, and whatever it lists, it walks every page.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from helpers import LIVE_GATE_BUCKET, LIVE_GATE_PREFIX, list_keys_paginated, live_gate_run_prefix

pytestmark = pytest.mark.unit


class _TwoPageS3:
    """A client whose first page is truncated, the state measured 2026-10-15."""

    page_one: ClassVar[dict[str, Any]] = {
        "Contents": [{"Key": "p/2/e7a2/artifacts/a.pt"}],
        "IsTruncated": True,
        "NextContinuationToken": "t1",
    }
    page_two: ClassVar[dict[str, Any]] = {
        "Contents": [{"Key": "p/2/eb84/artifacts/b.pt"}],
        "IsTruncated": False,
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"api": "list_objects_v2", **kwargs})
        return self.page_two if kwargs.get("ContinuationToken") == "t1" else self.page_one

    def get_paginator(self, name: str) -> Any:
        assert name == "list_objects_v2", name
        client = self

        class _Paginator:
            def paginate(self, **kwargs: Any) -> Any:
                client.calls.append({"api": "paginate", **kwargs})
                yield client.page_one
                yield client.page_two

        return _Paginator()


def test_the_run_prefix_is_scoped_below_the_gate_venue() -> None:
    prefix = live_gate_run_prefix("2", "eb8452dccabc4efcaa69301f0edcfa06")
    assert prefix == "mlflow-live-gate/2/eb8452dccabc4efcaa69301f0edcfa06/"
    assert prefix.startswith(LIVE_GATE_PREFIX)
    assert prefix != LIVE_GATE_PREFIX, "the probe must never list the whole venue"
    assert LIVE_GATE_BUCKET == "entropic-substrate-lm"


def test_list_keys_paginated_walks_past_a_truncated_first_page() -> None:
    s3 = _TwoPageS3()
    keys = list_keys_paginated(s3, LIVE_GATE_BUCKET, "p/2/eb84/")
    assert keys == ["p/2/e7a2/artifacts/a.pt", "p/2/eb84/artifacts/b.pt"], keys
    assert s3.calls, "the client was never asked"
    assert all(c["api"] == "paginate" for c in s3.calls), s3.calls
    assert s3.calls[0]["Bucket"] == LIVE_GATE_BUCKET
    assert s3.calls[0]["Prefix"] == "p/2/eb84/"
