"""FOR-1704 (L4d): the aggregate loader reads ``h-predictive-evaluation-v2`` and refuses, never drops.

Synthetic streams only (the writer is a reader member; its pins live with the staged patch).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tests import evaluation_v2_fixtures as fx
from tests.test_h_aggregate import REQUEST_A, write_h_run

from entropic_substrate.ladder.analyze import h_aggregate, h_final_probe_checks
from entropic_substrate.ladder.analyze.h_aggregate import load_h_run
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap
from entropic_substrate.ladder.run import h_metric_extractor

pytestmark = pytest.mark.unit


def _v2_rows(rows: list[dict[str, Any]], mutate: Callable[[dict[str, Any]], None] | None) -> None:
    """Rewrite every v1 evaluation row of a builder run into a v2 one."""
    for row in rows:
        if row.get("schema") != "h-predictive-evaluation-v1":
            continue
        row["schema"] = "h-predictive-evaluation-v2"
        row["latent_operand"] = "posterior_mean"
        for example in row["examples"]:
            q = example["by_sample_count"]
            example["proposals"] = {
                "prior": {
                    "proposal": "prior",
                    "by_sample_count": copy.deepcopy(q),
                },
                "mixture": {
                    "proposal": "mixture",
                    "by_sample_count": copy.deepcopy(q),
                },
            }
        if mutate is not None:
            mutate(row)


def _load(tmp_path: Path, mutate: Callable[[dict[str, Any]], None] | None = None):  # type: ignore[no-untyped-def]
    return load_h_run(*write_h_run(tmp_path, mutate_metrics=lambda rows: _v2_rows(rows, mutate)))


def test_a_v2_run_loads_and_carries_the_proposal_keys_beside_the_q_keys(tmp_path: Path) -> None:
    run = _load(tmp_path)
    _, metrics = run.evaluations[0]
    for key in ("eval/ess_k2", "eval/ess_prior_k2", "eval/ess_mixture_k4"):
        assert key in metrics, key
    # the E-LM projection is the single default-q one: no non-q key is merged into HRun.elm
    _, elm = run.elm[0]
    assert elm["proposal"] == "q"
    assert not any("prior" in key or "mixture" in key for key in elm["metrics"])


def test_a_malformed_prior_row_voids_the_run_instead_of_being_dropped(tmp_path: Path) -> None:
    def bad(row: dict[str, Any]) -> None:
        row["examples"][0]["proposals"]["prior"]["by_sample_count"]["2"][0][
            "effective_sample_size"
        ] = None

    with pytest.raises(TelemetryGap, match=r"prior proposal.*non-finite effective_sample_size"):
        _load(tmp_path, bad)


def test_a_mixture_block_with_another_sample_count_set_is_refused(tmp_path: Path) -> None:
    def bad(row: dict[str, Any]) -> None:
        del row["examples"][0]["proposals"]["mixture"]["by_sample_count"]["4"]

    with pytest.raises(TelemetryGap, match="mixture proposal sample counts differ"):
        _load(tmp_path, bad)


def test_a_v2_example_without_proposals_is_refused(tmp_path: Path) -> None:
    def bad(row: dict[str, Any]) -> None:
        del row["examples"][0]["proposals"]

    with pytest.raises(TelemetryGap, match="carries no proposals"):
        _load(tmp_path, bad)


def test_a_proposals_block_on_a_v1_row_is_refused(tmp_path: Path) -> None:
    def bad(row: dict[str, Any]) -> None:
        row["schema"] = "h-predictive-evaluation-v1"

    with pytest.raises(TelemetryGap, match="proposals on schema"):
        _load(tmp_path, bad)


def test_an_unknown_or_misnamed_proposal_is_refused(tmp_path: Path) -> None:
    def extra(row: dict[str, Any]) -> None:
        row["examples"][0]["proposals"]["posterior"] = {}

    def misnamed(row: dict[str, Any]) -> None:
        row["examples"][0]["proposals"]["prior"]["proposal"] = "mixture"

    with pytest.raises(TelemetryGap, match="proposals are"):
        _load(tmp_path, extra)
    with pytest.raises(TelemetryGap, match="malformed prior proposal block"):
        _load(tmp_path / "b", misnamed)


def test_a_v1_stream_still_loads_exactly_as_before(tmp_path: Path) -> None:
    run = load_h_run(*write_h_run(tmp_path))
    assert "eval/ess_prior_k2" not in run.evaluations[0][1]
    assert "profile" not in run.run_start


def test_every_analysis_side_schema_tuple_is_the_same_closed_pair() -> None:
    expected = ("h-predictive-evaluation-v1", "h-predictive-evaluation-v2")
    assert h_aggregate._EVALUATION_SCHEMAS == expected  # pyright: ignore[reportPrivateUsage]
    assert h_final_probe_checks._EVALUATION_SCHEMAS == expected  # pyright: ignore[reportPrivateUsage]
    assert h_metric_extractor.EVALUATION_SCHEMAS == expected
    assert (fx.V1, fx.V2) == expected


def test_the_final_probe_check_reads_the_q_curve_of_a_v2_report(tmp_path: Path) -> None:
    files = write_h_run(tmp_path, mutate_metrics=lambda rows: _v2_rows(rows, None))
    result = h_final_probe_checks.check_importance_curve([files[0]], (2, 4))
    assert result.verdict == "PASS", result.reasons
    assert REQUEST_A  # the builder's request digest is shared with the sibling tests
