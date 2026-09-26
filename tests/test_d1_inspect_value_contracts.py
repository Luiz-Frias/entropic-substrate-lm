"""Closed value contracts beneath the D1 inspect projection boundary."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from entropic_substrate.kernel.identity.d1_primitives import D1Arm, RunClass
from entropic_substrate.kernel.verdict.d1_evaluation_values import (
    TARGET_PREDICTED_EXACT_MATCH_SEMANTIC,
)
from entropic_substrate.ladder.analyze.d1_inspect_values import (
    D1_INSPECT_MAX_EVALUATIONS,
    D1InspectEvaluation,
    D1InspectEvaluationWindow,
    D1InspectJournal,
    D1InspectPartial,
    D1InspectPlan,
    D1InspectProgress,
    D1InspectQualityScalars,
    D1InspectRun,
    D1InspectView,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import JournalEventKind
from entropic_substrate.ladder.capture.records.d1_telemetry_values import (
    D1_REPORT_ONLY_METRICS,
    D1_SCALAR_METRICS,
    D1RunStatus,
)
from entropic_substrate.ladder.register.d1_execution import PAIRWISE_COSINE_SEMANTIC

from .d1_telemetry_fixtures import IDENTITY, LOGICAL_RUN_ID, resource, start_provenance

pytestmark = pytest.mark.unit

_DECIDING = D1_SCALAR_METRICS - D1_REPORT_ONLY_METRICS


def _quality() -> D1InspectQualityScalars:
    return D1InspectQualityScalars(
        MappingProxyType(dict.fromkeys(_DECIDING, 1.0)),
        MappingProxyType(dict.fromkeys(D1_REPORT_ONLY_METRICS, 1.0)),
    )


def _evaluation(step: int, index: int) -> D1InspectEvaluation:
    return D1InspectEvaluation(
        step,
        index,
        2,
        _quality(),
        PAIRWISE_COSINE_SEMANTIC,
        TARGET_PREDICTED_EXACT_MATCH_SEMANTIC,
        resource(step, index),
    )


def _run(logical_run_id: str = LOGICAL_RUN_ID) -> D1InspectRun:
    return D1InspectRun(
        logical_run_id,
        RunClass.CONFIRMATORY,
        str(IDENTITY["execution_digest"]),
        str(IDENTITY["aggregation_digest"]),
        str(IDENTITY["construction_digest"]),
        str(IDENTITY["family_digest"]),
        "a" * 64,
        "b" * 64,
    )


def test_quality_partitions_are_exact_disjoint_finite_and_immutable() -> None:
    value = _quality()

    assert set(value.deciding).isdisjoint(value.report_only)
    assert set(value.deciding) | set(value.report_only) == set(D1_SCALAR_METRICS)
    with pytest.raises(TypeError):
        value.deciding["future_nll"] = 2.0  # type: ignore[index]
    with pytest.raises(ValueError, match="keys must be exact"):
        D1InspectQualityScalars({}, value.report_only)
    with pytest.raises(ValueError, match="finite"):
        D1InspectQualityScalars(
            {**value.deciding, "future_nll": float("nan")},
            value.report_only,
        )


@pytest.mark.parametrize(
    ("semantic", "field"),
    [
        (("changed", TARGET_PREDICTED_EXACT_MATCH_SEMANTIC), "pairwise"),
        ((PAIRWISE_COSINE_SEMANTIC, "changed"), "target-predicted"),
    ],
)
def test_evaluation_semantics_and_resource_cursor_are_load_bearing(
    semantic: tuple[str, str], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        D1InspectEvaluation(0, 0, 2, _quality(), *semantic, resource(0, 0))
    with pytest.raises(ValueError, match="cursors differ"):
        D1InspectEvaluation(
            1,
            0,
            2,
            _quality(),
            PAIRWISE_COSINE_SEMANTIC,
            TARGET_PREDICTED_EXACT_MATCH_SEMANTIC,
            resource(0, 0),
        )


def test_window_proves_the_exact_requested_terminal_suffix() -> None:
    value = D1InspectEvaluationWindow(3, 1, (_evaluation(2, 2),))

    assert (value.returned_count, value.omitted_count) == (1, 2)
    for items in ((), (_evaluation(0, 0),)):
        with pytest.raises(ValueError, match=r"returned count|terminal suffix"):
            D1InspectEvaluationWindow(3, 1, items)
    for limit in (0, True, D1_INSPECT_MAX_EVALUATIONS + 1):
        with pytest.raises(ValueError, match="hard bound"):
            D1InspectEvaluationWindow(0, limit, ())


def test_view_rejects_cross_run_nested_evidence_and_derives_partial_status() -> None:
    evaluation = _evaluation(0, 0)
    progress = D1InspectProgress(1, 1, 0, True, 1, 0, 0, start_provenance(), evaluation, None)
    common = (
        D1InspectJournal("/tmp/run.jsonl", "run.jsonl", 1, "c" * 64),
        D1InspectPlan("d1-both", D1Arm.BOTH, 7, "seed-7"),
        progress,
        D1InspectEvaluationWindow(1, 1, (evaluation,)),
        D1InspectPartial(),
    )

    view = D1InspectView(common[0], _run(), *common[1:])

    assert view.status is D1RunStatus.PARTIAL
    with pytest.raises(ValueError, match="outer run identity"):
        D1InspectView(common[0], _run("another-run"), *common[1:])
    with pytest.raises(TypeError, match="arm must be nominal"):
        D1InspectPlan("d1-both", "both", 7, "seed-7")  # type: ignore[arg-type]
    assert JournalEventKind.RUN_END.value == "run_end"
