"""The H1 E-LM projection: per-canonical-byte log-marginal and ESS/k over captured rows.

The ``unit`` tests build evaluation events by hand, with every expected number computed
on paper beside the fixture. The ``integration`` tests read the final H0 probe capture
under ``results/h0-final-probe-1k-v4-92c9db/`` (``dvc pull
results/h0-final-probe-1k-v4-92c9db.dvc``) and RAISE when it is absent: a skip here
would be a false green. That bundle holds six runs (the HEpP7g pair, the final-probe
pair and a stop/resume pod pair); the tests select the two complete final-probe runs by
profile identity, never by globbing.
"""

from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
from tests.test_h_aggregate import (
    IDENTITY_A,
    IDENTITY_B,
    REQUEST_A,
    REQUEST_B,
    RUN_ID_B,
    write_h_run,
)

from entropic_substrate.ladder.analyze import h_elm_projection
from entropic_substrate.ladder.analyze.h_aggregate import aggregate_h_runs, load_h_run
from entropic_substrate.ladder.analyze.h_elm_projection import (
    DENOMINATORS,
    HEADLINE_SAMPLES,
    project_elm,
)
from entropic_substrate.ladder.analyze.h_report import h_report_from_run
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap


def _row(estimate: float, ess: float) -> dict[str, Any]:
    return {
        "effective_sample_size": ess,
        "gap": None,
        "likelihood_relative_mcse": 0.5,
        "log_estimate": estimate,
    }


_GAP_ROW: dict[str, Any] = {
    "effective_sample_size": None,
    "gap": "zero_importance_weight_mass",
    "likelihood_relative_mcse": None,
    "log_estimate": None,
}


def _event() -> dict[str, Any]:
    """Two examples (batches) of two rows each, k in {2, 4}.

    Bytes 100, 200 | 50, 400; boundary symbols 1 each.
    k=2: estimates -200, -300 | -100, -800 -> per byte -2, -1.5 | -2, -2; mean -1.875.
         ESS 1.0, 1.0 | 1.0, 2.0 -> ESS/k 0.5, 0.5, 0.5, 1.0; mean 0.625, median 0.5.
    k=4: estimates -100, -400 | -50, -400 -> per byte -1, -2 | -1, -1; mean -1.25.
         ESS 0.4*4=1.6 ... chosen so ESS/k = 0.4, 0.4, 0.4, 1.0 (mean 0.55, median 0.4):
         mean and median sit on opposite sides of 0.5, so the Q4 choice is load-bearing.
    """
    return {
        "examples": [
            {
                "boundary_symbols": [1, 1],
                "by_sample_count": {
                    "2": [_row(-200.0, 1.0), _row(-300.0, 1.0)],
                    "4": [_row(-100.0, 1.6), _row(-400.0, 1.6)],
                },
                "source_bytes": [100, 200],
            },
            {
                "boundary_symbols": [1, 1],
                "by_sample_count": {
                    "2": [_row(-100.0, 1.0), _row(-800.0, 2.0)],
                    "4": [_row(-50.0, 1.6), _row(-400.0, 4.0)],
                },
                "source_bytes": [50, 400],
            },
        ],
        "step": 4,
    }


# ------------------------------------------------------------------ unit: the projection
@pytest.mark.unit
class TestProjection:
    def test_per_byte_mean_is_hand_computed(self) -> None:
        elm = project_elm(_event())
        assert elm["denominator"] == "source_bytes"
        assert elm["samples"]["2"]["log_marginal_per_byte"] == [-2.0, -1.5, -2.0, -2.0]
        assert elm["metrics"]["eval/log_marginal_per_byte_k2"] == pytest.approx(-1.875)
        assert elm["metrics"]["eval/log_marginal_per_byte_k4"] == pytest.approx(-1.25)

    def test_ess_ratio_median_and_mean_differ_and_both_are_reported(self) -> None:
        elm = project_elm(_event())
        assert elm["samples"]["4"]["ess_ratio"] == pytest.approx([0.4, 0.4, 0.4, 1.0])
        assert elm["metrics"]["eval/ess_ratio_median_k4"] == pytest.approx(0.4)
        assert elm["metrics"]["eval/ess_ratio_mean_k4"] == pytest.approx(0.55)
        assert elm["metrics"]["eval/ess_ratio_median_k2"] == pytest.approx(0.5)
        assert elm["metrics"]["eval/ess_ratio_mean_k2"] == pytest.approx(0.625)

    def test_even_row_count_median_is_the_midpoint_average(self) -> None:
        event = _event()
        event["examples"] = event["examples"][:1]  # k=4 ESS/k = 0.4, 0.4 ; k=2 = 0.5, 0.5
        event["examples"][0]["by_sample_count"]["4"][1] = _row(-400.0, 4.0)  # 0.4, 1.0
        assert project_elm(event)["metrics"]["eval/ess_ratio_median_k4"] == pytest.approx(0.7)

    def test_the_curve_covers_every_k_the_event_carries_in_numeric_order(self) -> None:
        event = _event()
        for example in event["examples"]:
            example["by_sample_count"]["16"] = copy.deepcopy(example["by_sample_count"]["4"])
        assert list(project_elm(event)["samples"]) == ["2", "4", "16"]

    def test_row_counts_and_finite_counts_are_reported_per_k(self) -> None:
        samples = project_elm(_event())["samples"]
        assert (samples["2"]["rows"], samples["2"]["finite_rows"], samples["2"]["gap_rows"]) == (
            4,
            4,
            0,
        )
        assert project_elm(_event())["metrics"]["eval/finite_rows_k2"] == 4.0

    def test_the_boundary_inclusive_denominator_is_an_explicit_parameter(self) -> None:
        assert set(DENOMINATORS) == {"source_bytes", "source_bytes_plus_boundary"}
        elm = project_elm(_event(), denominator="source_bytes_plus_boundary")
        assert elm["denominator"] == "source_bytes_plus_boundary"
        assert elm["samples"]["2"]["log_marginal_per_byte"] == pytest.approx(
            [-200 / 101, -300 / 201, -100 / 51, -800 / 401]
        )

    def test_an_unknown_denominator_is_refused(self) -> None:
        with pytest.raises(ValueError, match="denominator"):
            project_elm(_event(), denominator="bytes")


@pytest.mark.unit
class TestHeadline:
    def test_headline_k_is_64(self) -> None:
        assert HEADLINE_SAMPLES == 64

    def test_an_absent_headline_k_is_reported_absent_never_zero(self) -> None:
        elm = project_elm(_event())
        assert elm["headline"] == {
            "ess_ratio_median": None,
            "log_marginal_per_byte": None,
            "present": False,
            "samples": 64,
        }
        assert "eval/log_marginal_per_byte_k64" not in elm["metrics"]
        assert "eval/ess_ratio_median_k64" not in elm["metrics"]

    def test_a_present_headline_k_carries_its_two_numbers(self) -> None:
        event = _event()
        for example in event["examples"]:
            example["by_sample_count"]["64"] = copy.deepcopy(example["by_sample_count"]["4"])
            for row in example["by_sample_count"]["64"]:
                row["effective_sample_size"] *= 16  # same ESS/k as k=4
        elm = project_elm(event)
        assert elm["headline"]["present"] is True
        assert elm["headline"]["log_marginal_per_byte"] == pytest.approx(-1.25)
        assert elm["headline"]["ess_ratio_median"] == pytest.approx(0.4)
        assert elm["metrics"]["eval/log_marginal_per_byte_k64"] == pytest.approx(-1.25)


@pytest.mark.unit
class TestGapRows:
    def test_a_gap_row_is_counted_and_excluded_from_both_means(self) -> None:
        event = _event()
        event["examples"][1]["by_sample_count"]["2"][1] = dict(_GAP_ROW)
        k2 = project_elm(event)["samples"]["2"]
        assert (k2["rows"], k2["finite_rows"], k2["gap_rows"]) == (4, 3, 1)
        assert k2["log_marginal_per_byte"] == [-2.0, -1.5, -2.0, None]
        assert project_elm(event)["metrics"]["eval/log_marginal_per_byte_k2"] == pytest.approx(
            (-2.0 - 1.5 - 2.0) / 3
        )

    def test_all_gap_rows_report_no_value_never_zero_and_never_raise(self) -> None:
        # registered collapse is valid negative evidence: counted, not refused
        event = _event()
        for example in event["examples"]:
            example["by_sample_count"]["4"] = [dict(_GAP_ROW), dict(_GAP_ROW)]
        elm = project_elm(event)
        assert elm["samples"]["4"]["gap_rows"] == 4
        assert elm["samples"]["4"]["finite_rows"] == 0
        assert elm["samples"]["4"]["mean_log_marginal_per_byte"] is None
        assert elm["samples"]["4"]["ess_ratio_median"] is None
        assert "eval/log_marginal_per_byte_k4" not in elm["metrics"]
        assert "eval/ess_ratio_median_k4" not in elm["metrics"]
        assert elm["metrics"]["eval/finite_rows_k4"] == 0.0


def _refused(event: dict[str, Any], match: str) -> None:
    with pytest.raises(TelemetryGap, match=match):
        project_elm(event, where="evaluation at global step 6")


@pytest.mark.unit
class TestInadmissibleRowsFailLoud:
    def test_no_examples(self) -> None:
        event = _event()
        event["examples"] = []
        _refused(event, "no examples")

    @pytest.mark.parametrize("value", [0, -3])
    def test_zero_or_negative_source_bytes_raise_never_divide(self, value: int) -> None:
        event = _event()
        event["examples"][0]["source_bytes"] = [100, value]
        _refused(event, "source_bytes")

    @pytest.mark.parametrize("value", [None, [100], [100.0, 200], [True, 200], "100,200"])
    def test_missing_short_or_non_integer_source_bytes_raise(self, value: object) -> None:
        event = _event()
        event["examples"][0]["source_bytes"] = value
        _refused(event, "source_bytes")

    def test_missing_boundary_symbols_raise(self) -> None:
        event = _event()
        del event["examples"][1]["boundary_symbols"]
        _refused(event, "boundary_symbols")

    def test_a_k_missing_from_one_example_raises(self) -> None:
        event = _event()
        del event["examples"][1]["by_sample_count"]["4"]
        _refused(event, "sample counts")

    def test_rows_that_disagree_with_the_byte_count_raise(self) -> None:
        event = _event()
        event["examples"][0]["by_sample_count"]["4"].append(_row(-1.0, 1.0))
        _refused(event, "rows")

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, None, "1.0"])
    def test_a_nonfinite_log_estimate_without_a_gap_raises(self, value: object) -> None:
        event = _event()
        event["examples"][0]["by_sample_count"]["2"][0]["log_estimate"] = value
        _refused(event, "log_estimate")

    def test_a_gap_row_that_carries_numbers_raises(self) -> None:
        event = _event()
        row = dict(_GAP_ROW)
        row["log_estimate"] = -5.0
        event["examples"][0]["by_sample_count"]["2"][0] = row
        _refused(event, "gap")

    @pytest.mark.parametrize("ess", [0.5, 2.5])
    def test_an_ess_outside_one_to_k_is_an_execution_defect(self, ess: float) -> None:
        # 1/sum(w^2) over k normalized weights lies in [1, k]; outside it the row is corrupt
        event = _event()
        event["examples"][0]["by_sample_count"]["2"][0]["effective_sample_size"] = ess
        _refused(event, "effective_sample_size")

    def test_a_non_integer_sample_count_key_raises(self) -> None:
        event = _event()
        for example in event["examples"]:
            example["by_sample_count"]["x"] = example["by_sample_count"].pop("4")
        _refused(event, "sample count")

    def test_the_refusal_names_where(self) -> None:
        event = _event()
        event["examples"][0]["source_bytes"] = [0, 0]
        with pytest.raises(TelemetryGap, match="evaluation at global step 6"):
            project_elm(event, where="evaluation at global step 6")


# ------------------------------------------------------------------ unit: offline surface
def _with_bytes(tmp_path: Path) -> HRunPair:
    a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a")
    b = write_h_run(tmp_path, IDENTITY_B, REQUEST_B, tag="b", run_id=RUN_ID_B, scale=2.0)
    return a, b


HRunPair = tuple[tuple[Path, Path], tuple[Path, Path]]


@pytest.mark.unit
class TestOfflineSurface:
    def test_the_run_carries_one_projection_per_evaluation_step(self, tmp_path: Path) -> None:
        a, _ = _with_bytes(tmp_path)
        run = load_h_run(*a)
        assert [step for step, _ in run.elm] == [step for step, _ in run.evaluations]

    def test_the_report_prints_the_headline_and_the_curve(self, tmp_path: Path) -> None:
        a, _ = _with_bytes(tmp_path)
        report = h_report_from_run(load_h_run(*a))
        elm = report["elm"]
        assert elm["denominator"] == "source_bytes"
        assert elm["headline_samples"] == 64
        assert set(elm["steps"]) == {"4", "6"}
        step = elm["steps"]["4"]
        assert step["headline"]["present"] is False
        assert list(step["samples"]) == ["2", "4"]
        assert "eval/log_marginal_per_byte_k2" in step["metrics"]

    def test_the_aggregate_pairs_the_per_byte_metrics_by_step(self, tmp_path: Path) -> None:
        a, b = _with_bytes(tmp_path)
        doc = aggregate_h_runs(
            [load_h_run(*a), load_h_run(*b)], {IDENTITY_A: "magnitude", IDENTITY_B: "sphere"}
        )
        elm = doc["evaluation"]["elm"]
        assert elm["denominator"] == "source_bytes"
        assert elm["headline_samples"] == 64
        assert elm["headline"] == {"magnitude": [], "sphere": []}  # k=64 absent at every step
        rows = elm["metrics"]["eval/log_marginal_per_byte_k2"]
        assert [r["step"] for r in rows] == doc["evaluation"]["paired_steps"]
        assert set(rows[0]["values"]) == {"magnitude", "sphere"}

    def test_the_run_summary_names_the_headline_steps(self, tmp_path: Path) -> None:
        a, _ = _with_bytes(tmp_path)
        from entropic_substrate.ladder.analyze.h_aggregate import run_summary

        summary = run_summary(load_h_run(*a))["elm"]
        assert summary == {
            "denominator": "source_bytes",
            "headline_samples": 64,
            "headline_steps": [],
            "samples": [2, 4],
        }

    def test_a_capture_without_source_bytes_is_refused_at_admission(self, tmp_path: Path) -> None:
        def strip(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                for example in row.get("examples", []):
                    example.pop("source_bytes", None)

        a = write_h_run(tmp_path, mutate_metrics=strip)
        with pytest.raises(TelemetryGap, match="source_bytes") as excinfo:
            load_h_run(*a)
        assert str(a[0]) in str(excinfo.value)


@pytest.mark.unit
class TestModuleContract:
    def test_pure_no_print_no_composition_no_private_cross_import(self) -> None:
        tree = ast.parse(Path(h_elm_projection.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            assert not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            )
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("entropic_substrate.composition")
                assert not node.module.startswith("entropic_substrate.ladder.run")
                for alias in node.names:
                    assert not alias.name.startswith("_"), f"{node.module}.{alias.name}"


# ------------------------------------------------------------------ integration: real capture
FINAL_PROBE_DIR = Path(__file__).resolve().parents[1] / "results" / "h0-final-probe-1k-v4-92c9db"
# sealed mapping (results/h0-final-probe-1k-v4.92c9db/wave.json): A = magnitude, B = sphere
_FINAL_PROBE = {
    "magnitude": "5ece5d1b160d593cebfe3a7bbefc0af40f56e59feb6c7db91777c68dd4004880-14637e75",
    "sphere": "3390f577bcb3df123a68d6ba4c51bfc735a39216343ca20fd08c13cca3238b66-38c25d26",
}


def _final_probe_pairs() -> dict[str, tuple[Path, Path]]:
    if not FINAL_PROBE_DIR.is_dir():
        raise AssertionError(
            f"{FINAL_PROBE_DIR} is absent: run `dvc pull results/h0-final-probe-1k-v4-92c9db.dvc` "
            "(this test fails rather than skips: a skip is a false green)"
        )
    pairs: dict[str, tuple[Path, Path]] = {}
    for arm, stem in _FINAL_PROBE.items():
        metrics = FINAL_PROBE_DIR / f"h-worker-{stem}-metrics.jsonl"
        worker_log = FINAL_PROBE_DIR / f"h-worker-{stem}-worker-log.jsonl"
        assert metrics.is_file() and worker_log.is_file(), stem
        pairs[arm] = (metrics, worker_log)
    return pairs


@pytest.mark.integration
class TestFinalProbeCapture:
    def test_k_64_is_absent_and_the_present_k_are_projected(self) -> None:
        for arm, pair in _final_probe_pairs().items():
            run = load_h_run(*pair)
            assert [step for step, _ in run.elm] == [500, 750, 1000], arm
            for _, elm in run.elm:
                assert list(elm["samples"]) == ["2", "4", "8"], arm
                assert elm["headline"]["present"] is False
                assert "eval/log_marginal_per_byte_k64" not in elm["metrics"]
                for key, block in elm["samples"].items():
                    k = int(key)
                    assert block["rows"] == 4 and block["gap_rows"] == 0, (arm, key)
                    # ESS = 1/sum(w^2) lies in [1, k], so ESS/k lies in [1/k, 1]
                    assert all(1 / k - 1e-12 <= r <= 1 + 1e-12 for r in block["ess_ratio"])
                    assert all(v < 0 for v in block["log_marginal_per_byte"])

    def test_per_byte_equals_the_raw_row_over_its_source_bytes(self) -> None:
        metrics, worker_log = _final_probe_pairs()["sphere"]
        run = load_h_run(metrics, worker_log)
        rows = [
            json.loads(line)
            for line in metrics.read_text().splitlines()
            if '"h-predictive-evaluation-v1"' in line
        ]
        example = rows[0]["examples"][0]
        expected = [
            r["log_estimate"] / n
            for r, n in zip(example["by_sample_count"]["8"], example["source_bytes"], strict=True)
        ]
        assert run.elm[0][1]["samples"]["8"]["log_marginal_per_byte"] == pytest.approx(expected)
        # the runtime per-sentence key is untouched: still the plain row mean
        per_sentence = run.evaluations[0][1]["eval/log_marginal_k8"]
        assert per_sentence == pytest.approx(
            sum(r["log_estimate"] for r in example["by_sample_count"]["8"]) / 4
        )

    def test_the_aggregate_over_both_arms_reports_no_headline(self) -> None:
        pairs = _final_probe_pairs()
        runs = [load_h_run(*pairs["magnitude"]), load_h_run(*pairs["sphere"])]
        labels = {run.profile_identity: arm for run, arm in zip(runs, pairs, strict=True)}
        elm = aggregate_h_runs(runs, labels)["evaluation"]["elm"]
        assert elm["headline"] == {"magnitude": [], "sphere": []}
        assert [r["step"] for r in elm["metrics"]["eval/ess_ratio_median_k8"]] == [500, 750, 1000]
