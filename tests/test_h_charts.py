"""The H chart set and its attach step (FOR-1648).

Charts are a pure transform of the aggregate document, so the ``unit`` tests build the document
from the synthetic runs of tests/test_h_aggregate.py. The attach step is tested against a stub
client; no test reaches a tracking server. ``integration`` tests read the real HEpP7g capture
and raise when it is absent (no skip).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from tests.test_h_aggregate import (
    IDENTITY_A,
    IDENTITY_B,
    REQUEST_A,
    REQUEST_B,
    RUN_ID_A,
    RUN_ID_B,
    fixture_arguments,
    write_h_run,
)

from entropic_substrate.ladder.analyze.h_aggregate import aggregate_h_runs, load_h_run
from entropic_substrate.ladder.analyze.h_attach import HAttachError, attach_h_figures
from entropic_substrate.ladder.analyze.h_charts import build_h_figures


def _document(tmp_path: Path) -> dict[str, Any]:
    a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a", run_id=RUN_ID_A, scale=1.0)
    b = write_h_run(tmp_path, IDENTITY_B, REQUEST_B, tag="b", run_id=RUN_ID_B, scale=2.0)
    return aggregate_h_runs(
        [load_h_run(*a), load_h_run(*b)], {IDENTITY_A: "magnitude", IDENTITY_B: "sphere"}
    )


@pytest.mark.unit
class TestBuildHFigures:
    def test_the_full_set_is_built_when_the_capture_holds_the_data(self, tmp_path: Path) -> None:
        figures, absent = build_h_figures(_document(tmp_path))
        assert absent == {}
        assert set(figures) == {
            "h_training_curves",
            "h_future_nll_per_dimension",
            "h_evaluation_rows",
            "h_publication_timing",
        }
        from matplotlib.figure import Figure

        assert all(isinstance(f, Figure) for f in figures.values())

    def test_a_chart_without_data_is_named_absent_never_rendered_empty(
        self, tmp_path: Path
    ) -> None:
        document = _document(tmp_path)
        document["publication_seconds"]["values"] = []
        for rows in document["evaluation"]["metrics"].values():
            rows.clear()
        figures, absent = build_h_figures(document)
        assert set(absent) == {"h_publication_timing", "h_evaluation_rows"}
        assert set(figures) == {"h_training_curves", "h_future_nll_per_dimension"}
        assert all(reason for reason in absent.values())

    def test_future_nll_chart_is_absent_when_every_row_is_null(self, tmp_path: Path) -> None:
        document = _document(tmp_path)
        for row in document["training"]["metrics"]["future_nll_per_dimension"]:
            row["values"] = dict.fromkeys(row["values"])
        figures, absent = build_h_figures(document)
        assert "h_future_nll_per_dimension" in absent
        assert "h_future_nll_per_dimension" not in figures

    @pytest.mark.parametrize("break_it", ["no arms", "no steps"])
    def test_empty_input_raises(self, tmp_path: Path, break_it: str) -> None:
        document = copy.deepcopy(_document(tmp_path))
        if break_it == "no arms":
            document["arms"] = {}
        else:
            document["training"]["paired_steps"] = []
        with pytest.raises(ValueError, match="nothing to chart"):
            build_h_figures(document)

    def test_every_arm_carries_a_non_color_channel_and_a_direct_label(self, tmp_path: Path) -> None:
        figures, _ = build_h_figures(_document(tmp_path))
        for name, figure in figures.items():
            for ax in figure.axes:
                lines = [ln for ln in ax.get_lines() if ln.get_label() in {"magnitude", "sphere"}]
                assert {ln.get_label() for ln in lines} == {"magnitude", "sphere"}, name
                channels = {(ln.get_linestyle(), ln.get_marker()) for ln in lines}
                assert len(channels) == 2, f"{name}: arms differ by colour alone"
                assert ax.get_legend() is not None, name

    def test_no_axis_is_logarithmic(self, tmp_path: Path) -> None:
        figures, _ = build_h_figures(_document(tmp_path))
        for figure in figures.values():
            for ax in figure.axes:
                assert ax.get_yscale() == "linear"
                assert ax.get_xscale() == "linear"

    def test_negative_future_nll_values_are_drawn_as_captured(self, tmp_path: Path) -> None:
        figures, _ = build_h_figures(_document(tmp_path))
        ax = figures["h_future_nll_per_dimension"].axes[0]
        drawn = [
            y for ln in ax.get_lines() if ln.get_label() == "magnitude" for y in ln.get_ydata()
        ]
        assert min(drawn) == pytest.approx(-0.6)
        assert len(drawn) == 4  # the two warm-up nulls are gaps, not zeros

    def test_training_curve_x_axis_is_the_global_step(self, tmp_path: Path) -> None:
        figures, _ = build_h_figures(_document(tmp_path))
        ax = figures["h_training_curves"].axes[0]
        line = next(ln for ln in ax.get_lines() if ln.get_label() == "magnitude")
        assert list(line.get_xdata()) == [1, 2, 3, 4, 5, 6]


class _Stub:
    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_for = fail_for or set()

    def log_figure(self, run_id: str, figure: object, artifact_file: str) -> None:
        del figure
        if run_id in self.fail_for:
            raise ConnectionError(f"cannot reach the server for {run_id}")
        self.calls.append((run_id, artifact_file))


@pytest.mark.unit
class TestAttachFailsLoud:
    def _figures(self, tmp_path: Path) -> Any:
        return build_h_figures(_document(tmp_path))[0]

    def test_every_figure_goes_to_every_run_under_charts(self, tmp_path: Path) -> None:
        figures = self._figures(tmp_path)
        stub = _Stub()
        attached = attach_h_figures(stub, {"magnitude": RUN_ID_A, "sphere": RUN_ID_B}, figures)
        assert sorted(attached) == [RUN_ID_A, RUN_ID_B]
        assert {name for _, name in stub.calls} == {f"charts/{n}.png" for n in figures}
        assert len(stub.calls) == 2 * len(figures)

    def test_a_failing_run_surfaces_its_id_after_the_others_were_attempted(
        self, tmp_path: Path
    ) -> None:
        figures = self._figures(tmp_path)
        stub = _Stub(fail_for={RUN_ID_A})
        with pytest.raises(HAttachError) as excinfo:
            attach_h_figures(stub, {"magnitude": RUN_ID_A, "sphere": RUN_ID_B}, figures)
        assert set(excinfo.value.failures) == {RUN_ID_A}
        assert RUN_ID_A in str(excinfo.value)
        assert {run for run, _ in stub.calls} == {RUN_ID_B}  # attempted, not abandoned
        assert excinfo.value.attached[RUN_ID_B]

    def test_every_failed_run_is_named_not_just_the_first(self, tmp_path: Path) -> None:
        figures = self._figures(tmp_path)
        stub = _Stub(fail_for={RUN_ID_A, RUN_ID_B})
        with pytest.raises(HAttachError) as excinfo:
            attach_h_figures(stub, {"magnitude": RUN_ID_A, "sphere": RUN_ID_B}, figures)
        assert set(excinfo.value.failures) == {RUN_ID_A, RUN_ID_B}

    def test_a_missing_run_id_is_a_failure_never_a_skip(self, tmp_path: Path) -> None:
        figures = self._figures(tmp_path)
        stub = _Stub()
        with pytest.raises(HAttachError) as excinfo:
            attach_h_figures(stub, {"magnitude": None, "sphere": RUN_ID_B}, figures)
        assert set(excinfo.value.failures) == {"magnitude"}
        assert {run for run, _ in stub.calls} == {RUN_ID_B}


@pytest.mark.integration
class TestChartsOnTheCapturedRuns:
    def test_the_full_set_renders_from_the_real_runs(self) -> None:
        pairs, labels = fixture_arguments()
        document = aggregate_h_runs([load_h_run(*p) for p in pairs], labels)
        figures, absent = build_h_figures(document)
        assert absent == {}
        assert len(figures) == 4
        evaluation = figures["h_evaluation_rows"]
        assert len(evaluation.axes) == 4  # two metrics x k in {2, 4}
        nll = figures["h_future_nll_per_dimension"].axes[0]
        drawn = [
            y for ln in nll.get_lines() if ln.get_label() in labels.values() for y in ln.get_ydata()
        ]
        assert min(drawn) < 0
