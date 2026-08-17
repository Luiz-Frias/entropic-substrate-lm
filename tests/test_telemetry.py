"""Admissibility of run telemetry — the read side of ``results/*.jsonl``.

These pin the *gate predicate*: what counts as a usable E3 kappa readout. The
CLI preflight and the ladder runner both go through :mod:`entropic_substrate.
telemetry`, so a rule pinned here holds for both — the duplication that let one
copy be hardened and the other not is the defect being fixed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from entropic_substrate.telemetry import (
    TelemetryGap,
    final_eval_metric,
    final_eval_record,
    main,
    read_eval_series,
    read_final_kappa,
    read_seed,
    read_train_series,
)

pytestmark = pytest.mark.unit


def _write(path: Path, *records: str) -> Path:
    path.write_text("".join(r if r.endswith("\n") else r + "\n" for r in records), encoding="utf-8")
    return path


def _eval(**fields: Any) -> str:
    record: dict[str, Any] = {"event": "eval", "step": 0, "loss": 1.0, "nats_per_byte": 1.5}
    record.update(fields)
    return json.dumps(record)


def _train(**fields: Any) -> str:
    record: dict[str, Any] = {"event": "train", "step": 1, "loss": 1.0, "nats_per_byte": 1.5}
    record.update(fields)
    return json.dumps(record)


def _run_start(*, seed: int = 42, **overrides: Any) -> str:
    config: dict[str, Any] = {"train": {"seed": seed}}
    config.update(overrides)
    return json.dumps({"event": "run_start", "step": 0, "config": config})


# ------------------------------------------------------------- happy path
def test_reads_the_final_eval_records_kappa(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "e3.jsonl",
        json.dumps({"event": "run_start", "seed": 42}),
        _eval(kappa=[[-0.9, 0.1]]),
        json.dumps({"event": "log", "step": 10}),
        _eval(kappa=[[-0.9, 0.1, 1.2, -0.4], [-0.2, 0.0, 0.3, 0.1]]),
    )
    readout = read_final_kappa(path)
    assert readout.rows == [[-0.9, 0.1, 1.2, -0.4], [-0.2, 0.0, 0.3, 0.1]]
    assert readout.spreads == pytest.approx([2.1, 0.5])
    assert readout.max_spread == pytest.approx(2.1)
    assert any("head spread" in line for line in readout.report_lines())


def test_integer_kappa_values_are_admissible(tmp_path: Path) -> None:
    """JSON has one number type; a kappa that landed on 0 serialises as ``0``."""
    path = _write(tmp_path / "e3.jsonl", _eval(kappa=[[0, 1]]))
    assert read_final_kappa(path).rows == [[0.0, 1.0]]


# --------------------------------------------------------- absent readout
def test_missing_file_is_a_gap(tmp_path: Path) -> None:
    with pytest.raises(TelemetryGap, match="no such file"):
        read_final_kappa(tmp_path / "absent.jsonl")


def test_no_eval_record_is_a_gap(tmp_path: Path) -> None:
    path = _write(tmp_path / "e3.jsonl", json.dumps({"event": "run_start"}))
    with pytest.raises(TelemetryGap, match="no eval record"):
        read_final_kappa(path)


def test_kappa_absent_from_the_final_eval_is_a_gap(tmp_path: Path) -> None:
    """Stricter than 'last record that happens to carry kappa': gating on a
    stale readout while the final eval lost telemetry is the failure mode."""
    path = _write(
        tmp_path / "e3.jsonl",
        _eval(kappa=[[-0.9, 0.1]]),
        _eval(),
    )
    with pytest.raises(TelemetryGap, match="carries no kappa"):
        read_final_kappa(path)


# ------------------------------------------------------ malformed readout
@pytest.mark.parametrize(
    ("kappa", "expected"),
    [
        ([[]], "empty"),
        ([[0.1, 0.2], []], "empty"),
        ([], "empty"),
        ([["a", "b"]], "not numeric"),
        ([[True, False]], "not numeric"),  # bool is an int subclass
        ([[None, 1.0]], "not numeric"),
        ([[float("nan"), 0.1]], "non-finite"),
        ([[float("inf"), 1.0]], "non-finite"),
        ([[0.1, float("-inf")]], "non-finite"),
        (5, "list of per-layer rows"),
        ({"layer0": [0.1]}, "list of per-layer rows"),
        ([0.1, 0.2], "list of per-layer rows"),
    ],
)
def test_malformed_kappa_is_a_gap_not_a_crash(tmp_path: Path, kappa: Any, expected: str) -> None:
    """A numerically failed run writes NaN that ``json`` happily parses back;
    an interrupted one writes short rows. Neither may reach a spread."""
    path = tmp_path / "e3.jsonl"
    path.write_text(
        json.dumps({"event": "eval", "nats_per_byte": 1.5, "kappa": kappa}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TelemetryGap, match=expected):
        read_final_kappa(path)


def test_truncated_jsonl_line_is_a_gap(tmp_path: Path) -> None:
    """A run killed mid-write leaves a partial line; json raises, not the gate."""
    path = tmp_path / "e3.jsonl"
    path.write_text(_eval(kappa=[[0.1]]) + "\n" + '{"event": "eval", "kap', encoding="utf-8")
    with pytest.raises(TelemetryGap, match="line 2"):
        read_final_kappa(path)


# ------------------------------------------------------------ eval metric
def test_final_eval_metric_reads_the_last_eval(tmp_path: Path) -> None:
    path = _write(tmp_path / "e2.jsonl", _eval(nats_per_byte=2.0), _eval(nats_per_byte=1.25))
    assert final_eval_metric(path, "nats_per_byte") == pytest.approx(1.25)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_metric_is_a_gap(tmp_path: Path, value: float) -> None:
    path = tmp_path / "e2.jsonl"
    path.write_text(json.dumps({"event": "eval", "nats_per_byte": value}) + "\n", encoding="utf-8")
    with pytest.raises(TelemetryGap, match="non-finite"):
        final_eval_metric(path, "nats_per_byte")


def test_absent_metric_is_a_gap(tmp_path: Path) -> None:
    path = _write(tmp_path / "e2.jsonl", json.dumps({"event": "eval"}))
    with pytest.raises(TelemetryGap, match="nats_per_byte"):
        final_eval_metric(path, "nats_per_byte")


def test_final_eval_record_returns_the_last_one(tmp_path: Path) -> None:
    path = _write(tmp_path / "e1.jsonl", _eval(step=1), _eval(step=2))
    assert final_eval_record(path)["step"] == 2


# --------------------------------------------------------------- eval series
def test_read_eval_series_returns_every_admissible_eval_in_order(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "e2.jsonl",
        _run_start(),
        _eval(step=10, loss=2.0, nats_per_byte=1.8),
        _train(step=15, loss=1.9),
        _eval(step=20, loss=1.5, nats_per_byte=1.3),
    )
    series = read_eval_series(path)
    assert [r.step for r in series] == [10, 20]
    assert series[0].loss == pytest.approx(2.0)
    assert series[0].nats_per_byte == pytest.approx(1.8)
    assert series[0].kappa is None
    assert series[1].nats_per_byte == pytest.approx(1.3)


def test_read_eval_series_carries_validated_kappa_per_record(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "e3.jsonl",
        _eval(step=1, kappa=[[-0.9, 0.1]]),
        _eval(step=2),  # legitimate: a run may drop kappa on a later eval
        _eval(step=3, kappa=[[0.2, 0.4]]),
    )
    series = read_eval_series(path)
    assert series[0].kappa == [[-0.9, 0.1]]
    assert series[1].kappa is None
    assert series[2].kappa == [[0.2, 0.4]]


def test_read_eval_series_on_empty_file_is_a_gap(tmp_path: Path) -> None:
    path = tmp_path / "e2.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(TelemetryGap, match=r"no such file|no eval record"):
        read_eval_series(path)


def test_read_eval_series_with_no_eval_records_is_a_gap(tmp_path: Path) -> None:
    path = _write(tmp_path / "e2.jsonl", _run_start(), _train(step=1))
    with pytest.raises(TelemetryGap, match="no eval record"):
        read_eval_series(path)


def test_read_eval_series_refuses_a_nonfinite_row_anywhere_in_the_series(tmp_path: Path) -> None:
    """A curve that looks healthy right up to a NaN-poisoned tail must not
    silently truncate — the whole series is inadmissible, not a prefix of it."""
    path = _write(
        tmp_path / "e2.jsonl",
        _eval(step=1, nats_per_byte=1.5),
        _eval(step=2, nats_per_byte=float("nan")),
    )
    with pytest.raises(TelemetryGap, match="non-finite"):
        read_eval_series(path)


def test_read_eval_series_refuses_malformed_kappa_anywhere(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "e3.jsonl",
        _eval(step=1, kappa=[[0.1]]),
        _eval(step=2, kappa=[[]]),
    )
    with pytest.raises(TelemetryGap, match="empty"):
        read_eval_series(path)


# -------------------------------------------------------------- train series
def test_read_train_series_returns_every_admissible_train_row(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "e2.jsonl",
        _run_start(),
        _train(step=1, loss=3.0, nats_per_byte=2.5),
        _eval(step=1),
        _train(step=2, loss=2.8, nats_per_byte=2.3),
    )
    series = read_train_series(path)
    assert [r.step for r in series] == [1, 2]
    assert series[1].loss == pytest.approx(2.8)


def test_read_train_series_with_no_train_records_is_a_gap(tmp_path: Path) -> None:
    path = _write(tmp_path / "e2.jsonl", _run_start(), _eval(step=1))
    with pytest.raises(TelemetryGap, match="no train record"):
        read_train_series(path)


def test_read_train_series_refuses_a_nonfinite_row(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "e2.jsonl",
        _train(step=1, loss=1.0),
        _train(step=2, loss=float("inf")),
    )
    with pytest.raises(TelemetryGap, match="non-finite"):
        read_train_series(path)


# --------------------------------------------------------------------- seed
def test_read_seed_reads_the_run_start_record(tmp_path: Path) -> None:
    path = _write(tmp_path / "e1.jsonl", _run_start(seed=7), _eval(step=1))
    assert read_seed(path) == 7


def test_read_seed_without_a_run_start_record_is_a_gap(tmp_path: Path) -> None:
    path = _write(tmp_path / "e1.jsonl", _eval(step=1))
    with pytest.raises(TelemetryGap, match="no run_start record"):
        read_seed(path)


def test_read_seed_missing_from_config_is_a_gap(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "e1.jsonl", json.dumps({"event": "run_start", "config": {"train": {}}})
    )
    with pytest.raises(TelemetryGap, match=r"no train\.seed"):
        read_seed(path)


# -------------------------------------------------------------- module CLI
def test_cli_kappa_report_prints_the_divergence(tmp_path: Path, capsys) -> None:
    path = _write(tmp_path / "e3.jsonl", _eval(kappa=[[-1.0, 1.0]]))
    assert main(["kappa-report", str(path)]) == 0
    out = capsys.readouterr().out
    assert "head spread 2.0000" in out
    assert "max within-layer head spread: 2.0000" in out


def test_cli_kappa_report_refuses_malformed_telemetry(tmp_path: Path, capsys) -> None:
    path = tmp_path / "e3.jsonl"
    path.write_text(json.dumps({"event": "eval", "kappa": [[math.nan]]}) + "\n", encoding="utf-8")
    assert main(["kappa-report", str(path)]) == 2
    assert "non-finite" in capsys.readouterr().err


def test_cli_final_eval_prints_only_the_number(tmp_path: Path, capsys) -> None:
    path = _write(tmp_path / "e2.jsonl", _eval(nats_per_byte=1.25))
    assert main(["final-eval", str(path), "--metric", "nats_per_byte"]) == 0
    assert capsys.readouterr().out.strip() == "1.25"


def test_cli_final_eval_refuses_a_missing_file(tmp_path: Path, capsys) -> None:
    assert main(["final-eval", str(tmp_path / "absent.jsonl"), "--metric", "nats_per_byte"]) == 2
    assert "no such file" in capsys.readouterr().err
