"""The H analysis surface on the CLI (FOR-1648, operator door ruling B, conductor DN-1648-B).

The frozen CLI-surface door (tests/test_cli_surface.py, 27 top-level names) is NOT amended:
the H surface is three optional nested ``h`` sub-subparsers on the EXISTING ``mlflow-aggregate``,
``charts`` and ``eval`` parsers. This module pins, in order:

* the E invocations keep EXACTLY the namespace they had at 501c0447, against LITERAL expected
  dicts dumped from that tree (never recomputed from ``build_parser()`` at the tip, which
  would be tautological), plus exactly one added lane dest whose value is ``None``;
* the new dests are disjoint from their host's (argparse writes a child's defaults over the
  parent's parsed values, so a shared dest silently clobbers a host option);
* the compensation for ``--ckpt`` losing ``required=True``: bare ``eval`` still exits 2 and
  never reaches the checkpoint path;
* mixed E/H invocations are refused with exit 2 naming the argument;
* ``mlflow-aggregate h`` and ``eval h`` never construct an MLflow client, even with a tracking
  URI in the environment; ``charts h`` attaches only on ``--attach`` and fails loud;
* exactly one JSON document on stdout per H invocation.

Fixture-backed tests are ``integration`` and raise when the capture directory is absent.
"""

from __future__ import annotations

import argparse
import ast
import json
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

from entropic_substrate.composition.entrypoints import cli as cli_module
from entropic_substrate.composition.entrypoints.cli import build_parser, main

MLFLOW_BUILDER = "entropic_substrate.ladder.capture.adapters.mlflow.build_mlflow_client"


@pytest.fixture(autouse=True)
def _no_tracking_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)


def run_main(argv: list[str]) -> int:
    """The exit code whether ``main`` returned it or argparse raised ``SystemExit``."""
    try:
        return main(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1


# ------------------------------------------------------ M5: the byte-identical E namespaces
# Dumped from the parser at 501c0447 with MLFLOW_EXPERIMENT_NAME and MLFLOW_TRACKING_URI
# removed from the environment (the --experiment default reads os.environ at build time).
_CHARTS_E: dict[str, Any] = {
    "command": "charts",
    "config_hash": None,
    "e1_control": [],
    "e1_epsilon": None,
    "e1_fisher": [],
    "e2": None,
    "e2_run_id": None,
    "e3": None,
    "e3_run_id": None,
    "experiment": "entropic-substrate-lm",
    "out_dir": Path("results/charts"),
    "tracking_uri": None,
}
_EVAL_E: dict[str, Any] = {
    "batches": None,
    "ckpt": Path("ck.pt"),
    "command": "eval",
    "device": None,
    "json": False,
}
_AGGREGATE_E: dict[str, Any] = {
    "command": "mlflow-aggregate",
    "experiment": "entropic-substrate-lm",
    "n_seeds": 3,
    "tracking_uri": None,
}

_E_NAMESPACES: list[tuple[list[str], dict[str, Any], str]] = [
    (["eval", "--ckpt", "ck.pt"], _EVAL_E, "eval_lane"),
    (
        ["eval", "--ckpt", "ck.pt", "--device", "cpu", "--batches", "1"],
        {**_EVAL_E, "device": "cpu", "batches": 1},
        "eval_lane",
    ),
    (
        ["eval", "--ckpt", "ck.pt", "--device", "cpu", "--batches", "1", "--json"],
        {**_EVAL_E, "device": "cpu", "batches": 1, "json": True},
        "eval_lane",
    ),
    (["charts"], _CHARTS_E, "charts_lane"),
    (["charts", "--out-dir", "out"], {**_CHARTS_E, "out_dir": Path("out")}, "charts_lane"),
    (
        ["charts", "--e2", "e2.jsonl", "--e3", "e3.jsonl", "--out-dir", "out"],
        {**_CHARTS_E, "e2": Path("e2.jsonl"), "e3": Path("e3.jsonl"), "out_dir": Path("out")},
        "charts_lane",
    ),
    (
        [
            "charts",
            "--e1-fisher",
            "f1.jsonl",
            "--e1-fisher",
            "f2.jsonl",
            "--e1-control",
            "c1.jsonl",
            "--e1-control",
            "c2.jsonl",
            "--e1-epsilon",
            "0.02",
            "--out-dir",
            "out",
        ],
        {
            **_CHARTS_E,
            "e1_fisher": [Path("f1.jsonl"), Path("f2.jsonl")],
            "e1_control": [Path("c1.jsonl"), Path("c2.jsonl")],
            "e1_epsilon": 0.02,
            "out_dir": Path("out"),
        },
        "charts_lane",
    ),
    (
        [
            "charts",
            "--e1-fisher",
            "f1.jsonl",
            "--e1-control",
            "c1.jsonl",
            "--config-hash",
            "deadbeefcafe",
            "--experiment",
            "exp",
            "--tracking-uri",
            "sqlite:///x.db",
            "--out-dir",
            "out",
        ],
        {
            **_CHARTS_E,
            "e1_fisher": [Path("f1.jsonl")],
            "e1_control": [Path("c1.jsonl")],
            "config_hash": "deadbeefcafe",
            "experiment": "exp",
            "tracking_uri": "sqlite:///x.db",
            "out_dir": Path("out"),
        },
        "charts_lane",
    ),
    (
        [
            "charts",
            "--e2",
            "e2.jsonl",
            "--e2-run-id",
            "r2",
            "--tracking-uri",
            "sqlite:///x.db",
            "--out-dir",
            "out",
        ],
        {
            **_CHARTS_E,
            "e2": Path("e2.jsonl"),
            "e2_run_id": "r2",
            "tracking_uri": "sqlite:///x.db",
            "out_dir": Path("out"),
        },
        "charts_lane",
    ),
    (
        [
            "charts",
            "--e2",
            "e2.jsonl",
            "--e3",
            "e3.jsonl",
            "--e3-run-id",
            "r3",
            "--tracking-uri",
            "sqlite:///x.db",
            "--out-dir",
            "out",
        ],
        {
            **_CHARTS_E,
            "e2": Path("e2.jsonl"),
            "e3": Path("e3.jsonl"),
            "e3_run_id": "r3",
            "tracking_uri": "sqlite:///x.db",
            "out_dir": Path("out"),
        },
        "charts_lane",
    ),
    (["mlflow-aggregate"], _AGGREGATE_E, "aggregate_lane"),
    (
        ["mlflow-aggregate", "--tracking-uri", "sqlite:///x.db", "--experiment", "cli-growth"],
        {**_AGGREGATE_E, "tracking_uri": "sqlite:///x.db", "experiment": "cli-growth"},
        "aggregate_lane",
    ),
    (
        ["mlflow-aggregate", "--tracking-uri", "sqlite:///x.db"],
        {**_AGGREGATE_E, "tracking_uri": "sqlite:///x.db"},
        "aggregate_lane",
    ),
    (["mlflow-aggregate", "--n-seeds", "5"], {**_AGGREGATE_E, "n_seeds": 5}, "aggregate_lane"),
]


@pytest.mark.unit
class TestExistingInvocationsKeepTheirNamespace:
    @pytest.mark.parametrize(
        ("argv", "expected", "lane_dest"),
        _E_NAMESPACES,
        ids=[" ".join(a)[:70] for a, _, _ in _E_NAMESPACES],
    )
    def test_namespace_equals_the_literal_dumped_at_501c0447(
        self, argv: list[str], expected: dict[str, Any], lane_dest: str
    ) -> None:
        actual = vars(build_parser().parse_args(argv))
        assert set(actual) - set(expected) == {lane_dest}
        assert actual.pop(lane_dest) is None
        assert actual == expected

    @pytest.mark.parametrize(
        "argv",
        [["eval", "extra"], ["charts", "extra"], ["mlflow-aggregate", "extra"], ["eval", "h"]],
    )
    def test_error_shapes_still_exit_2(self, argv: list[str]) -> None:
        assert run_main(argv) == 2

    def test_the_e_gates_are_unchanged_for_a_bare_invocation(self) -> None:
        assert run_main(["mlflow-aggregate"]) == 2  # no tracking URI
        assert run_main(["charts"]) == 2  # no chart inputs


def _subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    action: Any = next(
        a
        for a in parser._actions
        if isinstance(a, argparse._SubParsersAction)  # pyright: ignore[reportPrivateUsage]
    )
    return action.choices[name]


def _dests(parser: argparse.ArgumentParser) -> set[str]:
    return {a.dest for a in parser._actions if a.dest not in {"help", argparse.SUPPRESS}}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.unit
class TestNewSurfaceShape:
    @pytest.mark.parametrize(
        ("host", "lane"),
        [("eval", "eval_lane"), ("charts", "charts_lane"), ("mlflow-aggregate", "aggregate_lane")],
    )
    def test_h_dests_are_disjoint_from_the_host_dests(self, host: str, lane: str) -> None:
        host_parser = _subparser(build_parser(), host)
        h_parser = _subparser(host_parser, "h")
        host_dests = _dests(host_parser)
        h_dests = _dests(h_parser)
        assert lane in host_dests
        assert (host_dests - {lane}) & h_dests == set()
        assert all(d.startswith("h_") for d in h_dests)

    def test_no_new_top_level_subcommand(self) -> None:
        top: Any = next(
            a
            for a in build_parser()._actions
            if isinstance(a, argparse._SubParsersAction)  # pyright: ignore[reportPrivateUsage]
        )
        assert len(top.choices) == 27
        assert not {"aggregate", "h-aggregate", "h-charts", "h-report"} & set(top.choices)

    def test_the_lane_is_optional_on_every_host(self) -> None:
        for host in ("eval", "charts", "mlflow-aggregate"):
            lane: Any = next(
                a
                for a in _subparser(build_parser(), host)._actions  # pyright: ignore[reportPrivateUsage]
                if isinstance(a, argparse._SubParsersAction)  # pyright: ignore[reportPrivateUsage]
            )
            assert lane.required is False

    def test_the_cli_module_imports_no_h_analysis_module_at_module_level(self) -> None:
        tree = ast.parse(Path(cli_module.__file__ or "").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom | ast.Import):
                names = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [a.name for a in node.names]
                )
                for name in names:
                    assert not name.startswith("entropic_substrate.ladder.analyze.h_"), name
                    assert not name.startswith("matplotlib")
                    assert not name.startswith("mlflow")

    def test_cmd_eval_keeps_its_raw_print_and_the_dispatch_table_its_shape(self) -> None:
        text = Path(cli_module.__file__ or "").read_text(encoding="utf-8")
        assert "print(json.dumps(payload, sort_keys=True))" in text
        assert text.count("import_module") == 0


# ------------------------------------------------------ eval: --ckpt demotion compensation
@pytest.mark.unit
class TestBareEvalNeverReachesTheCheckpointPath:
    def test_bare_eval_exits_2_and_never_loads_a_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_: Any, **__: Any) -> Any:
            raise AssertionError("the checkpoint path ran for a bare `eval`")

        monkeypatch.setattr(cli_module.torch, "load", boom)
        monkeypatch.setattr(cli_module, "normalize_checkpoint", boom)
        monkeypatch.setattr(cli_module, "resolve_device", boom)
        assert run_main(["eval"]) == 2
        assert run_main(["eval", "--json"]) == 2

    def test_the_refusal_names_the_missing_option(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert run_main(["eval"]) == 2
        assert "--ckpt" in capsys.readouterr().err


# ------------------------------------------------------ mixed E/H invocations are refused
def _agg_args(tmp_path: Path) -> list[str]:
    a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a")
    return [
        "h",
        "--run",
        str(a[0]),
        str(a[1]),
        "--arm",
        f"{IDENTITY_A}=magnitude",
    ]


@pytest.mark.unit
class TestMixedInvocationsAreRefused:
    @pytest.mark.parametrize(
        ("prefix", "flag"),
        [
            (["eval", "--ckpt", "x.pt"], "--ckpt"),
            (["eval", "--json"], "--json"),
            (["charts", "--e2", "x.jsonl"], "--e2"),
            (["charts", "--out-dir", "elsewhere"], "--out-dir"),
            (["charts", "--tracking-uri", "sqlite:///x.db"], "--tracking-uri"),
            (["mlflow-aggregate", "--n-seeds", "5"], "--n-seeds"),
            (["mlflow-aggregate", "--experiment", "other"], "--experiment"),
        ],
    )
    def test_a_host_option_set_away_from_its_default_is_refused_by_name(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        prefix: list[str],
        flag: str,
    ) -> None:
        tail = _agg_args(tmp_path)
        if prefix[0] == "eval":
            tail = tail[:4]  # `eval h` takes --run only; --arm is not one of its options
        code = run_main([*prefix, *tail])
        assert code == 2
        captured = capsys.readouterr()
        assert flag in captured.err
        assert captured.out == ""

    def test_the_child_default_never_clobbers_a_host_value_silently(self) -> None:
        # MEASURED on 3.12 at the architect gate: with a shared dest this parsed fine and the
        # child's default overwrote the parent's value. Disjoint dests make both visible.
        ns = build_parser().parse_args(
            ["charts", "--out-dir", "PARENT", "h", "--run", "m", "l", "--arm", f"{IDENTITY_A}=x"]
        )
        assert ns.out_dir == Path("PARENT")
        assert ns.h_out_dir == Path("results/h-charts")


# ------------------------------------------------------ H handlers, offline and stdout contract
def _one_document(captured: pytest.CaptureFixture[str]) -> Any:
    out = captured.readouterr().out
    assert out.count("\n") == 1, f"stdout must be exactly one JSON document, got {out!r}"
    return json.loads(out)


def _two_arm_args(tmp_path: Path) -> tuple[list[str], list[str]]:
    a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a", run_id=RUN_ID_A, scale=1.0)
    b = write_h_run(tmp_path, IDENTITY_B, REQUEST_B, tag="b", run_id=RUN_ID_B, scale=2.0)
    runs = ["--run", str(a[0]), str(a[1]), "--run", str(b[0]), str(b[1])]
    arms = ["--arm", f"{IDENTITY_A}=magnitude", "--arm", f"{IDENTITY_B}=sphere"]
    return runs, arms


@pytest.fixture
def mlflow_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dummy tracking URI AND a builder that raises: a URI-less run would be vacuous."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")

    def boom(*_: Any, **__: Any) -> Any:
        raise AssertionError("an MLflow client was constructed on an offline H path")

    monkeypatch.setattr(MLFLOW_BUILDER, boom)


@pytest.mark.unit
class TestAggregateH:
    def test_pairs_the_labelled_arms_and_prints_one_json_document(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        runs, arms = _two_arm_args(tmp_path)
        assert main(["mlflow-aggregate", "h", *runs, *arms]) == 0
        doc = _one_document(capsys)
        assert doc["schema"] == "h-aggregate-v1"
        assert sorted(doc["arms"]) == ["magnitude", "sphere"]
        assert doc["training"]["paired_steps"] == [1, 2, 3, 4, 5, 6]

    def test_never_constructs_an_mlflow_client_even_with_a_tracking_uri_set(
        self, tmp_path: Path, mlflow_forbidden: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        runs, arms = _two_arm_args(tmp_path)
        assert main(["mlflow-aggregate", "h", *runs, *arms]) == 0
        assert _one_document(capsys)["schema"] == "h-aggregate-v1"

    def test_works_without_any_tracking_uri_unlike_the_e_lane(self, tmp_path: Path) -> None:
        runs, arms = _two_arm_args(tmp_path)
        assert main(["mlflow-aggregate", "h", *runs, *arms]) == 0

    def test_an_unknown_identity_exits_2_and_names_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        runs, arms = _two_arm_args(tmp_path)
        unknown = "f" * 64
        assert main(["mlflow-aggregate", "h", *runs, *arms, "--arm", f"{unknown}=ghost"]) == 2
        captured = capsys.readouterr()
        assert unknown in captured.err
        assert captured.out == ""

    def test_an_unlabelled_run_exits_2_and_names_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        runs, _ = _two_arm_args(tmp_path)
        assert main(["mlflow-aggregate", "h", *runs, "--arm", f"{IDENTITY_A}=magnitude"]) == 2
        assert IDENTITY_B in capsys.readouterr().err

    def test_a_short_identity_is_not_accepted(self, tmp_path: Path) -> None:
        runs, _ = _two_arm_args(tmp_path)
        assert main(["mlflow-aggregate", "h", *runs, "--arm", "00000000=a"]) == 2

    def test_inadmissible_telemetry_exits_2_names_the_file_and_prints_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def drop_run_end(rows: list[dict[str, Any]]) -> None:
            rows[:] = [r for r in rows if r.get("event") != "run_end"]

        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, mutate_metrics=drop_run_end)
        code = main(
            ["mlflow-aggregate", "h", "--run", str(a[0]), str(a[1]), "--arm", f"{IDENTITY_A}=x"]
        )
        assert code == 2
        captured = capsys.readouterr()
        assert str(a[0]) in captured.err
        assert captured.out == ""

    def test_h_options_are_required(self, tmp_path: Path) -> None:
        assert run_main(["mlflow-aggregate", "h"]) == 2
        runs, _ = _two_arm_args(tmp_path)
        assert run_main(["mlflow-aggregate", "h", *runs]) == 2  # no --arm


class _StubClient:
    def __init__(self, fail_run: str | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_run = fail_run

    def log_figure(self, run_id: str, figure: object, artifact_file: str) -> None:
        del figure
        if run_id == self.fail_run:
            raise ConnectionError("server unreachable")
        self.calls.append((run_id, artifact_file))


@pytest.mark.unit
class TestChartsH:
    def test_renders_the_png_set_locally_and_attaches_nothing_by_default(
        self, tmp_path: Path, mlflow_forbidden: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        runs, arms = _two_arm_args(tmp_path / "in")
        out = tmp_path / "charts"
        assert main(["charts", "h", *runs, *arms, "--out-dir", str(out)]) == 0
        doc = _one_document(capsys)
        assert doc["attached"] == {}
        assert doc["absent"] == {}
        names = {p.name for p in out.glob("*.png")}
        assert names == {
            "h_evaluation_rows.png",
            "h_future_nll_per_dimension.png",
            "h_publication_timing.png",
            "h_training_curves.png",
        }
        assert all(p.stat().st_size > 0 for p in out.glob("*.png"))
        assert sorted(doc["written"]) == sorted(str(p) for p in out.glob("*.png"))

    def test_attach_without_a_tracking_uri_is_refused_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        runs, arms = _two_arm_args(tmp_path / "in")
        out = tmp_path / "charts"
        assert main(["charts", "h", *runs, *arms, "--out-dir", str(out), "--attach"]) == 2
        assert not out.exists()

    def test_attach_publishes_every_figure_to_every_arms_run_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        stub = _StubClient()
        seen: list[str] = []

        def build(uri: str) -> _StubClient:
            seen.append(uri)
            return stub

        monkeypatch.setattr(MLFLOW_BUILDER, build)
        runs, arms = _two_arm_args(tmp_path / "in")
        code = main(
            [
                "charts",
                "h",
                *runs,
                *arms,
                "--out-dir",
                str(tmp_path / "charts"),
                "--attach",
                "--tracking-uri",
                "sqlite:///stub.db",
            ]
        )
        assert code == 0
        assert seen == ["sqlite:///stub.db"]
        # the run ids come from the captured worker log, never from a file-name fragment
        assert {run for run, _ in stub.calls} == {RUN_ID_A, RUN_ID_B}
        assert len(stub.calls) == 2 * 4
        doc = _one_document(capsys)
        assert sorted(doc["attached"]) == [RUN_ID_A, RUN_ID_B]

    def test_a_failing_run_fails_the_command_after_every_run_was_attempted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        stub = _StubClient(fail_run=RUN_ID_A)
        monkeypatch.setattr(MLFLOW_BUILDER, lambda uri: stub)
        runs, arms = _two_arm_args(tmp_path / "in")
        code = main(
            [
                "charts",
                "h",
                *runs,
                *arms,
                "--out-dir",
                str(tmp_path / "c"),
                "--attach",
                "--tracking-uri",
                "x",
            ]
        )
        assert code != 0
        captured = capsys.readouterr()
        assert RUN_ID_A in captured.err
        assert {run for run, _ in stub.calls} == {RUN_ID_B}  # the other run was still attempted
        doc = json.loads(captured.out)
        assert set(doc["attach_failures"]) == {RUN_ID_A}
        assert (
            tmp_path / "c" / "h_training_curves.png"
        ).exists()  # rendering never depended on MLflow

    def test_a_run_without_a_captured_mlflow_run_id_cannot_be_attached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stub = _StubClient()
        monkeypatch.setattr(MLFLOW_BUILDER, lambda uri: stub)
        a = write_h_run(tmp_path / "in", IDENTITY_A, REQUEST_A, run_id=None)
        code = main(
            [
                "charts",
                "h",
                "--run",
                str(a[0]),
                str(a[1]),
                "--arm",
                f"{IDENTITY_A}=lone",
                "--out-dir",
                str(tmp_path / "c"),
                "--attach",
                "--tracking-uri",
                "x",
            ]
        )
        assert code != 0
        assert "lone" in capsys.readouterr().err
        assert stub.calls == []

    def test_inadmissible_telemetry_writes_no_chart(self, tmp_path: Path) -> None:
        def drop_run_end(rows: list[dict[str, Any]]) -> None:
            rows[:] = [r for r in rows if r.get("event") != "run_end"]

        a = write_h_run(tmp_path / "in", IDENTITY_A, REQUEST_A, mutate_metrics=drop_run_end)
        out = tmp_path / "charts"
        code = main(
            [
                "charts",
                "h",
                "--run",
                str(a[0]),
                str(a[1]),
                "--arm",
                f"{IDENTITY_A}=x",
                "--out-dir",
                str(out),
            ]
        )
        assert code == 2
        assert not out.exists()


@pytest.mark.unit
class TestEvalH:
    def test_reports_one_run_as_one_json_document(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A)
        assert main(["eval", "h", "--run", str(a[0]), str(a[1])]) == 0
        doc = _one_document(capsys)
        assert doc["schema"] == "h-report-v1"
        assert doc["lineage"]["profile_identity"] == IDENTITY_A
        assert doc["evaluation"]["steps"] == [4, 6]

    def test_never_constructs_an_mlflow_client_even_with_a_tracking_uri_set(
        self, tmp_path: Path, mlflow_forbidden: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A)
        assert main(["eval", "h", "--run", str(a[0]), str(a[1])]) == 0
        assert _one_document(capsys)["schema"] == "h-report-v1"

    def test_inadmissible_telemetry_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def drop(rows: list[dict[str, Any]]) -> None:
            rows[:] = [r for r in rows if r.get("event") != "run_end"]

        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, mutate_metrics=drop)
        assert main(["eval", "h", "--run", str(a[0]), str(a[1])]) == 2
        captured = capsys.readouterr()
        assert str(a[0]) in captured.err
        assert captured.out == ""

    def test_the_receipt_pair_delegates_to_h_inspect_training_and_embeds_its_verdict(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        received: list[argparse.Namespace] = []

        def fake_sentence(args: argparse.Namespace) -> int:
            received.append(args)
            print(json.dumps({"status": "complete", "scientific_authority": "none"}))
            return 0

        monkeypatch.setattr("entropic_substrate.composition.cli.cmd_sentence", fake_sentence)
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A)
        code = main(
            [
                "eval",
                "h",
                "--run",
                str(a[0]),
                str(a[1]),
                "--receipt-root",
                str(tmp_path),
                "--receipt-manifest",
                "inspect.json",
            ]
        )
        assert code == 0
        doc = _one_document(capsys)  # the delegate's own line was captured, not emitted
        assert doc["receipt_verdict"] == {"scientific_authority": "none", "status": "complete"}
        assert doc["receipt_exit_code"] == 0
        assert received[0].command == "h-inspect-training"
        assert received[0].root == tmp_path
        assert received[0].manifest == "inspect.json"

    def test_a_refusing_delegate_makes_the_command_exit_with_its_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def refusing(args: argparse.Namespace) -> int:
            del args
            print(json.dumps({"status": "invalid_request"}))
            return 2

        monkeypatch.setattr("entropic_substrate.composition.cli.cmd_sentence", refusing)
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A)
        code = main(
            [
                "eval",
                "h",
                "--run",
                str(a[0]),
                str(a[1]),
                "--receipt-root",
                str(tmp_path),
                "--receipt-manifest",
                "m",
            ]
        )
        assert code == 2
        doc = _one_document(capsys)
        assert doc["receipt_verdict"] == {"status": "invalid_request"}

    def test_a_half_receipt_pair_is_refused(self, tmp_path: Path) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A)
        assert (
            main(["eval", "h", "--run", str(a[0]), str(a[1]), "--receipt-root", str(tmp_path)]) == 2
        )
        assert main(["eval", "h", "--run", str(a[0]), str(a[1]), "--receipt-manifest", "m"]) == 2


# ------------------------------------------------------ integration: the real HEpP7g capture
def _fixture_cli_args() -> tuple[list[str], list[str]]:
    pairs, labels = fixture_arguments()
    runs = [x for metrics, log in pairs for x in ("--run", str(metrics), str(log))]
    arms = [x for identity, label in labels.items() for x in ("--arm", f"{identity}={label}")]
    return runs, arms


@pytest.mark.integration
class TestEndToEndOnTheCapturedRuns:
    def test_aggregate_h(self, capsys: pytest.CaptureFixture[str]) -> None:
        runs, arms = _fixture_cli_args()
        assert main(["mlflow-aggregate", "h", *runs, *arms]) == 0
        doc = _one_document(capsys)
        assert doc["training"]["paired_steps"] == list(range(1, 1001))
        assert doc["evaluation"]["paired_steps"] == [500, 750, 1000]
        assert doc["publication_seconds"]["paired_steps"] == [250, 500, 750, 1000]

    def test_charts_h(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        runs, arms = _fixture_cli_args()
        assert main(["charts", "h", *runs, *arms, "--out-dir", str(tmp_path)]) == 0
        doc = _one_document(capsys)
        assert doc["absent"] == {}
        assert {Path(p).name for p in doc["written"]} == {
            "h_evaluation_rows.png",
            "h_future_nll_per_dimension.png",
            "h_publication_timing.png",
            "h_training_curves.png",
        }
        assert all((tmp_path / Path(p).name).stat().st_size > 0 for p in doc["written"])

    @pytest.mark.parametrize("which", [0, 1])
    def test_eval_h(self, which: int, capsys: pytest.CaptureFixture[str]) -> None:
        pairs, _ = fixture_arguments()
        metrics, log = pairs[which]
        assert main(["eval", "h", "--run", str(metrics), str(log)]) == 0
        doc = _one_document(capsys)
        assert doc["evaluation"]["steps"] == [500, 750, 1000]
        assert doc["steps"]["range"] == [1, 1000]
        assert doc["status"] == {
            "completed": True,
            "result_status": "probe_workload_complete",
            "status": "ok",
            "successful_optimizer_steps": 1000,
            "wall_seconds": doc["status"]["wall_seconds"],
        }
        stages = {s["stage"] for s in doc["final_leg"]["stages"]}
        assert {"training_readback", "generation", "span_diagnostics", "trace_capture"} <= stages

    def test_an_unlabelled_captured_run_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        runs, arms = _fixture_cli_args()
        assert main(["mlflow-aggregate", "h", *runs, *arms[:2]]) == 2
        assert "no --arm label" in capsys.readouterr().err
