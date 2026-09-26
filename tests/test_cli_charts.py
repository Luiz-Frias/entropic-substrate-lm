"""The ``charts`` CLI subcommand — local rendering always, MLflow attach only
when a tracking URI is reachable. Attach-path tests use a REAL, local,
sqlite-backed ``mlflow.tracking.MlflowClient`` (same idiom as
tests/test_mlflow_aggregate.py's CLI section), because ``log_figure`` writes
actual artifact bytes that a structural fake would not meaningfully verify.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from entropic_substrate.composition.entrypoints.cli import main
from entropic_substrate.ladder.capture.reader.mlflow_aggregate import CONFIG_HASH_TAG, PARENT_TAG

pytestmark = pytest.mark.unit


def _write(path: Path, *records: dict[str, Any]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _run_start(seed: int) -> dict[str, Any]:
    return {"event": "run_start", "step": 0, "config": {"train": {"seed": seed}}}


def _train(step: int, loss: float) -> dict[str, Any]:
    return {"event": "train", "step": step, "loss": loss, "nats_per_byte": loss / 2}


def _eval(step: int, npb: float, kappa: list[list[float]] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"event": "eval", "step": step, "loss": npb * 2, "nats_per_byte": npb}
    if kappa is not None:
        record["kappa"] = kappa
    return record


def _e2_jsonl(path: Path, seed: int = 1) -> Path:
    return _write(
        path,
        _run_start(seed),
        _train(1, 3.0),
        _eval(10, 1.8),
        _train(11, 2.6),
        _eval(20, 1.5),
    )


def _e3_jsonl(path: Path, seed: int = 1) -> Path:
    return _write(
        path,
        _run_start(seed),
        _train(1, 2.9),
        _eval(10, 1.7, kappa=[[-0.5, 0.2], [0.1, -0.3]]),
        _train(11, 2.4),
        _eval(20, 1.3, kappa=[[-0.7, 0.4], [0.2, -0.1]]),
    )


def _e1_jsonl(path: Path, seed: int, base: float) -> Path:
    return _write(path, _run_start(seed), _eval(10, base), _eval(20, base - 0.2))


# ------------------------------------------------------------ local rendering
def test_charts_renders_e2_e3_locally_without_a_tracking_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    e2 = _e2_jsonl(tmp_path / "e2.jsonl")
    e3 = _e3_jsonl(tmp_path / "e3.jsonl")
    out_dir = tmp_path / "charts"

    code = main(["charts", "--e2", str(e2), "--e3", str(e3), "--out-dir", str(out_dir)])

    assert code == 0
    for name in (
        "e2_spike_variance_texture",
        "e3_loss_separation_vs_e2",
        "e3_kappa_divergence_vs_loss_gap",
        "e3_kappa_trajectories",
    ):
        assert (out_dir / f"{name}.png").exists()


def test_charts_renders_e1_paired_charts_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fisher = [_e1_jsonl(tmp_path / f"f{s}.jsonl", s, 1.5) for s in (1, 2, 3)]
    control = [_e1_jsonl(tmp_path / f"c{s}.jsonl", s, 1.6) for s in (1, 2, 3)]
    out_dir = tmp_path / "charts"

    code = main(
        [
            "charts",
            *[arg for p in fisher for arg in ("--e1-fisher", str(p))],
            *[arg for p in control for arg in ("--e1-control", str(p))],
            "--e1-epsilon",
            "0.02",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    assert (out_dir / "e1_paired_loss_curves.png").exists()
    assert (out_dir / "e1_log_steps_washout.png").exists()


def test_charts_with_no_inputs_is_a_controlled_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    code = main(["charts", "--out-dir", str(tmp_path / "charts")])
    assert code == 2


def test_charts_with_a_duplicate_seed_within_one_e1_arm_is_a_controlled_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MEASURED gap this pins: ``pair_final_metrics`` raises ``ValueError``
    on a same-arm duplicate seed (see
    tests/test_analysis_series.py::test_pair_final_metrics_rejects_a_duplicate_seed_within_one_arm),
    but ``cmd_charts``'s ``try`` block only caught ``TelemetryGap`` — so this
    ValueError propagated straight through ``main()`` as an unhandled
    exception (a raw traceback) instead of the controlled exit-2 idiom every
    OTHER input-validation refusal in this subcommand uses (see
    ``--e1-fisher``/``--e1-control`` must-both-be-given, above)."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fisher_1a = _e1_jsonl(tmp_path / "f1a.jsonl", 1, 1.5)
    fisher_1b = _e1_jsonl(tmp_path / "f1b.jsonl", 1, 1.9)  # same seed as f1a
    control_1 = _e1_jsonl(tmp_path / "c1.jsonl", 1, 1.6)

    code = main(
        [
            "charts",
            "--e1-fisher",
            str(fisher_1a),
            "--e1-fisher",
            str(fisher_1b),
            "--e1-control",
            str(control_1),
            "--out-dir",
            str(tmp_path / "charts"),
        ]
    )

    assert code == 2
    assert "seed" in capsys.readouterr().err.lower()


def test_charts_with_only_e1_fisher_is_a_controlled_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fisher = _e1_jsonl(tmp_path / "f1.jsonl", 1, 1.5)
    code = main(["charts", "--e1-fisher", str(fisher), "--out-dir", str(tmp_path / "charts")])
    assert code == 2


def test_charts_propagates_a_telemetry_gap_as_a_controlled_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Measurement path: fail LOUD — a NaN-poisoned run must never render a
    blank or partial chart. Guard-tooth for this CLI's TelemetryGap handling:
    without the except clause this crashes with an unhandled TelemetryGap
    traceback instead of a controlled exit 2."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    poisoned = _write(tmp_path / "e2.jsonl", _run_start(1), _eval(10, math.nan))
    code = main(["charts", "--e2", str(poisoned), "--out-dir", str(tmp_path / "charts")])
    assert code == 2
    assert "telemetry gap" in capsys.readouterr().err


def test_charts_telemetry_gap_after_e1_figures_already_built_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Guard-tooth for the partial-figure hazard the sibling test above does
    NOT cover: that test poisons a LONE --e2 input, where the gap fires
    before ``figures`` holds anything. Here E1 is valid (its two figures are
    built and already sitting in ``figures``) and E3 is what's poisoned, so
    the gap fires strictly AFTER at least one figure exists — this is what
    would catch a regression from ``except TelemetryGap: return
    EXIT_TELEMETRY_GAP`` to a log-and-continue handler, which the sibling
    test's early-gap shape cannot distinguish from correct behavior. Refusal
    must be WHOLESALE: no e1 PNGs on disk either, not just no e3 PNGs."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fisher = [_e1_jsonl(tmp_path / f"f{s}.jsonl", s, 1.5) for s in (1, 2, 3)]
    control = [_e1_jsonl(tmp_path / f"c{s}.jsonl", s, 1.6) for s in (1, 2, 3)]
    poisoned_e3 = _write(tmp_path / "e3.jsonl", _run_start(1), _eval(10, math.nan))
    out_dir = tmp_path / "charts"

    code = main(
        [
            "charts",
            *[arg for p in fisher for arg in ("--e1-fisher", str(p))],
            *[arg for p in control for arg in ("--e1-control", str(p))],
            "--e3",
            str(poisoned_e3),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 2
    assert "telemetry gap" in capsys.readouterr().err
    assert not out_dir.exists() or not list(out_dir.glob("*.png"))


def test_cli_module_has_no_unconditional_analysis_or_matplotlib_import() -> None:
    """Guard-tooth mirroring tests/test_sinks.py's
    ``test_sinks_module_has_no_unconditional_psutil_or_pynvml_import`` AST
    walk: cli.py must not grow a module-level import of matplotlib, marimo,
    plotly, or entropic_substrate.analysis. The ``charts`` subcommand's
    lazy-import idiom (mirrors ``cmd_capture_results``) is what keeps these
    dev-only analysis deps out of every OTHER subcommand's import graph on a
    training pod — a hoisted module-level import here would make `train`/
    `eval`/etc. pay that import cost (or hard-dependency risk) for a feature
    they never touch. MODULE-LEVEL statements only (``tree.body``): a lazy
    import inside ``cmd_charts``'s function body is exactly the sanctioned
    shape and must not false-positive here.
    """
    import ast

    import entropic_substrate.composition.entrypoints.cli as cli_module

    source = Path(cli_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    # P13-EXCEPTIONS (addendum 2): composed at runtime, never a literal chain
    # -- the retired-namespace prefix must stay unresolvable forever for the
    # ban to mean anything, and a literal here would itself be flagged by
    # tests/test_no_old_prefix.py's whole-tree scan.
    banned_prefixes = ("matplotlib", "marimo", "plotly", "entropic_substrate" + ".analysis")
    for node in tree.body:
        if isinstance(node, ast.Import):
            dotted_names = [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            dotted_names = [(node.module or "", node.lineno)]
        else:
            continue
        for name, lineno in dotted_names:
            hit = [
                prefix
                for prefix in banned_prefixes
                if name == prefix or name.startswith(prefix + ".")
            ]
            assert not hit, f"cli.py must not import {name!r} at module level — line {lineno}"


# --------------------------------------------------------------- MLflow attach
def test_charts_attaches_per_run_figures_by_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # keeps the sqlite backend's ./mlruns artifact root local
    import mlflow.tracking

    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    exp_id = client.create_experiment("exp")
    e3_run = client.create_run(exp_id)

    e2 = _e2_jsonl(tmp_path / "e2.jsonl")
    e3 = _e3_jsonl(tmp_path / "e3.jsonl")

    code = main(
        [
            "charts",
            "--e2",
            str(e2),
            "--e3",
            str(e3),
            "--e3-run-id",
            e3_run.info.run_id,
            "--tracking-uri",
            tracking_uri,
            "--out-dir",
            str(tmp_path / "charts"),
        ]
    )

    assert code == 0
    artifact_paths = {a.path for a in client.list_artifacts(e3_run.info.run_id, "charts")}
    assert "charts/e3_kappa_trajectories.png" in artifact_paths
    assert "charts/e3_loss_separation_vs_e2.png" in artifact_paths


def test_charts_attaches_e1_seed_band_to_the_aggregate_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    import mlflow.tracking

    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    exp_id = client.create_experiment("exp")
    parent = client.create_run(exp_id, tags={PARENT_TAG: "1", CONFIG_HASH_TAG: "deadbeefcafe"})

    fisher = [_e1_jsonl(tmp_path / f"f{s}.jsonl", s, 1.5) for s in (1, 2)]
    control = [_e1_jsonl(tmp_path / f"c{s}.jsonl", s, 1.6) for s in (1, 2)]

    code = main(
        [
            "charts",
            *[arg for p in fisher for arg in ("--e1-fisher", str(p))],
            *[arg for p in control for arg in ("--e1-control", str(p))],
            "--config-hash",
            "deadbeefcafe",
            "--experiment",
            "exp",
            "--tracking-uri",
            tracking_uri,
            "--out-dir",
            str(tmp_path / "charts"),
        ]
    )

    assert code == 0
    artifact_paths = {a.path for a in client.list_artifacts(parent.info.run_id, "charts")}
    assert "charts/e1_paired_loss_curves.png" in artifact_paths
    assert "charts/e1_log_steps_washout.png" in artifact_paths


def test_charts_with_a_trailing_newline_on_config_hash_is_a_controlled_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MEASURED regression this pins: ``re.match`` (used pre-fix) does not
    require a match to reach the end of the string — Python's ``$`` anchor
    matches at end-of-string OR immediately before a single trailing
    ``\\n``, so a value like ``'deadbeefcafe\\n'`` (e.g. an unstripped
    ``$(cat file)`` or an env var carrying a trailing newline) satisfied
    ``_CONFIG_HASH_RE.match(...)`` despite not being a clean 12-char hex
    digest. ``fullmatch`` has no such carve-out. Exercises the actual CLI
    validation path (not the regex object in isolation), otherwise this
    test would pass regardless of whether ``cmd_charts`` used ``match`` or
    ``fullmatch``.
    """
    monkeypatch.chdir(tmp_path)
    import mlflow.tracking

    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.tracking.MlflowClient(tracking_uri=tracking_uri).create_experiment("exp")
    fisher = [_e1_jsonl(tmp_path / f"f{s}.jsonl", s, 1.5) for s in (1, 2)]
    control = [_e1_jsonl(tmp_path / f"c{s}.jsonl", s, 1.6) for s in (1, 2)]

    code = main(
        [
            "charts",
            *[arg for p in fisher for arg in ("--e1-fisher", str(p))],
            *[arg for p in control for arg in ("--e1-control", str(p))],
            "--config-hash",
            "deadbeefcafe\n",
            "--experiment",
            "exp",
            "--tracking-uri",
            tracking_uri,
            "--out-dir",
            str(tmp_path / "charts"),
        ]
    )

    assert code == 2
    assert "config-hash" in capsys.readouterr().err.lower()


def test_charts_with_a_malformed_config_hash_is_a_controlled_refusal_not_a_silent_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MEASURED gap this pins: ``--config-hash`` is user-typed input (unlike
    ``mlflow-aggregate``, which always computes it internally via
    ``entropic_substrate.sinks.config_hash`` — 12 lowercase hex chars). A
    typo here previously fell into ``attach_seed_band_to_aggregate``'s
    best-effort ``except Exception`` net and was silently logged as 'no
    aggregate parent yet' — indistinguishable from the LEGITIMATE case where
    mlflow-aggregate genuinely has not run yet. 'STOP and ask > silent
    fallback' (repo NON-NEGOTIABLE): a malformed hash is a caller mistake,
    not an absent-parent condition, and must exit non-zero rather than
    silently no-op. Local rendering still succeeds first — SACRED
    INVARIANTS: local rendering never depends on the tracking server."""
    monkeypatch.chdir(tmp_path)
    import mlflow.tracking

    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.tracking.MlflowClient(tracking_uri=tracking_uri).create_experiment("exp")
    fisher = [_e1_jsonl(tmp_path / f"f{s}.jsonl", s, 1.5) for s in (1, 2)]
    control = [_e1_jsonl(tmp_path / f"c{s}.jsonl", s, 1.6) for s in (1, 2)]
    out_dir = tmp_path / "charts"

    code = main(
        [
            "charts",
            *[arg for p in fisher for arg in ("--e1-fisher", str(p))],
            *[arg for p in control for arg in ("--e1-control", str(p))],
            "--config-hash",
            "not-a-hex-digest!!",
            "--experiment",
            "exp",
            "--tracking-uri",
            tracking_uri,
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 2
    assert "config-hash" in capsys.readouterr().err.lower()
    # Local PNGs still exist — the refusal is scoped to the MLflow attach
    # step, not a reason to withhold the already-rendered charts.
    assert (out_dir / "e1_paired_loss_curves.png").exists()


def test_charts_without_a_run_id_or_config_hash_renders_but_attaches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MLflow writes are never required for the charts to render locally —
    a reachable tracking URI with NO run_id/config_hash given must still
    return 0 and leave the local PNGs in place."""
    monkeypatch.chdir(tmp_path)
    import mlflow.tracking

    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.tracking.MlflowClient(tracking_uri=tracking_uri).create_experiment("exp")
    e2 = _e2_jsonl(tmp_path / "e2.jsonl")
    out_dir = tmp_path / "charts"

    code = main(
        ["charts", "--e2", str(e2), "--tracking-uri", tracking_uri, "--out-dir", str(out_dir)]
    )

    assert code == 0
    assert (out_dir / "e2_spike_variance_texture.png").exists()
