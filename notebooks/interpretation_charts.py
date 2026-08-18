import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go

    from entropic_substrate.analysis.palette import (
        OKABE_ITO,
        control_style,
        e2_style,
        e3_style,
        fisher_style,
        head_style,
    )
    from entropic_substrate.analysis.series import (
        T_0_9875,
        load_eval_run,
        load_train_run,
        pair_final_metrics,
        paired_verdict,
    )
    from entropic_substrate.telemetry import max_kappa_spread

    def e1_traces(fisher_runs: list, control_runs: list) -> list[go.Scatter]:
        """The E1 fisher/control traces, shared by chart 1 (paired loss curves)
        and chart 2 (the log-step washout view) — those two charts differ only
        in LAYOUT. Colors/dash/marker are a redundant non-color encoding
        channel, not decoration, so two copies drifting apart would leave the
        two charts disagreeing about which arm is which."""
        traces = []
        for run in fisher_runs:
            traces.append(
                go.Scatter(
                    x=[r.step for r in run.records],
                    y=[r.nats_per_byte for r in run.records],
                    mode="lines+markers",
                    name=f"{fisher_style().label}: {run.label}",
                    line={"color": OKABE_ITO["blue"], "dash": "solid"},
                    marker={"symbol": "circle"},
                )
            )
        for run in control_runs:
            traces.append(
                go.Scatter(
                    x=[r.step for r in run.records],
                    y=[r.nats_per_byte for r in run.records],
                    mode="lines+markers",
                    name=f"{control_style().label}: {run.label}",
                    line={"color": OKABE_ITO["vermillion"], "dash": "dash"},
                    marker={"symbol": "square"},
                )
            )
        return traces

    return (
        OKABE_ITO,
        Path,
        T_0_9875,
        control_style,
        e1_traces,
        e2_style,
        e3_style,
        fisher_style,
        go,
        head_style,
        load_eval_run,
        load_train_run,
        max_kappa_spread,
        mo,
        os,
        pair_final_metrics,
        paired_verdict,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # E1 / E2 / E3 interpretation charts — Lane B (ad hoc, interactive)

    Pulls DVC-captured JSONLs from `results/captured/` (checked out with
    `dvc checkout` beforehand) — or, for a run not yet captured, falls
    back to the MLflow artifacts API. Every JSONL read on this page goes
    through `entropic_substrate.telemetry` (via
    `entropic_substrate.analysis.series` — the SAME data-prep layer Lane
    A's matplotlib charts use) — never an ad-hoc `json.loads` of a
    results file. Charts render thresholds/CIs on the figure; they
    never decide — the gates this repo runs on stay sanity+selection,
    judgment human.
    """)
    return


@app.cell
def _(mo):
    captured_dir = mo.ui.text(value="results/captured", label="captured JSONL directory")
    e1_fisher_glob = mo.ui.text(value="*e1-fisher*.jsonl", label="E1 fisher glob")
    e1_control_glob = mo.ui.text(value="*e1-random*.jsonl", label="E1 control glob")
    e2_glob = mo.ui.text(value="*e2-*.jsonl", label="E2 glob (first match)")
    e3_glob = mo.ui.text(value="*e3-*.jsonl", label="E3 glob (first match)")
    mlflow_tracking_uri = mo.ui.text(
        value="", label="MLflow tracking URI (fallback ONLY — local files win)"
    )
    mlflow_experiment = mo.ui.text(value="entropic-substrate-lm", label="MLflow experiment")
    mo.vstack(
        [
            captured_dir,
            e1_fisher_glob,
            e1_control_glob,
            e2_glob,
            e3_glob,
            mlflow_tracking_uri,
            mlflow_experiment,
        ]
    )
    return (
        captured_dir,
        e1_control_glob,
        e1_fisher_glob,
        e2_glob,
        e3_glob,
        mlflow_experiment,
        mlflow_tracking_uri,
    )


@app.cell
def _(Path, captured_dir, mlflow_experiment, mlflow_tracking_uri, os):
    def _download_missing_from_mlflow(dest: Path) -> None:
        """DOCUMENTED FALLBACK ONLY (never the primary path — the primary
        path is DVC): pull FINISHED runs' results/*.jsonl artifacts down via
        the MLflow artifacts API, using the SAME capture naming convention as
        entropic_substrate.capture_results.capture_finished_runs, so a run
        captured this way lands under a name `dvc add` on the directory
        would also produce."""
        uri = mlflow_tracking_uri.value.strip() or os.environ.get("MLFLOW_TRACKING_URI", "")
        if not uri:
            return
        import mlflow.artifacts
        import mlflow.tracking

        client = mlflow.tracking.MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(mlflow_experiment.value.strip())
        if experiment is None:
            return
        dest.mkdir(parents=True, exist_ok=True)
        page_token = None
        while True:
            runs = client.search_runs(
                [experiment.experiment_id], max_results=500, page_token=page_token
            )
            for run in runs:
                if run.info.status != "FINISHED":
                    continue
                for artifact in client.list_artifacts(run.info.run_id, "results"):
                    # A FileInfo for a sub-directory has no bytes to download —
                    # only leaf .jsonl entries are downloadable artifacts.
                    if artifact.is_dir or not artifact.path.endswith(".jsonl"):
                        continue
                    # run_name is optional on the tracking server; an unnamed
                    # run would otherwise capture as a literal "None-<id>-...".
                    run_label = run.info.run_name or run.info.run_id
                    name = f"{run_label}-{run.info.run_id[:8]}-{Path(artifact.path).name}"
                    target = dest / name
                    if target.exists():
                        continue
                    downloaded = Path(
                        mlflow.artifacts.download_artifacts(
                            run_id=run.info.run_id, artifact_path=artifact.path, tracking_uri=uri
                        )
                    )
                    # Stage-then-rename: a write interrupted mid-flight would
                    # otherwise leave a truncated *.jsonl that the `target.exists()`
                    # skip above never re-downloads, and that read_eval_series then
                    # refuses forever. The .part suffix also keeps a partial file
                    # out of resolve_paths' *.jsonl glob while it's being written.
                    staged = target.with_name(target.name + ".part")
                    staged.write_bytes(downloaded.read_bytes())
                    os.replace(staged, target)
            page_token = runs.token
            if not page_token:
                break

    def resolve_paths(glob_pattern: str) -> list[Path]:
        """Local DVC-captured files first; the MLflow fallback only fires
        when the local glob comes back empty."""
        dest = Path(captured_dir.value)
        local = sorted(dest.glob(glob_pattern))
        if local:
            return local
        _download_missing_from_mlflow(dest)
        return sorted(dest.glob(glob_pattern))

    return (resolve_paths,)


@app.cell
def _(
    T_0_9875,
    e1_control_glob,
    e1_fisher_glob,
    e1_traces,
    go,
    load_eval_run,
    mo,
    pair_final_metrics,
    paired_verdict,
    resolve_paths,
):
    fisher_paths = resolve_paths(e1_fisher_glob.value)
    control_paths = resolve_paths(e1_control_glob.value)
    # Defined unconditionally (empty when no JSONLs are loaded yet) so the
    # washout cell (chart 2) below can depend on them reactively — a name
    # only bound inside one branch of an if/else would leave the downstream
    # cell with a NameError whenever this branch didn't run.
    fisher_runs: list = []
    control_runs: list = []

    if not fisher_paths or not control_paths:
        e1_output = mo.md("*(no E1 fisher/control JSONLs found under the captured directory yet)*")
    else:
        fisher_runs = [load_eval_run(p, label=p.stem) for p in fisher_paths]
        control_runs = [load_eval_run(p, label=p.stem) for p in control_paths]
        pairs, excluded_seeds = pair_final_metrics(fisher_paths, control_paths)
        verdict = paired_verdict(pairs)

        e1_fig = go.Figure()
        for _trace in e1_traces(fisher_runs, control_runs):
            e1_fig.add_trace(_trace)
        stage_note = (
            f" — Stage {verdict.stage}" if verdict.stage is not None else " — n not preregistered"
        )
        ci_note = (
            f"CI=[{verdict.ci_low:.4f}, {verdict.ci_high:.4f}]"
            if verdict.ci_low is not None and verdict.ci_high is not None
            else "CI not preregistered for this n"
        )
        e1_fig.add_annotation(
            text=(
                f"mean(d)={verdict.mean_d:.4f}  n={verdict.n}{stage_note}  {ci_note}  "
                f"(Stage 1 t={T_0_9875[4]}, Stage 2 t={T_0_9875[8]})"
            ),
            xref="paper",
            yref="paper",
            x=0.01,
            y=1.08,
            showarrow=False,
            font={"size": 11},
        )
        if excluded_seeds:
            e1_fig.add_annotation(
                text=f"excluded (unpaired) seeds: {excluded_seeds}",
                xref="paper",
                yref="paper",
                x=0.01,
                y=1.02,
                showarrow=False,
                font={"size": 10, "color": "dimgray"},
            )
        e1_fig.update_layout(
            title="E1: fisher vs control — paired loss curves (H1: mean(d) > 0)",
            xaxis_title="step",
            yaxis_title="nats/byte",
        )
        e1_output = mo.ui.plotly(e1_fig)

    e1_output
    return control_runs, fisher_runs


@app.cell
def _(control_runs, e1_traces, fisher_runs, go, mo):
    # --------------------------------------------------------- chart 2: washout
    if not fisher_runs or not control_runs:
        washout_output = mo.md(
            "*(no E1 fisher/control JSONLs found under the captured directory yet)*"
        )
    else:
        washout_fig = go.Figure()
        for _trace in e1_traces(fisher_runs, control_runs):
            washout_fig.add_trace(_trace)
        washout_fig.update_layout(
            title="E1: log-step washout view",
            xaxis_title="step (log scale)",
            xaxis_type="log",
            yaxis_title="nats/byte",
        )
        washout_output = mo.ui.plotly(washout_fig)

    washout_output
    return


@app.cell
def _(e2_glob, e3_glob, load_eval_run, load_train_run, mo, resolve_paths):
    e2_paths = resolve_paths(e2_glob.value)
    e3_paths = resolve_paths(e3_glob.value)

    e2_run = load_eval_run(e2_paths[0], label="e2") if e2_paths else None
    e3_run = load_eval_run(e3_paths[0], label="e3") if e3_paths else None
    e2_train_run = load_train_run(e2_paths[0], label="e2") if e2_paths else None

    if not e2_paths and not e3_paths:
        e2e3_output = mo.md("*(no E2/E3 JSONLs found under the captured directory yet)*")
    else:
        e2e3_output = mo.md(
            f"loaded: e2={e2_paths[0].name if e2_paths else 'missing'}, "
            f"e3={e3_paths[0].name if e3_paths else 'missing'}"
        )

    e2e3_output
    return e2_run, e2_train_run, e3_run


@app.cell
def _(OKABE_ITO, e2_train_run, go, mo):
    if e2_train_run is None:
        texture_output = mo.md("*(no E2 run loaded — skipping the spike/variance texture chart)*")
    else:
        texture_fig = go.Figure()
        texture_fig.add_trace(
            go.Scatter(
                x=[r.step for r in e2_train_run.records],
                y=[r.loss for r in e2_train_run.records],
                mode="lines",
                name=f"{e2_train_run.label} (raw)",
                line={"color": OKABE_ITO["blue"], "width": 1},
                opacity=0.55,
            )
        )
        texture_fig.update_layout(
            title="E2: per-step train loss — spike/variance texture",
            xaxis_title="step",
            yaxis_title="loss",
        )
        texture_output = mo.ui.plotly(texture_fig)

    texture_output
    return


@app.cell
def _(OKABE_ITO, e2_run, e2_style, e3_run, e3_style, go, mo):
    if e2_run is None or e3_run is None:
        gap_output = mo.md("*(need both E2 and E3 loaded for the loss-separation chart)*")
    else:
        gap_fig = go.Figure()
        gap_fig.add_trace(
            go.Scatter(
                x=[r.step for r in e2_run.records],
                y=[r.nats_per_byte for r in e2_run.records],
                mode="lines+markers",
                name=e2_style().label,
                line={"color": OKABE_ITO["blue"], "dash": "solid"},
            )
        )
        gap_fig.add_trace(
            go.Scatter(
                x=[r.step for r in e3_run.records],
                y=[r.nats_per_byte for r in e3_run.records],
                mode="lines+markers",
                name=e3_style().label,
                line={"color": OKABE_ITO["orange"], "dash": "dash"},
            )
        )
        common_steps = sorted({r.step for r in e2_run.records} & {r.step for r in e3_run.records})
        by_e2 = {r.step: r.nats_per_byte for r in e2_run.records}
        by_e3 = {r.step: r.nats_per_byte for r in e3_run.records}
        gap_fig.add_trace(
            go.Scatter(
                x=common_steps,
                y=[by_e2[s] - by_e3[s] for s in common_steps],
                mode="lines+markers",
                name="e2 − e3 gap",
                yaxis="y2",
                line={"color": OKABE_ITO["bluish_green"], "dash": "dot"},
            )
        )
        gap_fig.add_hline(y=0, line_color=OKABE_ITO["black"], line_dash="dot", yref="y2")
        gap_fig.update_layout(
            title="E3 vs E2: loss separation",
            xaxis_title="step",
            yaxis_title="nats/byte",
            yaxis2={"title": "Δ nats/byte", "overlaying": "y", "side": "right"},
        )
        gap_output = mo.ui.plotly(gap_fig)

    gap_output
    return


@app.cell
def _(OKABE_ITO, e3_run, go, head_style, mo):
    from entropic_substrate.geometry.kappa import KAPPA_MAX, KAPPA_MIN

    # matplotlib -> plotly codes for head_style()'s linestyle/marker channels.
    # Deliberately plain-``[]`` lookups (no ``.get(x, default)``): an
    # unmapped code must KeyError loud, not silently fall back to a shared
    # default that would recreate the exact color-only collision this
    # translation exists to avoid (head_style() is what keeps head 0 from
    # rendering identically to the black zero-line/clamp-bound references).
    _DASH_FOR_LINESTYLE = {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}
    _SYMBOL_FOR_MARKER = {
        "o": "circle",
        "s": "square",
        "^": "triangle-up",
        "D": "diamond",
        "v": "triangle-down",
        "P": "cross",
        "X": "x",
        "*": "star",
    }

    kappa_records = [r for r in e3_run.records if r.kappa is not None] if e3_run else []
    if not kappa_records:
        kappa_output = mo.md("*(no kappa telemetry in the loaded E3 run)*")
    else:
        # e3_run reaches this cell only through load_eval_run -> read_eval_series,
        # which is the ONLY path a kappa-bearing record arrives by. That reader
        # now refuses a cross-record kappa shape mismatch as an admissibility
        # rule (entropic_substrate.telemetry), so every record here is already
        # guaranteed to share n_layers/n_heads with kappa_records[0] — deriving
        # them from the first record is safe, and a second local shape check
        # here would re-create the duplicate-validator defect telemetry.py
        # exists to remove.
        n_layers = len(kappa_records[0].kappa)
        steps = [r.step for r in kappa_records]
        kappa_figs = []
        for layer_idx in range(n_layers):
            n_heads = len(kappa_records[0].kappa[layer_idx])
            layer_fig = go.Figure()
            for head_idx in range(n_heads):
                style = head_style(head_idx)
                layer_fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=[r.kappa[layer_idx][head_idx] for r in kappa_records],
                        mode="lines+markers",
                        name=f"head {head_idx}",
                        line={
                            "color": style.color,
                            "dash": _DASH_FOR_LINESTYLE[style.linestyle],
                        },
                        marker={"symbol": _SYMBOL_FOR_MARKER[style.marker]},
                    )
                )
            layer_fig.add_hline(y=0.0, line_color=OKABE_ITO["black"])
            layer_fig.add_hline(y=KAPPA_MIN, line_color=OKABE_ITO["black"], line_dash="dot")
            layer_fig.add_hline(y=KAPPA_MAX, line_color=OKABE_ITO["black"], line_dash="dot")
            layer_fig.update_layout(
                title=f"layer {layer_idx}: per-head κ trajectory (clamp [{KAPPA_MIN}, {KAPPA_MAX}])",
                xaxis_title="step",
                yaxis_title="κ",
            )
            kappa_figs.append(mo.ui.plotly(layer_fig))
        kappa_output = mo.vstack(kappa_figs)

    kappa_output
    return


@app.cell
def _(OKABE_ITO, e2_run, e3_run, go, max_kappa_spread, mo):
    kappa_records2 = [r for r in e3_run.records if r.kappa is not None] if e3_run else []
    if not kappa_records2 or e2_run is None:
        timing_output = mo.md(
            "*(need E3 kappa telemetry AND E2 loaded for the divergence-vs-gap chart)*"
        )
    else:
        kappa_by_step2 = {r.step: r.kappa for r in kappa_records2}
        by_e2b = {r.step: r.nats_per_byte for r in e2_run.records}
        by_e3b = {r.step: r.nats_per_byte for r in e3_run.records}
        common2 = sorted(set(kappa_by_step2) & set(by_e2b) & set(by_e3b))
        if not common2:
            timing_output = mo.md("*(kappa steps share no common eval step with e2/e3)*")
        else:
            spreads = [max_kappa_spread(kappa_by_step2[s]) for s in common2]
            gaps = [by_e2b[s] - by_e3b[s] for s in common2]
            timing_fig = go.Figure()
            timing_fig.add_trace(
                go.Scatter(
                    x=common2,
                    y=spreads,
                    mode="lines+markers",
                    name="max within-layer head κ spread (E3)",
                    line={"color": OKABE_ITO["orange"], "dash": "solid"},
                )
            )
            timing_fig.add_trace(
                go.Scatter(
                    x=common2,
                    y=gaps,
                    mode="lines+markers",
                    name="e2 − e3 loss gap",
                    yaxis="y2",
                    line={"color": OKABE_ITO["blue"], "dash": "dash"},
                )
            )
            timing_fig.update_layout(
                title="κ-divergence timing vs loss-gap timing",
                xaxis_title="step",
                yaxis_title="κ spread",
                yaxis2={"title": "Δ nats/byte", "overlaying": "y", "side": "right"},
            )
            timing_output = mo.ui.plotly(timing_fig)

    timing_output
    return


if __name__ == "__main__":
    app.run()
