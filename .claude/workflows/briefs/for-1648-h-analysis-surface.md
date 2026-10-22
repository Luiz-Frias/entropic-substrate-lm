# FOR-1648 [parity B9] H analysis surface: seed aggregation, charts and a single-run report over captured H runs

DRAFT. Drafted read-only at 1d3d67d1 (main 0f4d7215 v0.3.3 plus one docs commit adding
`results/h0-plumbing-1k-v3-HEpP7g.dvc` and `.claude/workflows/briefs/h0-analysis-hepp7g.md`). Every anchor below
was re-measured at that commit; where it differs from the ticket text it says so. Three premises of the ticket
are stale or unmeasurable from the repo alone and are marked DIFFERS in "Why". The conductor must read the
DIFFERS rows before launching.

## Branch

| Field                | Value                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Branch               | `feat/for-1648-h-analysis-surface`                                                                                       |
| Parent               | the wave base branch (conductor fills at launch; the lane brief does not pin a SHA)                                      |
| Cascade `phase.mode` | `arch` (operator ruling 2026-10-20)                                                                                      |
| Publish              | none in the lane: `publishMode` = `defer`, the lane returns CERTIFIED, the wave publishes once at Zip via `just ship-it` |
| Reader-tuple         | none of the Allowed files is a member of `_READER_MODULES` (verified below)                                              |

## Why (measured at 1d3d67d1)

Ticket text, verbatim: "[parity B9] H analysis surface: seed aggregation, charts and a single-run report over
captured H runs. Measured at edabe74: E has mlflow_aggregate grouping leaf runs by the config_hash tag
(ladder/capture/reader/mlflow_aggregate.py:66-74), charts `fig_e1_*` through `fig_e15d_*`
(ladder/analyze/charts.py:156-468) with readers read_eval_series and read_train_series (telemetry.py:341, :427),
chart attachment to runs (ladder/analyze/mlflow_attach.py:28, :48 via cmd_charts at cli.py:678), and the eval
--json report sink (cli.py:541). For H: grep for h-training-step, h-predictive-evaluation, h-probe, h_probe and
HProfile over ladder/analyze, ladder/capture, ladder/gate, ladder/file and ladder/review returns zero files;
h-inspect-training (composition/cli.py:29, :161) verifies a receipt and prints JSON and can serve as the reader.
Tests with no H counterpart: test_mlflow_aggregate.py (22), test_cli_charts.py (13),
test_analysis_mlflow_attach.py (7). Ask: after FOR-1639 (capturable) and FOR-1642 (config_hash tag) land and the
probe re-run yields two captured H runs as real fixtures: an H aggregate over paired arms (objective,
future_nll_per_dimension, `eval/*` per global step), a small chart set (training curve per carrier, evaluation
rows, publication timing per checkpoint from FOR-1643's tags), attachment of charts to the runs, and a
single-run report that reuses h-inspect-training. RED-first against the captured fixtures under results/*.dvc;
mutation = drop the carrier grouping goes RED. Fail LOUD on inadmissible H telemetry, never fail-open
(measurement path). Files: new modules under ladder/analyze, composition/entrypoints/cli.py, tests. No
checkpoint-reader-tuple member."

### E-side anchors (all under `src/entropic_substrate/`)

| Anchor             | Measured at 1d3d67d1                                                                                                                                                                                                                                                                                                                                       | Vs ticket                                                                               |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| leaf selection     | `ladder/capture/reader/mlflow_aggregate.py:66` `def _leaf_runs(client: Any, experiment_id: str) -> list[Any]:` body 67-73 filters `run.data.tags.get(CONFIG_HASH_TAG) and run.data.tags.get(PARENT_TAG) != "1"`                                                                                                                                            | matches (66-74)                                                                         |
| grouping           | `mlflow_aggregate.py:76` `def _group_by_config_hash(runs: Sequence[Any]) -> dict[str, list[Any]]:` and `:80` `groups.setdefault(run.data.tags[CONFIG_HASH_TAG], []).append(run)`                                                                                                                                                                           | matches                                                                                 |
| aggregate entry    | `mlflow_aggregate.py:125` `def aggregate(`, `:138` `groups = _group_by_config_hash(_leaf_runs(client, experiment.experiment_id))`                                                                                                                                                                                                                          | n/a                                                                                     |
| charts             | `ladder/analyze/charts.py:156` `def fig_e1_paired_loss_curves(` through `:468` `def fig_e15d_gram_retention(points: Sequence[tuple[str, GramRetention]]) -> Figure:`; `:73` `def _figure(figsize: tuple[float, float]) -> Figure:`, `:79` `def _series_xy(run: EvalSeriesRun \| TrainSeriesRun, metric: str)`, `:85` `def _plot_seed_band(`                | matches                                                                                 |
| E readers          | `ladder/capture/reader/telemetry.py:183` `def final_eval_record(path: str \| Path) -> dict[str, Any]:`, `:341` `def read_eval_series(path: str \| Path) -> list[EvalRecord]:`, `:427` `def read_train_series(path: str \| Path) -> list[TrainRecord]:`                                                                                                     | matches                                                                                 |
| attach             | `ladder/analyze/mlflow_attach.py:28` `def attach_figures_to_run(client: Any, run_id: str, figures: dict[str, Figure]) -> list[str]:` (best-effort, `:44` `except Exception as exc:  # best-effort by design`), `:48` `def attach_seed_band_to_aggregate(`                                                                                                  | matches; NOTE the E attach is deliberately fail-soft, the H attach must not be (see C2) |
| charts command     | `composition/entrypoints/cli.py:689` `def cmd_charts(args: argparse.Namespace) -> int:`; parser `:225` `"charts",`; dispatch `:1130` `"charts": cmd_charts,` inside `handlers` `:1126-:1136`                                                                                                                                                               | DIFFERS: ticket says cli.py:678                                                         |
| eval command       | `cli.py:506` `def cmd_eval(args: argparse.Namespace) -> int:`; the `--json` sink is the `if args.json:` branch at `:536` printing `json.dumps(payload, sort_keys=True)` at `:540`; parser flag `:177` `evaluate.add_argument("--json", action="store_true", help="print JSON only")`; dispatch `:1127` `"eval": cmd_eval,`                                 | DIFFERS: ticket says cli.py:541                                                         |
| aggregate command  | `cli.py:626` `def cmd_mlflow_aggregate(args: argparse.Namespace) -> int:`, dispatch `:1129`                                                                                                                                                                                                                                                                | n/a                                                                                     |
| H dispatch         | `cli.py:1136` `"h-inspect-training": cmd_sentence,` beside `"h-train-probe": cmd_sentence,` `:1135`                                                                                                                                                                                                                                                        | n/a                                                                                     |
| h-inspect-training | `composition/cli.py:38` `("h-inspect-training", "verify an H training receipt without running training"),`; `:53` `def cmd_sentence(args: argparse.Namespace) -> int:`; `:75` `"h-inspect-training": "h-probe-inspect-command-v1",` in the `expected` map; parser args `--root` (Path, required) and `--manifest` (required) at `composition/cli.py:43-44` | DIFFERS: ticket says composition/cli.py:29, :161                                        |

### H-side anchors

| Anchor                                                                                                                                                                                 | Measured at 1d3d67d1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H marker grep (`h-training-step`, `h-predictive-evaluation`, `h-probe`, `h_probe`, `HProfile`, `h-run-lifecycle`) over `ladder/analyze`, `ladder/gate`, `ladder/file`, `ladder/review` | 0 files in each of the four                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| same grep over `ladder/capture`                                                                                                                                                        | DIFFERS from the ticket (which says zero): 3 files, `ladder/capture/reader/sinks.py` (`:80`, `:226`), `ladder/capture/reader/capture_results.py` (`:15`, `:53`), `ladder/capture/reader/h_telemetry.py` (`:5`, `:6`, `:13`, `:43`, `:44`, `:101`). Total across the five directories: 3 files                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| the H reader that now exists                                                                                                                                                           | `ladder/capture/reader/h_telemetry.py:43` `_LIFECYCLE_SCHEMA = "h-run-lifecycle-v1"`, `:44` `_STEP_SCHEMA = "h-training-step-v2"`, `:45` `_COMMITTED = "committed"`, `:41` `__all__ = ["final_h_run_end", "require_json_object_lines"]`, `:96` `def final_h_run_end(path: str \| Path) -> dict[str, Any]:` raising `TelemetryGap` unless exactly one `run_start`, exactly one `run_end` and at least one committed step row (checks at `:105-:128`). The `h-predictive-evaluation-v1` schema name appears only in this module's docstring (`:5`), not as a literal                                                                                                                                                                                      |
| H grouping key                                                                                                                                                                         | `ladder/capture/reader/sinks.py:226` `def h_config_hash(profile: HProfile) -> str:` = sha256 of the profile manifest minus the top-level `seed`, first 12 hex (FOR-1642 landed). The carrier is part of the manifest, so the two carriers of one wave will NOT share an `h_config_hash`: the lane must measure what the `run_start`/`run_end` rows carry (carrier, config hash, source identity) before choosing the carrier grouping key                                                                                                                                                                                                                                                                                                               |
| H lane capture                                                                                                                                                                         | `capture_results.py:15` "A run whose tag starts with `h-` ... is the H lane"; `:53` "An H-tagged run with no `h-run-lifecycle-v1` records is refused" (FOR-1639 landed)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| H metric keys, step scalars                                                                                                                                                            | `ladder/run/h_metric_extractor.py:39-45` `_SCALAR_FIELDS = ("objective", "aggregate_norm", "decoder_ce", "future_nll_per_dimension", "wall_seconds")`; `:48` `_PHASE_INDEX = {"codec_warmup": 0.0, "prediction": 1.0}`; `:125` `metrics: dict[str, float] = {"phase_index": _PHASE_INDEX[cast(str, phase)]}`; global step = prediction step plus `warmup_steps` (`:130` `offset = 0 if phase == "codec_warmup" else self._warmup_steps`)                                                                                                                                                                                                                                                                                                                |
| H metric keys, evaluation                                                                                                                                                              | `h_metric_extractor.py:187` `metrics[f"eval/log_marginal_k{key}"] = ...`, `:189` `metrics[f"eval/ess_k{key}"] = ...`, `:190` `metrics[f"eval/gap_rows_k{key}"] = float(gaps[key])`, `:191-:194` `for field in ("wall_seconds", "cuda_seconds"): ... metrics[f"eval/{field}"] = value`                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| H metric keys, stage/admission                                                                                                                                                         | `h_metric_extractor.py:145-:147` `f"{prefix}/seals_walked"`, `f"{prefix}/evaluations"`, `f"{prefix}/journals_read"` (FOR-1669 `h-worker-stage-v1`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| publication timing (FOR-1643)                                                                                                                                                          | NOT found as MLflow tags or extractor keys in this tree. What exists: `ladder/run/runner.py:245` `publication_timing: MutableMapping[str, float] \| None = None`, `:374` `publication_timing["l1_seconds"] = ...`, `:487` `publication_timing["l2_seconds"] = ...`; `ladder/run/object_worker.py:136` `def _publication(` builds the checkpoint event's `publication` block with `:155` `"l1_seconds": timing.get("l1_seconds", 0.0),`; emitted at `object_worker.py:363` `"publication": publication,`. DIFFERS from the ticket ("from FOR-1643's tags"): the lane must grep the captured `worker-log.jsonl` and `metrics.jsonl` for the checkpoint events and use whatever is actually there, and file a one-line gap note if the fixtures carry none |
| tests, current counts (`grep -c 'def test_'`)                                                                                                                                          | `tests/test_mlflow_aggregate.py` 24 (DIFFERS: ticket says 22), `tests/test_cli_charts.py` 13, `tests/test_analysis_mlflow_attach.py` 7                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

### Fixtures

`results/h0-plumbing-1k-v3-HEpP7g.dvc` (tracked), quoted:

```yaml
outs:
- md5: 2dc65e2bdf918d43858c6b2d92a63c58.dir
  size: 5597931
  nfiles: 4
  hash: md5
  path: h0-plumbing-1k-v3-HEpP7g
```

Four files `h-worker-<request_sha256>-<runid>-metrics.jsonl` and `-worker-log.jsonl` for runs `a163161f`
(magnitude, pod 211tvsc7277d22) and `988d8388` (sphere, pod euikqmf95pnrga); per arm the metrics file holds 1009
rows, one `run_start`, one `run_end` (per `.claude/workflows/briefs/h0-analysis-hepp7g.md`). They are NOT present
in any checkout at drafting time (the DVC cache has no `2d/` entry; `results/h0-plumbing-1k-v3-HEpP7g/` does not
exist). The lane obtains them with `dvc pull results/h0-plumbing-1k-v3-HEpP7g.dvc`. R2 credentials come from
`.env`: a lane worktree needs the `.env` symlink from the main checkout first (memory: worktree live-gate
prerequisites). `.gitignore:51` `results/*`, `:52` `!results/.gitkeep`, `:53` `!results/*.dvc`: the JSONL files
are ignored by construction and must never be committed. Not measured here, the lane measures first: the actual
row shapes (the `JsonlSink.log` record is `json.dumps(dict(record), sort_keys=True)`, `ladder/capture/adapters/jsonl.py:84`),
the fields `run_start`/`run_end` carry, and whether checkpoint `publication` blocks are in either file.

### Reader-tuple (the identity this lane must not move)

`src/entropic_substrate/ladder/run/checkpoint.py:65` `_READER_MODULES = (` lists, among others,
`"composition.capture_cli"` (`:66`), `"composition.cli"` (`:67`), `"ladder.capture.capture"` (`:94`) and
`"ladder.run.*"` modules (`:95-:112`). Consequences:

| Module                                                                   | Member?                                                           | Consequence                                                                               |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `composition/cli.py`                                                     | YES                                                               | `h-inspect-training`'s module is NEVER edited; the report calls it, it does not change it |
| `composition/capture_cli.py`                                             | YES                                                               | not touched                                                                               |
| `ladder/capture/capture.py`                                              | YES                                                               | not touched                                                                               |
| `ladder/capture/reader/h_telemetry.py`, `sinks.py`, `capture_results.py` | no (not in the tuple)                                             | still outside Allowed files: they are reused as is, not edited                            |
| `ladder/analyze/*`                                                       | no (verified: no `ladder.analyze` entry in `:65-:112`)            | new modules here are safe                                                                 |
| `composition/entrypoints/cli.py`                                         | no (verified: `composition.entrypoints` is absent from the tuple) | safe to add subcommands                                                                   |

Editing any member re-pins the regime identity and goes RED at commit (memory: reader-member commits need a
conductor hand-off); this lane has no such hand-off and must not need one.

### The three DIFFERS the conductor should rule on before launch

1. `h-inspect-training` takes `--root` and `--manifest` of an H command manifest and verifies a RECEIPT
   (`composition/cli.py:43-44`, `:53`). The four captured files contain no receipt. So "a single-run report that
   reuses h-inspect-training" cannot be fed by the HEpP7g fixtures alone. Recommendation: the report's core is
   built from the captured `metrics.jsonl` through `final_h_run_end` plus the `worker-log.jsonl`; an optional
   `--receipt-root/--receipt-manifest` pair delegates to `h-inspect-training` (called through the existing
   `cmd_sentence` handler, never reimplemented) and embeds its JSON verdict verbatim. Delegation is test-pinned
   with a patched handler; no receipt fixture is claimed.
2. The ticket's "grouping by config_hash" for paired arms: the carrier is inside the manifest hashed by
   `h_config_hash`, so paired carriers differ in hash. The aggregate must group by the lineage keys the
   captured rows actually carry (measured first), and pair arms by everything except carrier.
3. FOR-1643 publication tags: see the anchor table; the chart is built from what the captured files hold, and
   its absence is reported, not papered over.

## Scope

### C1: tests RED at the parent

New tests only, no implementation, committed first, each RED at the parent for the stated reason (missing
module or missing subcommand, never a collection error from a typo). Fixture-dependent tests read
`results/h0-plumbing-1k-v3-HEpP7g/` and `pytest.skip` is FORBIDDEN on its absence: a missing fixture directory
raises (the measurement path fails loud; a skip is the false-green this repo already paid for). Tiny synthetic
JSONL builders for the negative cases are fine and live in the test files.

- `tests/test_h_aggregate.py`: aggregate over the two captured runs gives one summary per carrier group and a
  paired view keyed by global step for `objective`, `future_nll_per_dimension`, `eval/*`; the global step axis is
  monotonic and reaches 1000; `phase_index` boundary present; an inadmissible `metrics.jsonl` (no `run_end`,
  zero or two `run_start`, no committed step, non-finite scalar, non-monotonic global step) raises
  `TelemetryGap` naming the file, never returns an empty or partial aggregate.
- `tests/test_h_charts.py`: figure builders return a matplotlib `Figure` for training curve per carrier,
  evaluation rows, publication timing per checkpoint; empty input raises, mirroring the E builders' idiom
  (`charts.py:172` style `msg = ...; raise`). Test marker `unit`.
- `tests/test_h_report.py`: single-run report from one captured run: lineage, status, step range, eval steps
  (500/750/1000 per the analysis brief), final-leg stage table; deterministic JSON (`sort_keys=True`); delegation
  to `h-inspect-training` when a receipt pair is passed; report never touches MLflow.
- `tests/test_cli_h_analysis.py`: the three new subcommands parse, dispatch, return exit 0 on the fixtures and
  exit 2 on inadmissible input; attach step against a stub client.

### C2: implementation

New modules under `src/entropic_substrate/ladder/analyze/` (suggested names, the lane may rename within the
directory): `h_aggregate.py`, `h_charts.py`, `h_report.py`, and an `h_attach.py` if the attach logic does not fit
`h_charts.py`.

- Reader: reuse `final_h_run_end` and `require_json_object_lines` from `ladder/capture/reader/h_telemetry.py` for
  admissibility; do not fork a second predicate. AMENDED 2026-10-22 (CodeRabbit thread 4343317420): the
  non-finite and non-monotonic C1 checks have one owner, `ladder/analyze/h_aggregate.py` `load_h_run`
  (the `_finite_number` / `_check_step_scalars` / evaluation checks there), which runs after the two reader
  calls and is the only admission path into the aggregate; it is not a second lifecycle predicate. Series extraction reads the captured `metrics.jsonl` rows
  directly, using the key names in the table above (they are the extractor's projection, so they are the names
  in the captured sink rows; the lane confirms against the real rows).
- Aggregate: group by the measured carrier/lineage key, pair arms, emit per-step paired series; pure function of
  the files, no network.
- Charts: training curve per carrier, evaluation rows per carrier, publication timing per checkpoint (only from
  what the fixtures hold). Okabe-Ito palette and a non-color channel per series (the operator is red-green
  colorblind); reuse `ladder/analyze/palette.py` if it already provides this, do not invent a palette.
- Attachment: `client.log_figure(run_id, figure, f"charts/{name}.png")` per run like `mlflow_attach.py:28`, but
  H attach FAILS LOUD: a run that cannot be attached raises after all runs are attempted, with every failed
  run named. It must not copy the E function's `except Exception: ... continue` at `mlflow_attach.py:44`.
  Local rendering never depends on MLflow (same sacred invariant as `cmd_charts`).
- CLI: add `h-aggregate`, `h-charts`, `h-report` parsers and handlers in `composition/entrypoints/cli.py` next to
  `"charts"` (`:225`, `:1130`); handlers lazy-import matplotlib/mlflow like `cmd_charts` (`:689`). New
  subcommands only; no existing parser, handler or exit code changes.
- Failure semantics: inadmissible H telemetry raises `TelemetryGap`; the CLI maps it to exit 2 with the file
  named. No fail-open wrapper anywhere on this path.

## Guard teeth by mutation

Each mutation is applied alone, the named test must go RED for the stated reason, then the mutation is
reverted and the suite is green again. The lane report quotes the RED line for each.

| #  | Mutation                                                                                              | Expected RED                                                                                                                                   |
| -- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| M1 | drop the carrier (config_hash or exp) grouping in the aggregate: collapse both arms into one group    | `tests/test_h_aggregate.py` per-carrier grouping and pairing tests fail                                                                        |
| M2 | make an inadmissible `metrics.jsonl` (missing `run_end`) return an empty aggregate instead of raising | the fail-loud test in `tests/test_h_aggregate.py` fails (no `TelemetryGap`)                                                                    |
| M3 | chart attachment skips a failing run silently (swallow and continue, return success)                  | the attach test in `tests/test_cli_h_analysis.py` or `tests/test_h_charts.py` fails (stub client raising for one run must surface that run id) |
| M4 | the single-run report reads MLflow instead of the captured file                                       | `tests/test_h_report.py` fails with the MLflow client patched to raise                                                                         |

A mutation that stays green means the test is not load-bearing: fix the test, not the mutation.

## Allowed files

| Path                                                                                                            | Mode                       |
| --------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `src/entropic_substrate/ladder/analyze/h_aggregate.py`                                                          | new                        |
| `src/entropic_substrate/ladder/analyze/h_charts.py`                                                             | new                        |
| `src/entropic_substrate/ladder/analyze/h_report.py`                                                             | new                        |
| `src/entropic_substrate/ladder/analyze/h_attach.py`                                                             | new, only if needed        |
| `src/entropic_substrate/composition/entrypoints/cli.py`                                                         | edit, new subcommands only |
| `tests/test_h_aggregate.py`, `tests/test_h_charts.py`, `tests/test_h_report.py`, `tests/test_cli_h_analysis.py` | new                        |

Fixtures arrive only through `dvc pull`; never `git add` anything under `results/` other than an existing `*.dvc`.

Never: `src/entropic_substrate/composition/cli.py`, any `_READER_MODULES` member, `ladder/capture/reader/*`
(reused, not edited), `docs/*`, hooks, `.pre-commit-config.yaml`, `pyproject.toml`, `.gitignore`, any existing
test file.

## Acceptance

1. C1 tests RED at the parent (command and failure reason quoted), GREEN at the lane tip.
2. M1 through M4 each RED one at a time and restored; the lane report quotes each RED line.
3. `just check` exit 0 and the exit-gate runner (`uv run python scripts/run_exit_gate.py`) green under the suite
   lock, each with its exit code and the tip SHA recorded.
4. The aggregate, charts and report run end to end on the two captured HEpP7g runs (`a163161f`, `988d8388`);
   the lane report records each command line and its exit code. Any chart whose data the fixtures do not hold is
   named as absent with the reason, never rendered empty.
5. No Allowed-files violation: `git diff --name-only <parent>..HEAD` quoted; no reader-tuple member and nothing
   under `results/` in it.
6. Evidence classes: every number in the lane report is labelled MEASURED (from a captured file or a command
   output). No estimate.
