# for-1608-cli-handler-families: bring composition/entrypoints/cli.py under 1000 lines

DRAFT 2026-10-21 by the conductor (wave B `wave-b-2026-10-21`; issue FOR-1608, thermo-nuclear review file-size
finding at epoch `0e55d9d`, re-measured). Anchors measured at the wave base `434b63d2` with `wc -l`/`grep -n`;
the implementer re-verifies them in recon. The Linear ticket is authoritative; its "related dispatch-table issue"
FOR-1571 is Done, so the remaining lever is the one named in its remedy: move handler families into sibling modules.

## Branch

| field                | value                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1608-cli-handler-families`                                                                                 |
| parent               | the wave base branch `fix/wave-b-2026-10-21` (`434b63d2` + the briefs commit)                                       |
| cascade `phase.mode` | `normal` (a pure move of handler functions; the parser, the subcommand set and the handlers table keep their shape) |
| publish              | none in the lane (`publishMode: defer`, CERTIFIED); the zip publishes                                               |

## Why (measured at `434b63d2`)

- `src/entropic_substrate/composition/entrypoints/cli.py` is 1433 lines (1161 at v0.3.2; the H-series stack beneath this base added `_add_h_analysis_lanes` `:118-207` and the H-analysis handler family `_emit_h_document` `:1199`, `_h_host_conflict` `:1205`, `_h_refuse_host_args` `:1217`, `_h_aggregate_document` `:1228`, `_cmd_aggregate_h` `:1240`, `_cmd_charts_h` `:1257`, `_h_receipt_verdict` `:1309`, `_cmd_eval_h` `:1327`). Other families: E-series training/eval (`cmd_train` `:537`, `cmd_init_fisher` `:576`, `cmd_eval` `:601`, the two preflights `:457`, `:498`), capture/aggregate/charts (`cmd_capture_results` `:650`, `cmd_mlflow_aggregate` `:728`, `cmd_charts` `:793`), embedding geometry (`_load_ckpts_by_seed` `:937` through `cmd_embedding_geometry` `:1025`), `build_parser` `:208-437`, `main` `:1357` with the D1 early return (`:1365-1374`) and the `handlers` table (`:1396-1418`, 21 string keys).
- Pins in `tests/test_cli_surface.py`: `_FROZEN_SUBCOMMANDS` (`:35`) equals the parser's discovered subcommands (`:136`); prog name (`:148`); top-level help exits 0 and is stable (`:155`, `:160`); frozen dotted literals by module site (`:81`, `:176`); ONLY the D1 early return branches on `args.command` (`:299`); every non-D1 subcommand resolves through the `handlers` table (`:310`, AST: keys are unique string literals equal to `_FROZEN_SUBCOMMANDS - _D1_COMMANDS`); handler values are statically resolved NAMES (`:326`, `ast.Name`: an imported function name satisfies it, an attribute access `module.func` does NOT); function-local handler routing (`:358`).
- 30 importers of `entropic_substrate.composition.entrypoints.cli` across src/tests/scripts (grep) import `build_parser`, `main` and specific `cmd_*` names; the move must keep every imported name importable from `cli.py` (re-export) or update the importers (prefer re-export; measure which names the tests import before choosing).
- `composition.entrypoints.cli` is NOT a reader-tuple member (`ladder/run/checkpoint.py:66-116` lists `composition.cli`, a different module, plus `capture_cli`/`source_cli`/`span_cli`).

## Scope

1. **C1, tests RED at the parent**: a size pin in `tests/test_cli_surface.py` that `cli.py` is under 1000 lines (RED at 1433); a re-export pin that every name the test-suite imports from `cli` is still reachable (GREEN at both, guard for M2); the existing AST pins stay green by construction.
2. **C2, the move**: at least the H-analysis family (`_add_h_analysis_lanes` + the eight `_h_*`/`_cmd_*_h` functions) and the embedding-geometry family into sibling modules under `composition/entrypoints/` (e.g. `h_analysis_cli.py`, `embedding_geometry_cli.py`), imported into `cli.py` by NAME so the `handlers` table values stay `ast.Name`; `build_parser` may delegate lane construction to the sibling (as `_add_h_analysis_lanes` already does) but the subcommand set is unchanged; `main` and the D1 early return unchanged. Pick the smallest set of moves that gets under 1000; record `wc -l` of `cli.py` and every sibling in the commit message.
3. **C3**: `--help` snapshot stability pins (`:155`, `:160`) and the 30 importers green; no behaviour change (handlers are moved, not edited).

## Guard teeth by mutation

| id | mutation                                                 | expected RED                                                   |
| -- | -------------------------------------------------------- | -------------------------------------------------------------- |
| M1 | inline one moved family back                             | the <1000 size pin RED                                         |
| M2 | drop one re-exported name from `cli.py`                  | the re-export pin RED (and the importing test RED)             |
| M3 | reference a moved handler as `module.func` in `handlers` | `test_handlers_table_values_are_statically_resolved_names` RED |
| M4 | remove one subcommand's parser lane                      | `test_build_parser_subcommand_set_matches_frozen_tuple` RED    |

## Allowed files

`src/entropic_substrate/composition/entrypoints/cli.py`, new sibling modules under `src/entropic_substrate/composition/entrypoints/` created by this lane, `tests/test_cli_surface.py`; importers ONLY if a re-export is impossible (report why). Never: `scripts/entrypoints.env` (it names `composition.entrypoints.cli` and must keep working), `composition/wiring/*`, `composition/cli.py` / `capture_cli.py` / `source_cli.py` / `span_cli.py` (reader-tuple members), `docs/*` (post-zip sweep), `pyproject.toml`.

## Acceptance

C1 RED at the parent, GREEN at the tip; M1-M4 RED one at a time and restored; all `tests/test_cli_surface.py` pins and the 30 importing test modules green unchanged in meaning; `python -m entropic_substrate --help` exits 0; `just check` exit 0 and the exit-gate runner green under the suite lock; line counts recorded in the commit message; diff confined to the Allowed files.
