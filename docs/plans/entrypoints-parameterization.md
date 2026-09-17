# Deliverable 10 (part 2): parameterizing the Python-side dotted allowlists

DESCRIBE ONLY. Nothing under `src/` is patched here: `provider_launch.py`,
`d1_runpod_adapter.py`, `h_probe_launch.py`, and
`analysis/d1_external_analysis_adapter.py` are owned by structural phases
(`p5-kernel`, `p9-run-lower`/`p10-run-upper`, `p4-h-tracer`, and
`p11-analyze-review-file` respectively, per `docs/plans/src-hexagonal-refactor.plan.md`);
this file is the design note those phases implement against, per the plan's
deliverable-10 line: "the dotted module names become ONE parameterized source
... consumed by ... the justfile, run_ladder.sh, and the worker allowlists".

## Sites enumerated (grep for `-m entropic_substrate` and the dotted-string
## allowlists, 2026-09-17, HEAD bbe7976)

| file | line(s) | dotted name | role |
|---|---|---|---|
| `src/entropic_substrate/provider_launch.py` | 33-34 | `entropic_substrate.d1_trainer_mailbox_command`, `entropic_substrate.d_ladder_worker_command` | `PodLaunchRequest.__post_init__`'s allowlist set — the only two worker modules any provider adapter is permitted to launch |
| `src/entropic_substrate/d1_runpod_adapter.py` | 345 | `entropic_substrate.d1_trainer_mailbox_command` | literal tuple built in `RunPodAdapter.create()`, passed into `PodLaunchRequest` (checked against the same allowlist above at construction) |
| `src/entropic_substrate/h_probe_launch.py` | 164 | `entropic_substrate.sentence_prediction.worker_command` | literal tuple in `create_arguments`, the H0 remote-worker launch command |
| `src/entropic_substrate/analysis/d1_external_analysis_adapter.py` | 224 | `entropic_substrate.analysis.d1_external_worker` | literal tuple in `_argv_v2`, the sandboxed D1 external-analysis worker command |
| `src/entropic_substrate/telemetry.py` | 477 | `entropic_substrate.telemetry` | `argparse.ArgumentParser(prog=...)` display string only — cosmetic (`--help` text), not an invocation site; update it for consistency when telemetry's phase lands, but it is not part of the frozen-literal contract test |
| `tests/test_d1_runpod_adapter.py` | 214 | `entropic_substrate.d1_trainer_mailbox_command` | test fixture asserting the exact `--docker-args` string RunPodAdapter emits; must track provider_launch.py's constant or it silently stops testing the real value |
| `Containerfile` | 90 | `entropic_substrate.cli` | a comment showing an example manual override of the default `CMD` (`sleep infinity`); never executed by the build, but the literal will read stale once `entropic_substrate.cli` moves — p12 (which owns `justfile`/`run_ladder.sh`/`Containerfile` per the phase table) should update it in the same pass |

Excluded deliberately: `analysis/d1_reader_source_closure.py:24` lists
`"entropic_substrate.telemetry"` as one of 52 entries in the D1 reader-source
closure manifest (a set of modules whose *source text* is hashed for a
provenance digest, not an executable allowlist). Ruling E-1 retires that whole
closure in phase `p2-d1-closure`; it is out of scope for this parameterization
and must not be folded into `entrypoints.env`.

No `[project.scripts]` console-script entry point exists in `pyproject.toml` —
every one of these is invoked as `python -m <dotted>`, so there is no fifth
site hiding in packaging metadata.

## The source of truth

`scripts/entrypoints.env` (deliverable 10, part 1) is the single source, in
the constrained form the plan requires — because it must be simultaneously:

* **bash-sourceable** by `scripts/run_ladder.sh` (`source scripts/entrypoints.env`
  after its `cd "$(dirname "$0")/.."`, under `set -euo pipefail` so a missing
  key raises `unbound variable` at first use, not silently);
* **`just dotenv`-loadable** (`set dotenv-load` + `set dotenv-path := "scripts/entrypoints.env"`
  in the justfile, consumed via `env_var("ENTRYPOINT_CLI")` etc. — verified
  2026-09-17: a missing key fails `just` at PARSE time, before any recipe
  runs, because `cli := env_var(...)` is evaluated eagerly for the whole file);
* **parseable by a stdlib-only Python reader** — plain `KEY=dotted.module`
  lines, no quotes, no spaces, no `export` keyword. `python-dotenv` is
  deliberately NOT proposed as a dependency: this file's grammar is a strict
  subset of `.env` (one key, one dotted Python module path, per line) and a
  five-line `str.split("=", 1)` loop over `splitlines()` covers it completely
  with zero new supply-chain surface.

Keys (matching the sites table above):

```text
ENTRYPOINT_CLI=entropic_substrate.cli
ENTRYPOINT_TELEMETRY=entropic_substrate.telemetry
ENTRYPOINT_H_WORKER=entropic_substrate.sentence_prediction.worker_command
ENTRYPOINT_D1_TRAINER_MAILBOX=entropic_substrate.d1_trainer_mailbox_command
ENTRYPOINT_D_LADDER_WORKER=entropic_substrate.d_ladder_worker_command
ENTRYPOINT_D1_EXTERNAL_WORKER=entropic_substrate.analysis.d1_external_worker
```

## How the four Python files should read it

Proposed shape (for the owning phase to implement, not this deliverable):
a tiny `entropic_substrate.entrypoints` module, e.g.
`src/entropic_substrate/entrypoints.py` (or, once `p1-anchors` lands the
layout-invariant `paths.py` resolver, alongside it under `kernel/runtime/`),
that:

1. Locates `scripts/entrypoints.env` via the **p1 `paths.py` anchor**
   (the layout-invariant depth-derived resolver p1 introduces at five sites),
   not a hand-rolled `Path(__file__).parents[N]` — the anchor is precisely
   the mechanism p1 exists to centralize, and a runtime file read from
   `scripts/` is otherwise fragile across install/container contexts (see
   point 3 below).
2. Parses it with the same five-line stdlib reader described above (shared,
   not duplicated, with any test that also needs the mapping — one parser,
   one place it can be wrong).
3. Exposes one constant per key, e.g. `ENTRYPOINT_CLI: str`, `ENTRYPOINT_D1_TRAINER_MAILBOX: str`,
   etc., so `provider_launch.py`'s allowlist becomes
   `{ENTRYPOINT_D1_TRAINER_MAILBOX, ENTRYPOINT_D_LADDER_WORKER}` and
   `h_probe_launch.py` / `d1_external_analysis_adapter.py`'s literal tuples
   splice in `ENTRYPOINT_H_WORKER` / `ENTRYPOINT_D1_EXTERNAL_WORKER` instead
   of a hand-typed dotted string.
4. FAILS LOUD (raises, never returns a default) if `scripts/entrypoints.env`
   is absent or a required key is missing — consistent with this repo's
   measurement-code posture, and because a silently-empty entrypoint name
   would pass `PodLaunchRequest`'s own type checks (`type(item) is not str`)
   right up until the RunPod API call rejects an empty argv token.

**Confirmed no pod-side coupling**: `grep -rln "provider_launch" src/ tests/`
(2026-09-17) shows only `d1_runpod_adapter.py` (operator-side adapter) and two
test files import it; nothing that ships INSIDE a container image imports
`provider_launch`. That is why reading `scripts/entrypoints.env` at runtime is
viable at all for these four files — they run on the operator's machine,
where the repo checkout (and therefore `scripts/`) is present. If a future
phase moves any of these four into a module that CAN run pod-side (inside the
image `Containerfile` builds, where `scripts/` is not guaranteed to be
copied — check `Containerfile`'s `COPY` lines before assuming it is), the
env-file read is not portable and the fallback is: the
`entropic_substrate.entrypoints` module holds the six values as plain
constants (a small literal dict, not a file read), with a unit test asserting
those constants exactly equal what `scripts/entrypoints.env` parses to
(`test_entrypoints_module_matches_env_file` or equivalent) — so the two
representations cannot silently drift, without requiring `scripts/` inside
the image. Whichever of the two the owning phase picks, it must be a
symmetric two-sided test (module -> env file AND env file -> module), the
same "GREEN control catches the RED check going vacuous" pattern used in
`tools/test_gen_shims.py`.

## Interaction with the p0 frozen-literal pin

`docs/plans/src-hexagonal-refactor.plan.md`'s Phase 0 contents pin the exact
current literals at `provider_launch.py:33-34`, `d1_runpod_adapter.py:345`,
`h_probe_launch.py:164`, `justfile:42`, and
`d1_external_analysis_adapter.py:224` as a `--help`-snapshot-style contract
fixture. Once `entrypoints.env` exists and these files read through it, that
fixture's assertions should move from "the literal string equals X" to "the
literal string equals `entropic_substrate.entrypoints.ENTRYPOINT_<NAME>`,
which equals X" — the frozen VALUE does not change, but the frozen SOURCE
does, and the phase that performs this swap should say so explicitly in its
PR body per house rule 4 (comments/fixtures record measurements; a value that
becomes derived rather than literal is a fact worth stating, not silently
overwriting the old comment).
