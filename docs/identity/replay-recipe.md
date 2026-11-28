# Replay recipe: reading evidence sealed before the platform refactor

Ruling R18 of `docs/plans/series-plugin-topology.plan.md` resets every wire family to a new name
at `-v1` with no read-old adapter. After the first family is renamed, the working tree can no
longer read older evidence, and it is not meant to. Older evidence is read by checking out the
commit that could read it. This file is that procedure, and the list of evidence it must keep
working for.

It exists before horizon M2 renames the first family. Horizon M12 runs it from a fresh clone as
part of the closing check.

## Two epochs, not one

"Replay at the epoch" means two different things for two kinds of evidence. Mixing them up is
the likeliest way to get a false result.

| what is read | where it replays | why |
| --- | --- | --- |
| captured telemetry: `metrics.jsonl`, `worker-log.jsonl`, controller results | the tag `epoch-pre-platform-refactor`, on any host | the admissible readers at that commit accept every retained capture; this recipe proves it below |
| an H checkpoint, and any check that re-executes a sealed chain | the image and source commit named in that capture's own `run_start` row | the checkpoint embeds the reader identity and the loader refuses any other; this was already true before the refactor and the refactor does not change it |

The second row is not a regression of the refactor. The H0 verdict records that its chain check
can only be re-executed inside the image. The recipe for that row is the capture's own verdict
file, which names the image digest and the source commit.

## The epoch tag

| item | value |
| --- | --- |
| tag | `epoch-pre-platform-refactor`, annotated, on `origin` |
| tag object | `0edde818b69fa80d9df1a1007e2a43501e813d38` |
| commit | `5f6b6dd92b9da007624ce255492f55b779fd4bfd`, the last commit of `main` before the refactor trunk |

The tag is never moved and never re-created. If it is ever missing from a clone, fetch it by
name; do not re-tag.

## Procedure, host side

Each step names what a correct result looks like. A step that does not match is a stop, not a
warning.

1. Get the commit. From a fresh clone, or as a detached worktree of an existing one:

   ```bash
   git fetch origin tag epoch-pre-platform-refactor
   git worktree add --detach ../esl-epoch epoch-pre-platform-refactor
   cd ../esl-epoch
   git rev-parse HEAD
   ```

   Correct: `5f6b6dd92b9da007624ce255492f55b779fd4bfd`.

2. Provision the toolchain the commit pins, never the one on the host:

   ```bash
   mise trust && mise install
   uv sync --all-groups --frozen
   ```

   Correct: `uv sync` changes nothing in `uv.lock`. `--frozen` makes a drifted lock an error.

3. Supply the DVC remote credentials. They live in `.env` at the repository root, which is
   gitignored and so absent from a fresh checkout. Copy it from a provisioned checkout. Never
   paste its contents anywhere.

4. Pull the evidence named in the corpus below, one pointer at a time:

   ```bash
   uv run dvc pull results/h0-final-probe-1k-v4-92c9db.dvc
   ```

   Correct: the file count matches the pointer's `nfiles`.

5. Read it with the commit's own readers. For an H capture, one arm at a time:

   ```bash
   D=results/h0-final-probe-1k-v4-92c9db
   STEM=h-worker-5ece5d1b160d593cebfe3a7bbefc0af40f56e59feb6c7db91777c68dd4004880-14637e75
   PYTHONPATH=src uv run python -m entropic_substrate.composition.entrypoints.cli \
     eval h --run "$D/$STEM-metrics.jsonl" "$D/$STEM-worker-log.jsonl" > eval-h.json
   shasum -a 256 eval-h.json
   ```

   Correct: exit 0, one JSON document on stdout, and the digest in "Reference run" below. The
   `h` lane takes no `--json` flag; it refuses one with exit 2. Then the two test selections
   named in "Reference run".

6. Compare against the reference results recorded in this file. Equal means the evidence is
   intact and the readers are the ones that sealed it.

7. Remove the worktree when done: `git worktree remove ../esl-epoch`. Do not run `git gc` or a
   prune as part of cleanup.

## Replay corpus

The plan defines the corpus as one H0 capture and every retained D1 fixture. Horizon M4 takes its
golden vectors from it, M9 proves replay equality over it, and M12 replays it from a fresh clone.

### The H0 capture

| item | value |
| --- | --- |
| pointer | `results/h0-final-probe-1k-v4-92c9db.dvc` |
| content | md5 `ed953d4af265275a9609e7e4a09e2903.dir`, 12 files, 14,096,149 bytes |
| what it is | the MLflow-side capture of the H0 final probe: `metrics.jsonl` and `worker-log.jsonl` for six H runs |
| verdict it supports | `docs/h0/verdict-final-probe-2026-10-26.md`, filed `tracer_complete` |
| arms used as the reference | magnitude, run `14637e75`; sphere, run `38c25d26`; the mapping is the wave's sealed `wave.json` |

Its sibling `results/h0-final-probe-1k-v4.92c9db.dvc`, with a dot where this one has a hyphen, is
a different dataset: the same wave's host-side bundle, 161 files. Neither replaces the other.

### The retained D1 fixtures

No sealed D1 evidence exists (register row D25). What is retained is thirteen fixture modules
under `tests/support`. Each builds its records in code; none is a stored byte artifact, and none
carries a digest literal. Replaying one means running the tests that consume it.

| fixture module | what it builds |
| --- | --- |
| `d1_billing_finalizer_fixtures.py` | real-trainer fixtures for operator finalization |
| `d1_capture_aggregate_fixtures.py` | a mixed complete and terminal-defect capture |
| `d1_capture_fixtures.py` | content-addressed four-arm captures |
| `d1_checkpoint_fixtures.py` | typed values for the checkpoint contract |
| `d1_launch_admission_fixtures.py` | registered-design and launch-admission values |
| `d1_resume_fixtures.py` | real checkpoint-boundary and hard-crash fixtures |
| `d1_resume_hard_crash_fixture.py` | a hard process loss that leaves journal work after a checkpoint |
| `d1_runtime_fixtures.py` | deterministic runtime contract fixtures |
| `d1_telemetry_fixtures.py` | small canonical journals for the authoritative read side |
| `d1_trainer_fixtures.py` | typed plans and clocks for the fresh trainer |
| `d1_verdict_v2_fixtures.py` | v2 decision fixtures |
| `d_ladder_golden_fixtures.py` | services around real tiny trainer and capture bytes, for the golden flywheel |
| `ladder_campaign_fixtures.py` | a local authority built from the real filing verifier |

The consumer set is found by command, so it cannot go stale as a list:

```bash
grep -rlE '^(from|import) (d1_[a-z_0-9]*fixtures?|d_ladder_golden_fixtures|ladder_campaign_fixtures)\b' \
  tests --include='test_*.py' | sort
```

Open for the M9 design gate: "replay equality on every retained D1 fixture" has no stored bytes
to compare against, because the fixtures are generators. The reference this file can give is the
consumer set and its result at the epoch tag. If M9 needs byte equality, it captures each
fixture's canonical output at the epoch tag first, before it moves the owning codec.

### Other sealed evidence, same procedure, not in the corpus

| pointer | content |
| --- | --- |
| `results/h0-final-probe-1k-v4.92c9db.dvc` | H0 final probe, wave 1 host side, 161 files |
| `results/h0-final-probe-resume-v4.1755b0.dvc` | H0 final probe, wave 2 host side, 123 files |
| `results/h0-final-probe-checks-1755b0.dvc` | the in-image K1 to K7 check run, 7 files |
| `results/h0-plumbing-1k-v3-HEpP7g.dvc` | H0 plumbing wave, 4 files, verdict `tracer_partial` |
| `results/captured.dvc` | E-series captured telemetry, 76 files |
| `results/field-pretrain-s42.jsonl.dvc` | E field pre-training telemetry, one file |
| `results/h-image-publication-*.dvc` | thirteen image publication bundles |

M0 did not read the E pointers or the image bundles. Steps 1 to 4 apply to them unchanged; the
read command in step 5 is the one their own verdict or deploy document names.

### Every tracked pointer is retained

Three more pointers are tracked outside `results/`:

| pointer | content |
| --- | --- |
| `models/captured.dvc` | 18 files, 6,163,051,386 bytes |
| `models/field-pphi-v1.pt.dvc` | one file, 2,502,592 bytes |
| `data/enwik8.dvc` | 2 files, 100,000,073 bytes |

That makes 23 tracked `.dvc` pointers at the epoch tag: 20 under `results/`, 2 under `models/`
and 1 under `data/`. This file says which of them form the replay corpus. It names none of them
as unneeded. An object that no pointer at the current tip names may still be named by a pointer
at an older commit, and replaying that commit needs it. A storage audit therefore starts from
the pointers of every commit that holds evidence, never from this list alone.

## Reference run

The procedure above was executed once, on 2026-11-28 UTC, in a detached worktree of the tag on a
macOS arm64 host. Every row is `[measured]` in that run. The worktree was removed afterwards.

| step | result |
| --- | --- |
| 1, commit | `5f6b6dd92b9da007624ce255492f55b779fd4bfd` |
| 2, toolchain | `mise install` exit 0; `uv sync --all-groups --frozen` exit 0; `uv.lock` unchanged; Python 3.12.15 |
| 4, `h0-final-probe-1k-v4-92c9db` | exit 0; 12 files on disk, 12 in the pointer |
| 4, `h0-plumbing-1k-v3-HEpP7g` | exit 0; 4 files on disk, 4 in the pointer |
| 5, `eval h`, magnitude arm `14637e75` | exit 0; 10,040 bytes; run twice, same digest |
| 5, `eval h`, sphere arm `38c25d26` | exit 0; 10,058 bytes; run twice, same digest |
| 5, the four DVC-backed integration files | 377 passed |
| 5, the D1 fixture consumer set | 117 files, 114 under `tests/unit` and 3 under `tests/e2e`; 1629 passed |

Reference digests of the `eval h` output, SHA-256 of stdout:

| arm | digest |
| --- | --- |
| magnitude, `14637e75` | `443777b45561da117f4e234ec25115ea3509a7bb4d6aba74a5647bdd0c238436` |
| sphere, `38c25d26` | `abeaaf17dd1d05b70ef1c53ae38639274adcbc178c835614b1ca7cc9ce9d99b0` |

The integration selection:

```bash
uv run pytest tests/integration/test_h_elm_projection.py tests/integration/test_h1_observations.py \
  tests/integration/test_h_aggregate.py tests/integration/test_h_final_probe_checks_hepp7g.py -q
```

Limits of this reference, stated so nobody reads more into it:

- It was measured on one host. The digests are of text output that includes floating-point
  values; a different platform may print a different last digit. A mismatch on another platform
  is a question to investigate, not yet a proof of damage.
- It is a worktree of an existing clone. The fresh-clone run is M12's.
- It covers the host side only. Nothing here re-executed a sealed chain inside an image.

## What this recipe does not cover

- **Migration.** There is none. Sealed evidence is never rewritten to a new family name.
- **Checkpoint resume across the epoch.** Not possible by design; see "Two epochs".
- **Evidence sealed after the refactor.** It is read by the closing tip's readers, under the
  closing identity pin of horizon M12.
- **The pre-hexagonal epoch.** Artifacts older than the 2026-09 re-layout replay at the commit
  `docs/identity/read-old-table.md` names, by the same procedure with that commit in step 1.
