# pre-for-1567: Checkpoint publication independent of run history, bounded R2 verification-read retry

Authoritative brief for one pre-phase rung, written 2026-09-28 by the rung-brief lane. Plan: `docs/plans/p14-consolidation.plan.md`, section *Prerequisites and pre-phase rungs*, row 1. Linear: FOR-1567. Every premise below was measured on the live tree whose trunk ref `refactor/p14-consolidation` read `4e0ae6a` at authoring (read from the ref file). The phase briefs cite `eb93d5d`; that drift is recorded for the conductor, and the prelaunch ceremony re-probes every premise at the re-stamped SHA.

## Branch

| field                | value                                                                                                      |
| -------------------- | ---------------------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1567-checkpoint-publication-history`                                                              |
| parent               | `refactor/p14-consolidation`, first rung, landed before every other rung                                   |
| kind                 | pre-phase rung, not a manifest phase (ruling 7)                                                            |
| cascade `phase.mode` | `arch`, always explicit                                                                                    |
| `gates.liveRequired` | `false`: no file under `ladder/capture/adapters/` is in the envelope                                       |
| gates                | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":false}`           |
| `repoToplevel`       | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout, never a worktree                     |
| publish              | by the conductor, after its re-pin commit, through `just ship-it`; never a bare `gt submit` (house rule 1) |
| launch precondition  | no H run in flight or pending resume; see *Identity members*                                               |

## Contract surface owned

| contract                                                                                                                                                                                                     | kind     | frozen_after |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | ------------ |
| per-snapshot publication cost: the k-th checkpoint snapshot PUTs only keys this worker process has not already published and verified; the manifest still lists every file                                   | behavior | true         |
| bounded verification-read retry: 3 attempts total with backoff, on R2 verification READS only, never on a PUT; the count lands in the worker event record                                                    | behavior | true         |
| `publication` block in `h-worker-event-v1` records: object counts, retry count, per-leg seconds; written on every `checkpoint-*` event (design-gate ruling DG-C), read-old tolerated for one rollback window | door     | true         |

"The receipt" in the ticket is read here as the worker event record that follows each snapshot (`object_worker.py:50-68`, read by `h_probe_events.py`). A placement in `h-worker-snapshot-v1` instead would change the exact-key check at `worker_snapshot.py:122-125` and the host materialize path: that is a `DECISION`, not an edit.

## Contracts consumed

- nothing beyond the trunk at the parent

## Sacred invariants

Verbatim source, then the tripwire that detects a breach at fire time, then the guard-teeth recipe.

| invariant                    | verbatim                                                                                                                                                                             | tripwire                                                                                                                                                                                                                                                                                                                                                                 | guard-teeth recipe                                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| House rule 5                 | "**TDD is non-negotiable, and guard-teeth prove both arms.** Failing test first. Then break the guarded thing and show the test goes RED — a test that cannot fail is not evidence." | every behaviour-changing test is committed RED before its fix commit                                                                                                                                                                                                                                                                                                     | perform each mutation in *Guard teeth by mutation*, show RED, revert, show GREEN; both arms in the PR body      |
| House rule 3, with carve-out | "**Never fix flakiness by raising a timeout or a retry budget.**"                                                                                                                    | the diff adds exactly one retry budget: the verification-read retry ruled as an H-lane prerequisite 2026-09-28 and measured by the magnitude arm's `D1TrainerS3ObjectError provider_failure` on a READ at 15:01:55Z (FOR-1567). `worker_command.py:68` keeps `max_attempts` 0, so the SDK never retries a PUT. Any other timeout, deadline or retry change is a DECISION | a reviewer greps the diff for timeout, retry, deadline, backoff, max_attempts; the one ruled hit cites this row |
| House rule 4                 | "**Comments record measurements, with the measurement.**"                                                                                                                            | the retry constants' comment says the attempt count is ruled and the backoff schedule is UNMEASURED; no comment claims a wall-clock saving                                                                                                                                                                                                                               | a wrong measured claim is retracted explicitly in the commit that fixes it                                      |
| Reader identity (R-PIN)      | "the door frozen after p4 is the LABEL and the module-set SHAPE, while the composed digest below re-measures on every phase that edits a member's bytes (ruling R-PIN)"              | ruling 7 exempts this rung from the rail-only rule: `git diff --name-only <parent>...HEAD` intersected with the 51 reader files equals a subset of the members listed under *Identity members*, and nothing else                                                                                                                                                         | proven at p0: flipping one byte in a member turns the regime-row tests RED until the conductor re-pins          |
| Failure semantics            | "measurement/eval path → fail LOUD, never a fail-open wrapper."                                                                                                                      | the retry re-raises the last error with its original class after attempt 3; no `except` swallows, counts-and-continues or returns a default                                                                                                                                                                                                                              | the exhaustion test asserts the original `D1TrainerS3ObjectError` class propagates                              |
| Layer contract               | `kernel < core < ladder < historical < composition`, enforced by `lint-imports` with zero exceptions since p13                                                                       | `lint-imports` green at the tip                                                                                                                                                                                                                                                                                                                                          | proven at p0 and p13: an upward import turns the layers contract RED                                            |
| Exception surface            | "no change to any caller-visible exception type/message beyond what the remedy text states" (every p14 ticket's re-validation checklist)                                             | existing tests asserting exception types and messages pass unmodified                                                                                                                                                                                                                                                                                                    | a changed-output-path test still raises `worker output changed during snapshot publication`                     |
| Flywheel prerequisite        | "no p14 phase or pre-phase rung launches an h-ladder flywheel experimental run, spends GPU, or starts a pod without a fresh operator green light obtained at that time" (registry)   | no command in the lane creates compute or touches RunPod                                                                                                                                                                                                                                                                                                                 | any such action is a DECISION-NEEDED halt                                                                       |

## Identity members

All touched members are in the checkpoint tuple (`ladder/run/checkpoint.py:65-117`, 51 members) and none is in the source tuple (`ladder/capture/capture.py:29-37`, 7 members) `[M, comm of the envelope against both tuples]`. So only the `h-checkpoint-reader-v2` composed rows re-measure; `h-source-reader-v3` stays byte-identical.

Expected, 6: `ladder/run/object_worker.py`, `ladder/run/worker_snapshot.py`, `ladder/run/worker_transport.py`, `ladder/run/runner.py`, `ladder/run/seals.py`, `ladder/run/evaluation_records.py` (added by design-gate ruling DG-B below: `read_evaluation` may consume the memoised, already-verified parent seal so the L1/L2 bullet holds at evaluation checkpoints too).
Permitted only when the lane shows the need in the PR body, up to 7: `ladder/run/worker_execution.py`, `ladder/run/worker_command.py`.
The plan's row names `journal.py` on the path; this rung leaves it byte-identical (ruling 5 also exempts it as a JSON site).

Re-pin: the conductor, never the lane, runs `tools/regime_rows.py` in a separate commit after the lane's last commit, touching `tests/test_identity_epoch.py` and one dated row in `docs/identity/read-old-table.md`. Until then the identity tests are RED by design. Consequence, stated plainly: `checkpoint.py:304-305` refuses a checkpoint whose reader digest differs, so every checkpoint already on R2 from exit-gate attempt 1 becomes unreadable, attempt 2 starts fresh, and the H image must be rebuilt from the new tip before any green-lit run.

## Scope

Measured cause reading, FOR-1567 preflight section: the stall grows about 90 s per checkpoint from the first one. Three legs repeat the whole run history at every checkpoint `[M, code reading at 4e0ae6a]`:

- L1: `seal_step` re-walks the whole seal chain through `read_seal` (`runner.py:236-243`, `seals.py:88-186`), re-reading every ancestor journal and every ancestor `.pt` through `checkpoint_metadata`.
- L2: `read_training_progress` walks the same chain again after every seal (`runner.py:304-311`).
- L3: the snapshot hashes every accumulated file twice (`worker_snapshot.py:61-76` and `:89-96`) and republishes every one (`object_worker.py:206-212`); each unchanged key fails the `IfNoneMatch` precondition and is then read back in full, inside the adapter and again at `worker_transport.py:35`.

Work, in order:

1. Instrument first. Per-leg seconds for L1, L2, L3 plus object counts land in the event's `publication` block. The carrier from the runner to the worker is the lane's choice, as a keyword with a `None` default, so `composition/cli.py` and `composition/wiring/h_tiny_tracer.py` stay untouched.
2. L3: the worker process keeps an in-process ledger of keys it has published and verified. A path already in the ledger is not re-read or re-PUT; its manifest entry comes from the ledger. A path whose bytes differ from its ledger entry still fails loud, before any network call. The snapshot manifest (`h-worker-snapshot-v1`) keeps listing every file, so `materialize_worker_snapshot` is unchanged.
3. L1 and L2: inside one training process, a new seal is verified against its already-verified parent, a Merkle-chain step: read the new node, check it links to the memoised parent digest. `read_seal`'s full-chain walk stays the admission predicate at resume, pause and completion, and in every fresh process.
4. Retry: wrap the verification read at `worker_transport.py:35` only, never `objects.publish` at `:32`. Whether the adapter's in-publish read-back (`d1_trainer_s3_object_adapter.py` `publish`, post-PUT `_read` and collision-confirm `_read`) can raise on this path, and how to cover it without re-issuing `put_object`, is the lane's measurement; covering it by re-calling `publish` is forbidden and is a DECISION. The adapter's default for D1 callers stays zero retries.

## ACCEPTANCE

Machine-checkable, one line each.

- `tests/test_checkpoint_publication_history.py` (new, RED-first): a counting fake object port over k = 1, 2, 3, 5 checkpoints; PUTs at checkpoint k equal the files new since checkpoint k-1 plus the marker and the event, independent of k. Mutation: restore republish-all, RED.
- Same file: verification reads at checkpoint k are bounded by new files, independent of k.
- Same file: L1 and L2 ancestor reads per checkpoint are constant in k (count `checkpoint_metadata` and `read_journal` calls). Mutation: drop the memoised parent, RED.
- A resume in a fresh process still walks the full chain: a tampered ancestor journal is refused at resume. Mutation: let resume trust a memo, RED.
- A path whose bytes change after it was published raises before any PUT for that path.
- Retry, both arms: two transient read failures then success publishes once and records 2; three failures raise the original error class; `put_object` is called exactly once in both arms.
- `h_probe_events` admits a checkpoint event with and without `publication` (read-old) and the worker always writes it (write-new); both arms tested in `tests/test_h_probe_events.py`.
- Wall-clock on R2 is UNMEASURED offline: the PR body claims operation counts only and names the unmeasured leg timings.
- remap_check at the tip over a mapping re-derived by the p13 procedure: inversions <= 1, package cycles <= 4, output in the PR body.
- Mechanical counts at the tip at or under the epoch values below, output in the PR body.
- `git diff --name-only refactor/p14-consolidation...HEAD` is a subset of *Allowed files*; its intersection with the 51 reader files is a subset of *Identity members*.
- `just check` green; exit gate `just pc-ci` through `uv run python scripts/run_exit_gate.py`; `lint-imports` clean. One full suite at a time, with `PYTEST_XDIST_AUTO_NUM_WORKERS=2`.

Ratchet commands:

```text
uv run python tools/import_graph.py src/entropic_substrate <out>
uv run python - <out>/mapping.json   # stdin: the reproducing procedure in docs/plans/measurements/p13-sweep-remap.txt
uv run python tools/remap_check.py <out>/edges.json <out>/mapping.json
# must print: layer inversions <= 1, package-level cycles <= 4 (epoch 4e0ae6a (src tree identical to 651af30): 1 and 4, 378 modules, 1805 edges)
```

```text
grep -rlE 'def (_pairs|_strict_pairs|_nonfinite|_reject_constant)\b' src --include='*.py' | wc -l   # epoch 36
grep -rn 'def _mapping(' src --include='*.py' | wc -l                                             # epoch 23
grep -rn 'separators=(",", ":")' src --include='*.py' | wc -l                                      # epoch 68
grep -rliE 'lowercase SHA-?256|lowercase sha256' src --include='*.py' | wc -l                      # epoch 29
grep -rn 'type: ignore\[arg-type\]' src --include='*.py' | wc -l                                  # epoch 19
find src scripts -name '*.py' -exec wc -l {} + | awk '$1>1000 && $2!="total"' | wc -l              # epoch 2
```

## Non-goals

Thoroughness beyond ACCEPTANCE is NOT welcome. Anything that looks necessary for correctness is a `DECISION`, not an edit.

- Does NOT change `checkpoint_interval`, `evaluation_interval`, the schedule, `controller_timeout_seconds` or any profile field: operator-ruled scientific and launch fields.
- Does NOT change the request's `max_objects` / `max_total_bytes` budgets. Latent risk to report, not fix: the manifest still inventories every file, so the budget checks at `worker_snapshot.py:57` and `:87` still grow with history.
- Does NOT retry any PUT, list or request read; does NOT touch `worker_command.py:68`'s `Config` line.
- Does NOT add a parameter without a default to `run_probe_training` or `execute_probe_request`; `composition/cli.py`, a reader-tuple member, stays untouched.
- Does NOT add the operator stop (FOR-1617, next rung).
- Does NOT build or publish an image, launch pods, spend GPU, or start any h-ladder flywheel run (ruling 8).
- Does NOT edit the runbook memory; that is the conductor's.

## Allowed files

The escalation door: if correctness would require a file outside this envelope, a contract surface, or a sacred invariant, DO NOT EDIT. Return `DECISION` with the verification you ran. Every path was verified to exist at `4e0ae6a` unless marked new.

- `src/entropic_substrate/ladder/run/object_worker.py`
- `src/entropic_substrate/ladder/run/worker_snapshot.py`
- `src/entropic_substrate/ladder/run/worker_transport.py`
- `src/entropic_substrate/ladder/run/runner.py`
- `src/entropic_substrate/ladder/run/seals.py`
- `src/entropic_substrate/ladder/run/evaluation_records.py` (design-gate ruling DG-B; the fresh-process full walk stays the admission predicate)
- `src/entropic_substrate/ladder/run/worker_execution.py` (only if the timing carrier needs it)
- `src/entropic_substrate/ladder/run/worker_command.py` (only if the retry wiring needs it; never its `Config` line)
- `src/entropic_substrate/ladder/run/transport/d1_trainer_s3_object_adapter.py` (not a tuple member; D1 default unchanged)
- `src/entropic_substrate/ladder/run/h_probe_events.py` (not a tuple member)
- `tests/test_checkpoint_publication_history.py` (new)
- `tests/test_sentence_object_worker.py`, `tests/test_sentence_worker_snapshot.py`, `tests/test_sentence_worker_transport.py`, `tests/test_sentence_worker_execution.py`, `tests/test_sentence_worker_command.py`, `tests/test_sentence_seals.py`, `tests/test_sentence_runner.py`, `tests/test_sentence_evaluation_records.py`, `tests/test_d1_trainer_s3_object_adapter.py`, `tests/test_h_probe_events.py`
- `tests/test_identity_epoch.py`, `docs/identity/read-old-table.md` (conductor re-pin commit only, never the lane)
- `.secrets.baseline` (standing rule: mechanical companion, digest literals only, stated in the PR body)

Touch set: 20 existing plus 1 new, 21 files; the manifest phases share none of them `[M, overlap against sprint-manifest.json]`.

Never edit, in this rung:

- `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/run_ladder.sh`, `tools/mapping.json`, `tools/regime_rows.py`, `.importlinter`, `pyproject.toml`, `uv.lock`
- `docs/identity/*` and `tests/test_identity_epoch.py`, except the conductor's re-pin commit
- `docs/plans/*`, `docs/reviews/*`, `docs/verdicts/*`, `CLAUDE.md`, `docs/architecture.md`, briefs, the manifest, engine scripts
- every reader-tuple file not listed under *Identity members*, `composition/cli.py` above all

## Already ruled

| id       | status                             | ruling                                                                                                                                                              |
| -------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7        | OPERATOR 2026-09-28                | FOR-1567 and FOR-1617 are `fix/` branches stacked on the trunk before the phases, each with its own conductor re-pin.                                               |
| 8        | OPERATOR 2026-09-28                | No h-ladder flywheel run without a fresh operator green light; no pods, no spend.                                                                                   |
| H-PREREQ | OPERATOR 2026-09-28                | Before any flywheel run: FOR-1567 lands, making per-checkpoint work history-independent, with a bounded retry on R2 verification reads (runbook memory, section 6). |
| R-PIN    | CONDUCTOR 2026-09-22               | Rewriting a reader-tuple member re-pins its regime rows in the same change; label and module set unchanged.                                                         |
| P14-LIVE | PLANNING LANE 2026-09-28, ratified | Only the phase touching `mlflow.py` is live-required; publication still runs `just ship-it`.                                                                        |

Conflict named aloud: the registry's never-edit invariant says `tests/test_identity_epoch.py` is touched "exactly once for the whole sprint", by the rail. Ruling 7 and the registry's own flywheel invariant grant this rung its own conductor re-pin. This brief follows ruling 7; the conductor reconciles the registry sentence.

## Guard teeth by mutation

| mutation                                                         | must go RED                                      |
| ---------------------------------------------------------------- | ------------------------------------------------ |
| republish every accumulated path at each checkpoint              | the O(new) publication count test                |
| drop the memoised parent so `seal_step` walks the full chain     | the constant ancestor-read test                  |
| let a fresh-process resume trust the memo                        | the tampered-ancestor resume test                |
| retry the PUT instead of the read                                | the `put_object` exactly-once assertion          |
| swallow the third read failure                                   | the exhaustion test expecting the original class |
| drop the `publication` block from the reader's accepted key sets | the write-new event test                         |

## Planned commits

Conventional Commits, title <= 72 characters, message from a file with `git commit -F`, hooks always on, never `--no-verify`, never amend.

Superseded by design-gate ruling DG-A (below): the pre-commit hook runs the unit suite, and the unit-marked regime-row digest test refuses any commit that changes a reader member's bytes until the rows are re-pinned, which ruling 7 reserves to the conductor. The rung therefore lands as ONE conductor commit, `fix(run): publish only new snapshot keys; bounded verify-read retry` (title <= 72), carrying the lane's staged change set plus the re-pin of `h-checkpoint-reader-v2` (`tests/test_identity_epoch.py`, one dated row in `docs/identity/read-old-table.md`). RED-first evidence lives in the lane's report, not in separate RED commits (the hook cannot admit a RED unit test).

## Design-gate rulings (conductor, 2026-09-28, autonomous mode)

The Opus design gate (run `wf_dc1f35cd-87a`) returned DECISION on three items. Each is ruled here by the conductor inside the seam (implementation topology, acceptance instrument, retry leg scope); none re-opens an operator ruling. The alternatives are recorded for the operator's morning review as DECISION-NEEDED items on the artifact (`DN-1567-B`, `DN-1567-C`); the ruled branch is the safest reversible default in each case.

| id   | ruling                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DG-A | **Staged hand-off.** The lane commits nothing that changes a reader member's bytes. It implements the whole rung, runs its RED/GREEN and mutation arms locally and quotes them, then STAGES the complete change set (`git add` every touched file, including new tests and `.secrets.baseline` if the digest literals moved), saves a durable copy with `git diff --cached --binary > /tmp/pre-for-1567-staged.patch`, runs `just check` verbatim and quotes the expected RED (the regime-row digest tests in `tests/test_identity_epoch.py` and nothing else; any other RED is a defect to fix first), runs `lint-imports`, the ratchet and the mechanical counts as labelled diagnostics, quotes both regime rows for both readers (local and CI regimes, `tools/regime_rows.py`), and returns status `DECISION` whose text begins with the literal marker `REPIN-HANDOFF`. The conductor then lands ONE commit (lane change set + re-pin), adds its SHA to `args.conductorCommits`, and resumes the lifecycle; every later reviewer or fixer round that must touch a reader member repeats the same protocol (fixer returns DECISION `REPIN-HANDOFF`; a fixer never edits the identity test). |
| DG-B | **Constant ancestor reads at every checkpoint, evaluation checkpoints included.** `ladder/run/evaluation_records.py` joins *Identity members* and *Allowed files* (its re-pin rides the same conductor commit; the manifest phases touch it 0 times). `read_evaluation` may accept the already-verified parent seal from the in-process memo; the fresh-process full-chain walk remains the admission predicate at resume, pause and completion (acceptance bullet 4 unchanged). Alternative not taken, recorded as `DN-1567-B`: keep the file out and restate the L1/L2 bullet with an O(k) residual at the five evaluation checkpoints per 6K run.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| DG-C | **Retry and `publication` block on the checkpoint leg only.** The `publication` block is written on every `checkpoint-*` worker event; the bounded retry applies only to the checkpoint-snapshot verification reads (outputs and marker). The started, final and failure snapshots and every event publication keep zero retries, because the `complete` event is admitted by exact key-set equality in `worker_completion.py` (outside the envelope; a reader member) and a retry there would be unrecorded, contradicting the operator's "count recorded in the receipt". This covers the measured failure site (step-750 checkpoint sync, 15:01:55Z). Residual, named in the PR body: the final snapshot stays unretried. Alternative not taken, recorded as `DN-1567-C` for the operator: amend the `h-worker-event-v1` door (add `publication` to the `complete` event with read-old/write-new) so the final leg is covered; a door amendment is operator-ruled.                                                                                                                                                                                                                            |

Standing consequence of DG-A for the engine: `just check` cannot be exit 0 at a lane tip that changed reader bytes before the conductor commit; the lane's `REPIN-HANDOFF` report is the certified artifact for the implement stage, and the CERTIFIED verdict is claimed only at the conductor's tip.
