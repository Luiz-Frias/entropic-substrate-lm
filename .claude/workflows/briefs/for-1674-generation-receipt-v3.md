# for-1674-generation-receipt-v3: N continuations per prompt with the sampling law recorded (receipt schema v3)

DRAFT 2026-10-22 by the conductor (wave C, fourth lane of a K=4 mass-patch-fanout; issue FOR-1674, parent FOR-1640,
operator ruling D1 2026-10-22: the H0 "stochastic rollout" exit is unprovable from a one-prompt one-continuation
receipt with no sampler, seed or RNG field). Anchors measured at `v0.4.1` (`2823668e`) with `grep -n`; the
implementer re-verifies them in recon. The Linear ticket is authoritative, with one anchor CORRECTED below.
**DG-A lane: every touched source module is a reader-tuple member, and the profile schema id bumps v2 -> v3.**

## Branch

| field                | value                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1674-generation-receipt-v3`                                                                     |
| parent               | the wave base branch (wave B's zipped tip + the briefs commit)                                           |
| cascade `phase.mode` | `arch` (two frozen contracts move: the profile schema id and the generation receipt schema; design gate) |
| publish              | none in the lane; the lane returns `REPIN-HANDOFF` and the conductor lands the single commit (DG-A)      |

## Why (measured at `2823668e`)

- CORRECTION to the ticket: there is no `ladder/run/rollout.py` at v0.4.1. The rollout is `src/entropic_substrate/core/sentence/rollout.py`: `begin_rollout` (`:46`), `advance_rollout` (`:72`), `sample = model.predict_latents(state.context).law.rsample().detach()` (`:92`), the comment "Stable log scores, no probability floor or top-k/top-p truncation" (`:95`), `observation = int(torch.multinomial(log_probabilities.exp(), 1).item())` (`:97`). Nothing records the draw; no generator object is threaded through.
- `src/entropic_substrate/ladder/run/generation.py`: `_prompt` (`:81-87`, ONE validation span), `_summary` (`:89`), the resume reader refusing anything but `h-generation-receipt-v2` (`:132` v1 branch, `:140-145`, parent receipt `:174`), the writer `"schema": "h-generation-receipt-v2"` (`:354`), ONE `begin_rollout` per call inside the autocast context (`:261-264`).
- Other v2 receipt readers: `src/entropic_substrate/ladder/run/trace_capture.py:31-32` (accepted schema list) and `:474` (payload check). No `scripts/*` reader.
- Profile: `src/entropic_substrate/core/sentence/profile.py` `RunSchedule` (`:70`, fields warmup_steps .. `generation_sentences` .. `diagnostic_samples`, the last added by FOR-1664 as "required, no default, last"); `seed`/`source_seed` (`:143-144`, `:173-179`, the step-keyed mixing comment); schema id written at `:223` and checked at `:233` (`h-run-profile-v2`). The seed mixing lives at `ladder/run/execution.py:333-340` (`torch.manual_seed((self.profile.seed * 0x9E3779B1 + self.step) % 2**32)`).
- `h-run-profile-v2` appears in 9 tracked files: the three sources above plus `tests/test_sentence_profile.py`, `tests/test_h_run_descriptor.py`, `tests/test_h_probe_launch.py`, `tests/test_identity_epoch.py`, `docs/deploy-h100.md`, `docs/identity/read-old-table.md` (and the FOR-1664 brief, historical). The FOR-1664 bump is the precedent to copy for shape and for the read-old row.
- Reader tuple (`ladder/run/checkpoint.py:65-118`) MEMBERS in the touch set: `core.sentence.profile`, `core.sentence.rollout`, `core.sentence.sampling` (if a generator helper lands there), `ladder.run.generation`. DG-A protocol.
- Tests naming the surfaces: `tests/test_sentence_rollout.py`, `tests/test_sentence_generation.py`, `tests/test_sentence_trace_capture.py`, `tests/test_sentence_recovery_state.py`, `tests/test_final_leg_admission.py`, `tests/test_sentence_profile.py`.

## Contract surface owned (arch: the design gate rules the form)

Two doors move together. (a) Profile: `RunSchedule.generation_continuations: int` (required, no default, placed per the FOR-1664 convention; smoke classes carry 2; exact-equality rule for development classes as the existing fields), schema id `h-run-profile-v2` -> `h-run-profile-v3`, every committed manifest/fixture updated, a dated read-old row. (b) Receipt: `h-generation-receipt-v3` with `continuations: [...]`, each `{index, seed, sampling_law: {"sampler": "multinomial_temperature_1", "latent": "rsample", "seed": int, "rng_digest": sha256 of the generator state before the first draw}, sentences, log_scores, outcome, boundary}`, `summary` keeping the aggregate; per-continuation seed = the step-keyed family applied to the continuation index (document the exact expression, e.g. `(seed * 0x9E3779B1 + (step << 8 | index)) % 2**32` or whatever the gate rules; it must be injective over index for a fixed step and stay below 2**32). Design questions for the gate: G1 thread an explicit `torch.Generator` through `advance_rollout` (rsample via the generator's device, multinomial with `generator=`) vs G2 seed the global RNG per continuation as `execution.py:340` does for evaluation. Recommendation: G1 (an explicit generator makes "same seed reproduces byte-identically" a property of the function, not of global state, and the digest is read from the generator object); the gate records the alternative. v2 receipts stay readable through the read-old window (the HEpP7g capture replays against its own epoch; `trace_capture.py:31` keeps v2 in its accepted list).

## Scope

1. **C1, tests RED at the parent** (`tests/test_sentence_rollout.py`, `tests/test_sentence_generation.py`, `tests/test_sentence_profile.py`, additive): two continuations of one prompt with different seeds differ; the same seed reproduces byte-identically; the receipt carries `continuations` of length `generation_continuations` each with the sampling-law block; a v2 receipt is still readable (read-old) and a profile without the new field is refused.
2. **C2, implementation** per the gate: rollout generator threading, N continuations under one prompt in `generation.py`, receipt v3 writer and readers (`generation.py:132-174`, `trace_capture.py:31-32/:474`), profile field + schema bump + fixtures/manifests + `docs/identity/read-old-table.md` row (the read-old row is conductor-owned docs; the lane STAGES it and the conductor commits it in the DG-A commit).
3. **C3, DG-A hand-off** as in FOR-1672/1673: stage everything (including the regenerated fixtures and the `.secrets.baseline` rewrite if the hook demands it), save the cached diff, quote the expected RED (regime-row digest tests only) and the regime rows, return `REPIN-HANDOFF`.

## Guard teeth by mutation

| id | mutation                                              | expected RED                                                  |
| -- | ----------------------------------------------------- | ------------------------------------------------------------- |
| M1 | ignore the seed (draw from the global RNG)            | the same-seed byte-identical pin RED                          |
| M2 | drop the sampling-law block from each continuation    | the receipt-shape pin RED (and the K5 check reports UNPROVEN) |
| M3 | emit one continuation regardless of the profile count | the length pin RED                                            |
| M4 | keep writing `h-generation-receipt-v2`                | the schema pin RED                                            |

## Allowed files

`src/entropic_substrate/core/sentence/rollout.py`, `src/entropic_substrate/core/sentence/profile.py`, `src/entropic_substrate/core/sentence/sampling.py` (only if the gate places a generator helper there), `src/entropic_substrate/ladder/run/generation.py`, `src/entropic_substrate/ladder/run/trace_capture.py` (reader list only), the committed profile manifests/fixtures that carry `h-run-profile-v2` (regenerated, listed in the hand-off), `tests/test_sentence_rollout.py`, `tests/test_sentence_generation.py`, `tests/test_sentence_profile.py`, `tests/test_sentence_trace_capture.py`, `tests/test_h_run_descriptor.py`, `tests/test_h_probe_launch.py` (fixture strings only), `docs/identity/read-old-table.md` (STAGED for the conductor). Never: `tests/test_identity_epoch.py` (conductor re-pins), `docs/deploy-h100.md` (the post-wave image re-pin ceremony owns it), `ladder/run/execution.py` evaluation seeding, numerics, the sampler or temperature themselves (out of scope: preregistration territory).

## Acceptance

C1 RED at the parent, GREEN at the staged tip; M1-M4 RED one at a time and restored with the RED output quoted; `just check` at the staged tip RED ONLY on the regime-row digest tests and GREEN after the conductor's re-pin commit; the exit-gate runner green at the conductor's commit; the design-gate ruling (G1 or G2) and the per-continuation seed expression in the hand-off report and the commit message; the v2 read-old window stated in the read-old row; diff confined to the Allowed files. Consequence recorded, not acted on: image rebuild and deploy-doc re-pin (v0.4.2) follow wave C before the final probe.
