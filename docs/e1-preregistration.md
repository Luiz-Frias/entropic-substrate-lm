# E1 pre-registration — embedding-init comparison (paired, two-stage)

This document is committed BEFORE any confirmatory run or analysis and is the
binding design. Any analysis deviating from it must say so explicitly and is
exploratory by definition. Confirmatory analysis happens in a fresh agent
session under `docs/peer-review-protocol.md`.

## Hypothesis

H1: at ~85M params on enwik8 (byte-level, configs `e1-fisher` / `e1-random`,
identical except `embedding_init`), the √p Fisher–Rao init improves final
nats/byte over uniform-random unit-vector init. Direction: with
`d_s = y_random,s − y_fisher,s`, H1 predicts `mean(d) > 0`.

## Design

- **Pairing:** same seed `s` ⇒ identical data order and all non-embedding
  init; evidenced per run by the `data_order_digest` JSONL field (Appendix A).
  A digest mismatch within a pair INVALIDATES the pair (reported, excluded).
- **Stage 1 seeds:** {1, 2, 3, 4, 5} (df = 4), analyzed once.
- **Stage 2 seeds:** {6, 7, 8, 9} (cumulative n = 9, df = 8) — run ONLY if the
  stage-1 CI straddles 0.
- **Primary metric:** final nats/byte, read exclusively via
  `entropic_substrate.telemetry.final_eval_metric(path, "nats_per_byte")` on
  DVC-captured JSONLs (`capture-results`).
- **Estimator:** CI = mean(d) ± t_{0.9875, n−1} · sd(d)/√n — the total 5%
  alpha is split across the two looks (Bonferroni, per-look 2.5%):
  t_{0.9875,4} ≈ 3.50 (stage 1), t_{0.9875,8} ≈ 2.75 (stage 2, cumulative).
- **Verdict rule:** CI entirely above 0 → H1 supported. CI entirely below
  +ε → no practically relevant improvement (ε reported alongside). Otherwise
  → inconclusive (stage 2 if not yet run; else reported as inconclusive).
- **Exclusion rule (only this one):** a run is excluded solely for a
  non-finite final loss; an exclusion always drops the whole PAIR and is
  reported in the analysis.
- **Pilot declaration:** the 2026-08-17 `e1-fisher` run (run id `c6025e…`,
  default config seed) established cost/duration/plumbing. Pilots are never
  part of the confirmatory set.

## Integrity chain

`data/enwik8.dvc` md5 `7722de7ace6575299cd8ab788c2fd201.dir` → `config_hash`
param/tag per run → `git_sha` param per run → run artifacts → DVC-captured
JSONL md5s. The analysis session receives only the review packet
(`docs/peer-review-protocol.md`) — never this repository's narrative docs and
never the live tracking server.

## Appendix A — data_order_digest

SHA-256 over the source offsets of the first 32 training batches
(`DATA_ORDER_DIGEST_BATCHES`), emitted ONCE as the additive fields
`data_order_digest` + `data_order_batches` on the first logged train record
at/after the window — or on the final record of a run shorter than 32 steps,
which then attests its actual batch count. Fresh runs only: a resumed run
never attests (absence = not attested, never "same order"). Spec pinned by
unit tests (tests/test_data.py, tests/test_train_smoke.py). Purely additive
to the JSONL schema: existing-field serialization is unchanged.

## Amendment 1 — ε fixed (2026-08-18, post-run / pre-analysis)

ε = 0.01 nats/byte. Basis, chosen with NO access to any confirmatory e1
outcome (the amender had seen: the e0 baselines, and the declared 2026-08-17
pilot): one enwik8 reporting step (0.01 bits/byte ≈ 0.0069 nats/byte) rounded
up to ≈1.3% of the measured e0-byte baseline (0.7738 nats/byte). Committed
after the stage-1 runs completed but before any analysis session or arm-level
readout; every other registration term is unchanged.
