# Pre-registration — paired two-arm comparison (arms A, B)

Committed before any confirmatory run; binding. Amendment 1 (epsilon) was
committed after the runs completed but before any analysis or arm-level
readout (timestamps in manifest.txt).

## Hypothesis

H1: arm A's embedding initialization improves final nats/byte over arm B's
(lower is better). With d_s = y_B,s - y_A,s, H1 predicts mean(d) > 0.
Architecture, data, and per-seed conditions are identical between arms; only
the embedding initialization differs.

## Design

- Pairing: same seed s implies identical data order and all non-embedding
  init; evidenced per run by the data_order_digest field (Appendix A). A
  digest mismatch within a pair INVALIDATES the pair (reported, excluded).
- Stage 1 seeds: {1, 2, 3, 4, 5} (df = 4), analyzed once.
- Stage 2 seeds: {6, 7, 8, 9} (cumulative n = 9, df = 8) — run ONLY if the
  stage-1 CI straddles 0.
- Primary metric: final nats/byte, read exclusively via reader.py's
  final_eval_metric(path, "nats_per_byte").
- Estimator: CI = mean(d) +/- t_{0.9875, n-1} * sd(d)/√n — the total 5%
  alpha is split across the two looks (Bonferroni, per-look 2.5%):
  t_{0.9875,4} ~= 3.50 (stage 1), t_{0.9875,8} ~= 2.75 (stage 2, cumulative).
- Verdict rule: CI entirely above 0 -> H1 supported. CI entirely below
  +epsilon -> no practically relevant improvement. Otherwise -> inconclusive
  (stage 2 if not yet run; else reported as inconclusive).
- Exclusion rule (only this one): a run is excluded solely for a non-finite
  final loss; an exclusion always drops the whole PAIR and is reported.
- Pilot declaration: one declared pilot run of arm A (2026-08-17) established
  cost/duration/plumbing. Pilots are never part of the confirmatory set.

## Amendment 1 — epsilon fixed (2026-08-18, post-run / pre-analysis)

epsilon = 0.01 nats/byte. Basis, chosen with NO access to any confirmatory
outcome of this comparison (the amender had seen baseline runs outside this
comparison and the declared pilot): one enwik8 reporting step (0.01 bits/byte
~= 0.0069 nats/byte) rounded up to ~=1.3% of a byte-level baseline outside
this comparison (0.7738 nats/byte). Every other registration term unchanged.

## Appendix A — data_order_digest

SHA-256 over the source offsets of the first 32 training batches, emitted
ONCE as the additive fields data_order_digest + data_order_batches on the
first logged train record at/after the window — or on the final record of a
run shorter than 32 steps, which then attests its actual batch count. Fresh
runs only: a resumed run never attests (absence = not attested, never "same
order").

## Extension registration — seeds 6-9 (outcome-aware; committed before any extension run)

The stage-1 confirmatory phase closed with its filed verdict. This extension
was registered AFTER that outcome was known and BEFORE any extension run
launched: seeds {6, 7, 8, 9}, same paired design, same container image
digest, same GPU class, same corpus and configs.

- Primary: cumulative paired analysis over seeds {1..9}. STRONGLY_SUPPORTED
  iff mean(d) / (sd(d)/√9) > t_{0.99865, 8} = 4.277 (the one-sided 99.865%
  lower confidence bound of mean(d) exceeds 0). Otherwise
  NOT_STRONGLY_SUPPORTED. This label is reported ALONGSIDE — never
  replacing — the stage-1 verdict.
- Practical relevance is read against the ratified epsilon = 0.01 nats/byte,
  unchanged.
- Clause precedence (declared after extension runs launched, before any
  completed or was analyzed): the direction label is primary; the
  practical-relevance rider is mandatory on every quotation of the verdict.
- Exclusion rule unchanged (non-finite final loss drops the whole pair);
  pairing rule unchanged (a within-pair data_order_digest mismatch
  invalidates the pair).
