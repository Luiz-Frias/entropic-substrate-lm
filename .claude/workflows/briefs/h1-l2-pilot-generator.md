# FOR-1690 [h1-l2] Add the H1 pilot block generator to ladder/register/rungs and the Q7 no-hidden-normalization teeth

Source of truth: Linear FOR-1690 (project P-FOR-44). In-repo copy for the W1 lane (preflight corrections of 2026-10-28 applied on the Linear issue; anchors measured at main 45330219). The Linear issue text is authoritative where the two differ.

Wave W1 (K=3, non-member). Seat Sonnet xhigh. Class: feature + test-hardening (Q7 is a guard-teeth test; no production defect assumed).

## Touch-set (non-member)

- NEW src/entropic_substrate/ladder/register/rungs/h1_pilot.py: generator taking source_digest, prediction steps and transfer limits as inputs; emits the four HProfiles per block (MM, SM, MS, SS; run_class="pilot", stage H1, consumer isotropic, future_weight=0, dev widths, dev schedule); validates through ladder/register/h_preservation.py H1PreservationBlock (reuse; codec_sharing_groups gives donor-first order (MM, MS), (SM, SS))
- NEW tests/test_h1_pilot_manifests.py, tests/test_h1_q7_magnitude_norms.py
- NOT touched: core/sentence/profile.py, h_preservation.py body (read only), docs/h1/preregistration-skeleton.md

## Ruled regime values to encode

lr 1e-3, weight decay 0.01, betas (0.9, 0.95), clip 1, batch 4, codec warm-up 250, evaluation interval = steps/20 (steps a multiple of 20), 20 evaluation batches, importance curve (2..64)+256, 5 continuations; seeds 1001..1008 (one per block, shared by its four cells). TransferLimits sized from the MEASURED checkpoints (quote pods s3tkc88i0oyp96, qkfduyfla3b6rt): codec warm-up checkpoint 524,752,197 bytes (500.4 MiB); each prediction checkpoint 1,149,203,227 bytes (1.07 GiB); the kit runs max_object_bytes 2 GiB / max_total_bytes 16 GiB / max_objects 5000; the earlier "~182 MB, 512 MiB" figure is RETRACTED 2026-10-28. The H construction digest fixture = profile identity with name and seed removed (conductor ruling DH-14).

## RED-first tests

8 blocks with seeds 1001..1008 and no overlap with 1..n; donor-first cell order from codec_sharing_groups; interval = steps/20 and a non-multiple refused; 20 evaluation batches; the step cap parameter honored; round trip through HProfile.from_manifest; Q7: on the development widths and on a checkpointed state (saved and reloaded), the MM and MS sentence and spine state norms are not 1 within tolerance and equal the unnormalized reference values.

## Guard teeth (mutation, each RED)

swap the cell order; break steps/20; let the seed lists overlap; inject F.normalize or an RMS overwrite into the magnitude path in core/sentence/{codec,numerics,spine}.py (temporary mutation, reverted byte-identical; a real finding escalates to W2 as decision-required).

## Acceptance

generator only, digests and steps are parameters; Q7 RED on both injected overwrites; PR body quotes the RED outputs.

## Out of scope

sealed manifests with the real capture digest (W4 L8); profile rule changes.
