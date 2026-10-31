# FOR-1690 [h1-l2] Add the H1 pilot block generator to ladder/register/rungs and the Q7 no-hidden-normalization teeth

Source of truth: Linear FOR-1690 (project P-FOR-44). In-repo copy for the W1 lane (preflight corrections of 2026-10-28 applied on the Linear issue; anchors measured at main 45330219). The Linear issue text is authoritative where the two differ.

Wave W1 (K=3, non-member). Seat Sonnet xhigh. Class: feature + test-hardening (Q7 is a guard-teeth test; no production defect assumed).

## Touch-set (non-member)

- NEW src/entropic_substrate/ladder/register/rungs/h1_pilot.py: generator taking source_digest, prediction steps and transfer limits as inputs; emits the four HProfiles per block (MM, SM, MS, SS; run_class="pilot", stage H1, consumer isotropic, future_weight=0, dev widths, dev schedule); validates through ladder/register/h_preservation.py H1PreservationBlock (reuse; codec_sharing_groups gives donor-first order (MM, MS), (SM, SS)). Two orders are named separately and both are pinned: the block storage order is MM, SM, MS, SS (`_AXES`, src/entropic_substrate/ladder/register/h_preservation.py:13-18; `CELL_AXES`, ladder/register/rungs/h1_pilot.py:44-50), and the launch order is derived from `codec_sharing_groups`, never listed by hand, as MM, MS, SM, SS with each donor before its recipient (`h1_pilot_blocks`, h1_pilot.py:255-256, into `H1PilotBlock.launch_order`; tests/test_h1_pilot_manifests.py:114-130)
- NEW tests/test_h1_pilot_manifests.py, tests/test_h1_q7_magnitude_norms.py
- NOT touched: core/sentence/profile.py, h_preservation.py body (read only), docs/h1/preregistration-skeleton.md

## Ruled regime values to encode

lr 1e-3, weight decay 0.01, betas (0.9, 0.95), clip 1, batch 4, codec warm-up 250, evaluation interval = steps/20 (steps a multiple of 20), 20 evaluation batches, importance curve (2..64)+256, 5 continuations; seeds 1001..1008 (one per block, shared by its four cells). TransferLimits sized from the MEASURED checkpoints (quote pods s3tkc88i0oyp96, qkfduyfla3b6rt): codec warm-up checkpoint 524,752,197 bytes (500.4 MiB); each prediction checkpoint 1,149,203,227 bytes (1.07 GiB); the kit runs max_object_bytes 2 GiB / max_total_bytes 16 GiB / max_objects 5000; the earlier "~182 MB, 512 MiB" figure is RETRACTED 2026-10-28. The H construction digest fixture = profile identity with name, seed and source_seed removed (conductor ruling DH-14 as shipped, src/entropic_substrate/ladder/register/h_preservation.py `h1_construction_digest` :30-53; this generator sets source_seed = seed, h1_pilot.py:184).

## RED-first tests

8 blocks with seeds 1001..1008 and no overlap with 1..n; donor-first cell order from codec_sharing_groups; interval = steps/20 and a non-multiple refused (steps must be an int >= 1 and a multiple of 20: `h1_pilot_blocks`, src/entropic_substrate/ladder/register/rungs/h1_pilot.py:225-228; tests/test_h1_pilot_manifests.py:175-184 refuse non-multiples, zero, negatives, booleans and non-integers); 20 evaluation batches; the step cap parameter honored; round trip through HProfile.from_manifest; Q7: on the development widths and on a checkpointed state (saved and reloaded), the MM and MS sentence and spine state norms are not 1 within tolerance and equal the unnormalized reference values.

Q7 norm contract, as shipped (tests/test_h1_q7_magnitude_norms.py): the norm is the L2 norm over the last dimension (:149, :165); "is 1" means within `UNIT_TOL` 1e-4 absolute (:34, :120-125); a magnitude sentence norm is "constant" when its spread std/mean is within `SPREAD_TOL` 1e-3 relative (:35-37, :152); equality with the unnormalized reference is `torch.allclose` with `REFERENCE_TOL` 1e-3 as both rtol and atol (:38, :154, :170). The cells are built at the development widths, codec 384 x 4 layers and spine 768 x 12 with context 64 (:48-65), and the claims are re-read on a state that was trained one step, saved, and reloaded into a differently initialized model (:206-235). The tolerances are set for FP32 on CPU (:38). The eight mutants (:334-352) must each report their named finding (:355-361).

## Guard teeth (mutation, each RED)

swap the cell order; break steps/20; let the seed lists overlap; inject F.normalize or an RMS overwrite into the magnitude path in core/sentence/{codec,numerics,spine}.py (temporary mutation, reverted byte-identical; a real finding escalates to W2 as decision-required).

## Acceptance

generator only, digests, steps and transfer_limits are parameters; Q7 RED on both injected overwrites; PR body quotes the RED outputs.

`transfer_limits` is part of the parameter contract, as shipped: `limits: TransferLimits` is a required keyword with no default, it is type-checked, and it is refused when any field is below the measured checkpoint floor, so an implementation cannot hard-code limits (`h1_pilot_blocks`, src/entropic_substrate/ladder/register/rungs/h1_pilot.py:216, :223-224, :243-249; tests/test_h1_pilot_manifests.py:418-438). The limits fix adds a required, caller-supplied `non_checkpoint_bytes` allowance to `max_total_bytes`, because the snapshot publishes journals, evaluations, generation outputs, receipts and the marker beyond the checkpoints and the corpus copy; its dev-width value is measured before L8 seals the limits. As shipped in the L2 fix commit on this stack, `pilot_transfer_limits` takes `non_checkpoint_bytes: int` as a required keyword (h1_pilot.py:147-148) and adds it to `max_total_bytes` exactly once (:175); the checkpoint floor is documented as necessary and not sufficient (:112-122, :246, :275).

## Out of scope

sealed manifests with the real capture digest (W4 L8); profile rule changes.
