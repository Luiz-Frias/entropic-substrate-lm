# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## North star

Build and test a **codec-separated hierarchical latent predictor** that generates text by
predicting the next continuous chunk representation in a sequence. A boundary encoder
constructs latent vectors or sets of vectors from observed text; the main Transformer or
state-space spine advances on that chunk sequence; a local decoder realizes predicted
latents as text. Start with declared sentence boundaries. Paragraph, document-context,
learned-boundary, and recursive levels must earn their value through controlled experiments.

The deployment destination is roughly **500M total parameters, with an upper design envelope
of 1B**, counting every deployed encoder, spine, and decoder parameter once. Training-only
teachers, external pretraining data, and resource costs are accounted for separately. This
is a target, not a measured capability, resource quote, or spending authorization.

Latents remain continuous rather than being forced through an alphabet-sized probability
state. Future laws may induce Fisher geometry for their dynamics, but sensitivity to output
change is not itself evidence of useful prediction. Compare signed predictive improvement,
off-target disturbance, information preservation, and measured resources separately.
Codec separation is an architectural contract; codec equivariance and a meaningful
predictive manifold remain hypotheses to test, not assumptions.

The approved construction tests sentence-vector normalization and spine normalization as
independent 2×2 axes. A magnitude-preserving carrier must not contain a hidden final UnitNorm
or RMS overwrite. No probability-floor/square-root substitution is permitted in the main
latent path. The law is directional plus log-magnitude, with unused magnitude integrated out
for the spherical interface; its Fisher pullback shapes updates rather than becoming state.
Normalization, density, and Fisher math establish their own FP32 computation islands.

The forward contract is `docs/hierarchical-latent-predictor.plan.md`, linked from
`docs/architecture.md`. Existing E-series evidence and D-series construction/registration
identities keep their original scope. The old D-ladder is retained as mechanistic groundwork,
not relabelled as an implemented or scientifically validated chunk-level generator.

## Horizon

The next sufficient slice is one causal sentence-level encoder → predicted latent → text
decoder path that generates multiple continuations without access to future target latents,
survives checkpoint/resume, and traverses capture, fresh analysis, independent review, filing,
and an explicit gate. Design bypasses and interventions before training so latent-signal use
is testable. A passing forward call or teacher-forced reconstruction alone is not completion.

The programme is complete only when a model within the declared deployment envelope has
reproducible generation and held-out evaluation, and each retained mechanism has its own
controlled evidence. The approved H0–H5 itinerary tests the complete tracer, preservation/codec
fidelity, predictive value, geometry, interface/scale transfer, and approximately 500M deployment.
Stages may narrow or retire mechanisms; the final model need not contain every proposed one.

Each scientific stage requires excluded pilots, a committed preregistration, a pilot-powered
paired seed matrix, authoritative capture, fresh-session analysis, arm-blind review where
possible, and filed decisions. Retain the agreed 95% familywise decision standard, secondary
3-sigma report on the same sealed data, and 90% planned-power target. Exact estimands, margins,
corpus, sampling laws, and new construction identities must be sealed before confirmation.
The target parameter count does not authorize a GPU run or fix a training-data budget.

Reuse existing numerical and evidence machinery only after verifying the new composition.
Do not infer new-stage eligibility from old E/D results or local test fixtures. Preserve
registered collapse as valid negative evidence and unexpected invariants as execution defects.
The `Auditable compute-to-quality telemetry` dependency is reused, not duplicated. Quantum
hardware, topology capstones, and unregistered modality expansion are not implied by this goal.

## Commands

```bash
mise trust && mise install   # pinned toolchain (python 3.12, uv, just, prek, ...)
just setup                   # tools + uv sync --all-groups + prek hooks
just check                   # quality gate: fmt-check + lint + typecheck + unit tests (= CI)
just test                    # unit tests only (CPU, <60s)
just test-all                # unit + integration
PYTHONPATH=src uv run pytest tests/test_kappa.py::test_name -m unit   # single test
just train exp=e1-fisher     # exp ∈ {e0-byte, e0-bpe, e1-random, e1-fisher, e2, e3, tiny}
just eval ckpt=models/e3-latest.pt
just download-data           # enwik8 → data/, then: dvc add data/enwik8 && dvc push
just container-build         # podman build (CUDA 12.6 image; deploy: docs/deploy-h100.md)
just pc-ci                   # prek commit + pre-push stages (CI parity)
```

mise loads `.env` into the environment; `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` there
are the R2 S3-protocol credentials the DVC remote needs (remote itself is committed in
`.dvc/config`, bucket `entropic-substrate-lm`). `MLFLOW_S3_ENDPOINT_URL` must be in
`.env` too — `just container-test` preflight-aborts without it rather than failing later
as an opaque boto3 retry-exhaustion error, so a checkout missing it cannot run the live
gate at all. Commit messages are gitlint-checked:
Conventional Commits, title ≤ 72 chars.

## House rules

These live here, inline, rather than in a `HOUSE-RULES.md` imported with `@` —
**measured 2026-08-14: `@`-imports do not expand.** A sentinel placed in an imported
file came back ABSENT from a fresh `claude --print`, while a control question answered
correctly from this file. Whatever an agent must see belongs in `CLAUDE.md` itself.
(A gitignored `CLAUDE.local.md` is likewise never loaded — humans only.)

1. **Publish with `just ship-it`, never a bare `gt submit`.** The live gate is the only
   thing in this repo that runs the MLflow stack: `tests/test_mlflow_live.py` `skipif`s
   on `MLFLOW_TRACKING_URI`, so with no stack up the live tests **skip rather than
   fail** — reporting success by testing nothing. CI runs no containers either. That
   gap is how the gate went red at `main` and stayed red under a green CI.

2. **Probe sequentially. Never draw a conclusion from concurrent containers.** The
   podman VM is 4 vCPU / 4 GB. Several MLflow containers booting at once each import
   the full FastAPI/SQLAlchemy stack and all appear to hang — *including a control*.
   This has produced two wrong conclusions already, one of which became a load-bearing
   comment justifying ~110 lines of machinery. One container at a time, always.

3. **Never fix flakiness by raising a timeout or a retry budget.** `sinks.py` and
   `compose.yaml` both carry the measurement showing bigger numbers did not work. Find
   the cause; the symptom taxonomy will keep growing otherwise.

4. **Comments record measurements, with the measurement.** This is why a comment here
   can be *wrong* rather than merely stale — and a wrong measured claim is a defect to
   retract explicitly, in the commit that fixes it, not to quietly overwrite.

5. **TDD is non-negotiable, and guard-teeth prove both arms.** Failing test first. Then
   break the guarded thing and show the test goes RED — a test that cannot fail is not
   evidence. For a guard, prove it is *load-bearing*: the happy path passing does not
   distinguish "the guard worked" from "the hazard never reached it".

6. **Secrets never enter git.** gitleaks runs at pre-commit and pre-push. `gt auth`
   needs a per-user token; it lives in your own gt config or `.env`, never in
   `mise.toml`. `CC_COMMIT_API_KEY` (used by the personal `commit-msg` recipe) likewise.

7. **Upgrading an existing venv across the mlflow → mlflow-skinny swap breaks it.** Both
   distributions own the same `mlflow/` directory, so removing the former deletes files
   the latter still claims — metadata present, package gone, `import mlflow` fails. CI
   never sees it (fresh venv). Fix:
   `uv sync --all-groups --reinstall-package mlflow-skinny`.

`just ship-it` is tracked, so everyone has it. `just stack` / `just commit-msg` are
personal (`justfile.local`, gitignored) because they need an API key.

## Historical E-series architecture

The components below describe retained E-series implementations. The current forward
architecture is the codec-separated predictor above. Its local generative path is implemented;
complete flywheel integration and remote validation remain open.

Three swappable components under a standard decoder-only transformer, selected per
experiment by `configs/*.yaml` (init × block × norm) — `docs/architecture.md` is the
authoritative version:

1. **Substrate init** (`core/substrate/fisher.py`) — √p of Laplace-smoothed next-byte transition
   rows → points on S²⁵⁵ ∩ ℝ₊, lifted 256→`d_model` by a fixed semi-orthogonal QR map.
   Control arm: uniform-random unit vectors. This is the only thing E1 varies.
2. **Backbone** (`core/substrate/blocks.py`) — nGPT-style: all embedding-dim vectors unit-norm,
   affine-free `Norm()` as the retraction step, residual `h ← Norm(h + α(h_block − h))`
   with learnable eigen-learning-rates. Optimizer stays plain AdamW; geometry lives
   entirely in the forward pass. This is the E2 swap.
3. **κ-attention** (`core/substrate/blocks.py`) — the novel artifact (redesigned 2026-08-11, design
   record in `docs/e-3-instruction.md`): per-head attention logits `−d_κ(q,k)²` with one
   learnable clamped curvature per head (init spread −1/0/+1); E2 is the exact κ=0 member
   (softmax shift-invariance), heads are the mixture-over-geometries. The earlier
   `core/geometry/kappa_norm.py` is retained but measured radially-annihilated (`≡ UnitNorm`,
   pinned by regression test — curvature is relational and cannot act on a single point at
   a chart origin under a unit-norm contract). The anchored arms `e3-anchor` (per-head κ +
   global asymmetric anchors) and `e3-vantage` (global κ + per-head anchors; attribution
   ablation) are code-ready but execution-gated on the E3 readout: per-head κ divergence.
   Symmetric anchors are exactly null (isometry) — pinned by test. `core/geometry/kappa.py` implements the
   κ-stereographic primitives directly; the geoopt submodule is a **test-time oracle
   only**, never a runtime dependency.

**Non-negotiable numerics:** all geometric math runs in FP32 inside
`torch.amp.autocast(enabled=False)` blocks regardless of surrounding bf16 autocast; κ is
clamped and inputs are projected into the component domain with a boundary epsilon before
any map; the load-balance loss must stay finite and gradients must reach κ (unit-tested).
`references/holobyte/holobyte.py` demonstrates the failure mode this prevents.

**Data flow:** enwik8 is DVC-tracked in `data/` (pointer `.dvc` files are git's half of the
contract — `data|models|results/*` are gitignored *except* `*.dvc`); FineWeb-Edu is
streamed, never local. `historical/e_series/train.py` writes `results/*.jsonl` (nats/byte, κ
trajectories, gate histograms) and `models/*.pt` checkpoints; both sync to the R2 remote via
`dvc push`. Both are read back through validated readers, because both feed gates:
`ladder/capture/reader/telemetry.py` is the only admissibility predicate for `results/*.jsonl`
(shared by the CLI preflight and `scripts/run_ladder.sh` — inadmissible telemetry raises, it
never returns an empty value), and `kernel/runtime/checkpoint.py` stamps/checks
`format_version` on `models/*.pt`. See `docs/architecture.md` for why an unversioned `tau` is
refused rather than guessed.

**Tests** (`tests/`, pytest markers `unit` / `integration`) check the vendored κ-math
against geoopt and pin the numerics invariants above.
