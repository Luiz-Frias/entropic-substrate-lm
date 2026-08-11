> Resolution note (2026-08-11): all 15 findings addressed in the initial commit —
> F1/F2 Containerfile rebuilt (dep layer via uv pip compile, git installed, ENTRYPOINT dropped),
> F3 ruff format applied, F4 CI uses --no-sync, F5 c_proj axis fixed + regression test,
> F6/F7/F8/F9/F10 docs corrected, F11 env knob honored in resolve_device, F12/F13 comments,
> F14 deptry+bandit wired (just audit), F15 assumption noted in e2/e3 configs.

# REVIEW — entropic-substrate-lm (Fable, 2026-08-11)

## Verdict: PASS-WITH-FIXES

Core science, tests, and CLI hold up under adversarial checking: 166 unit tests green
(re-run, not trusted from the report), no stubs, tiny E3 run reproduces κ + gate-usage
JSONL, FP32 geometry discipline is structural, aux loss is consumed by the trainer,
submodules/DVC intact. What fails is the packaging/CI shell around it: the Containerfile
cannot build at all (F1), the repo fails its own `just fmt-check`/CI format gate (F3),
and the CI workflow undoes its own CPU-torch install (F4). Also one real doc-vs-code math
divergence in the nGPT weight normalization (F5/F6). All fixable without touching the
experiment logic; do not ship the container or trust CI until F1–F4 land.

## Findings

| # | Sev | File | Issue | Suggested fix |
|---|-----|------|-------|---------------|
| F1 | blocker | Containerfile | Dependency layer runs `uv pip install --system .` with only `pyproject.toml` copied; setuptools src-layout build fails: `error in 'egg_base' option: 'src' does not exist` (reproduced in isolation). Image cannot build, so README step 5, `just container-build`, and all of docs/deploy-h100.md are dead. Secondary: the geoopt `git+https://` dep needs a `git` binary, which the pytorch runtime base image does not guarantee. | Move `COPY src/ ./src/` above the install, or install deps without the project (`uv pip install --system -r <(uv export ...)` / dummy-package trick). Add `apt-get install -y git` or vendor a geoopt sdist. |
| F2 | major | Containerfile | `ENTRYPOINT ["python","-m","entropic_substrate.cli"]` + `CMD ["sleep","infinity"]` compose to `python -m entropic_substrate.cli sleep infinity` → argparse error, container exits immediately. The "sleep-friendly pod" comment is false as written. | Drop the ENTRYPOINT (use CMD-only with an explicit `train ...` example), or make ENTRYPOINT a shell wrapper that execs `"$@"`. |
| F3 | major | src/, tests/, scripts/ | `ruff format --check .` fails: **10 files would be reformatted** (e.g. `init/fisher.py`, manual line wraps that the formatter collapses). This reds out `just fmt-check`, `just check`, mise `check`, the prek pre-push gate, and the CI "Ruff format check" step. Build report only ran `ruff check`, which passes. (Checked with ruff 0.16.2.) | Run `just fmt` (i.e. `ruff format .`) once and commit. |
| F4 | major | .github/workflows/ci.yaml | Steps use bare `uv run ...` after manually installing CPU torch via `uv pip install`. `uv run` auto-syncs the project env from `pyproject.toml`, which re-resolves `torch>=2.6` from the default index (CUDA-bundled, multi-GB) and replaces the CPU wheel — defeating the CPU-only install, massively inflating CI time, and risking runner disk exhaustion. | Use `uv run --no-sync` (or `--no-project`), or configure `[tool.uv.sources]`/`[[tool.uv.index]]` to pin torch to the CPU index so a sync is harmless. |
| F5 | major | src/entropic_substrate/models/blocks.py `normalize_weights()` | Silent divergence from the cited nGPT recipe: the reference (`references/ngpt/train.py:394-408`) normalizes `att_c_proj`/`mlp_c_proj` along **dim=0** (the embedding axis) and everything else along dim=1; this repo normalizes all six matrices along **dim=1**, so `out_proj` and `mlp.proj` end up unit-norm over the hidden/input axis, not the embedding axis. The module docstring's "unit-norm weights on embedding dims" claim is false for those two matrices. Training still runs, but E2/E3 is not the nGPT constraint it cites. | Normalize `out_proj.weight` and `mlp.proj.weight` with `dim=0`; add a unit test asserting the axis against the reference convention. |
| F6 | major | docs/references.md (nGPT entry) | States "**Not taken:** their per-matrix post-step renormalization … E2 normalizes at forward time for simplicity." The code does the exact opposite: `train.py` calls `normalize_weights()` after every optimizer step and there is no forward-time weight normalization. Doc directly contradicts the implementation. | Rewrite the "Not taken" sentence to match the post-step renorm reality (and F5's axis semantics). |
| F7 | minor | docs/architecture.md (Component 3 diagram) | Diagram shows per-component `… → expmap0_κi` then "weighted recombination"; code recombines in the tangent space and applies a **single** `expmap0` at the gate-weighted effective curvature κ_eff (`kappa_norm.py`). Deviation is disclosed in the build report and the layer docstring is accurate, but the docs diagram is not. | Redraw: `projx_κi → logmap0_κi` per component → gated tangent mix → `expmap0(κ_eff)` → renorm. |
| F8 | minor | README.md, docs/deploy-h100.md | Checkpoint paths in examples don't exist: `just eval ckpt=models/e3/last.pt` and `--resume models/<exp>-last.pt`. Actual naming is flat: `models/<name>-latest.pt` / `models/<name>-final.pt` (verified: `models/e3-tiny-final.pt`). | Fix examples to `models/e3-latest.pt` etc. |
| F9 | minor | docs/e-3-instruction.md | `init-fisher --corpus data/enwik8` passes a directory; `load_enwik8_bytes` memmaps a file and would raise (correct path `data/enwik8/enwik8`, which is the CLI default). Also `init-fisher` defaults to `--out data/fisher_init.pt` while e1-fisher/e2/e3 configs expect `data/fisher_init_d768.pt` — the precompute step silently doesn't feed the configs (trainer then falls back to recomputing from the corpus, so runs still work). | Fix the doc command; align the `--out` default with the config path. |
| F10 | minor | docs/e-3-instruction.md | "Every run writes `results/<exp>-<seed>.jsonl`" — actual is `results/<cfg.name>.jsonl`, no seed suffix; multi-seed runs of the same exp append to one file (seed is recoverable only from the `run_start` config record). | Either interpolate seed into the default results path or fix the doc. |
| F11 | minor | .env.example, .github/workflows/ci.yaml | `ENTROPIC_SUBSTRATE_DEVICE` is set in CI and advertised in `.env.example` but never read anywhere in `src/` or `tests/` (grep: zero hits). Dead knob. | Remove it, or honor it in `resolve_device()`. |
| F12 | minor | Containerfile header vs justfile | Header says "SINGLE-FILE multi-stage build" but there is exactly one stage; justfile comment correctly says "single-stage". Cosmetic self-contradiction. | Fix the comment (or actually split dep/src stages). |
| F13 | minor | mise.toml | `[tasks.check-all]` runs `just test-all` but is described as the "Full gate" — dlt convention's check-all = full quality gate + integration, and there is no `check-all` just recipe. Drift is mild but the description over-promises. | Point it at `just check && just test-all` or rename/describe as tests-only. |
| F14 | minor | pyproject.toml | `[tool.deptry]` and `[tool.bandit]` sections exist (mirroring dlt) but neither tool is in the dev dependency group nor invoked by any just recipe/CI step — config for tools that can never run. (BUILD_SPEC's dev-dep list omits them, so this is spec-vs-convention tension, not agent error.) | Either add `deptry`/`bandit` to dev deps + a recipe, or drop the sections. |
| F15 | minor | configs/e2.yaml, configs/e3.yaml | Hardcode `embedding_init: fisher_sqrt`, pre-committing to the "E1 winner" that docs/e-3-instruction.md says to determine empirically ("Compare against E1-winner init; keep the winner"). If E1-random wins, configs need manual editing and the ladder's one-variable-per-rung claim (E1→E2 changes init *and* block) is muddied. | Note the assumption in the configs, or ship `e2-random.yaml` variants. |

## Verified-good (checked, not taken from the report)

- **Tests**: 166/166 green under `-m unit`; all 166 collected tests carry the unit marker; suite <4s CPU. geoopt-oracle comparisons, Taylor-seam continuity, FP32-under-autocast spy test (asserts `float32` + `is_autocast_enabled()==False` inside the layer while caller is in bf16 autocast), κ-gradient-through-clamp, top-2 dispatch accounting all present and real.
- **No stubs**: zero `NotImplementedError` / TODO / FIXME / `...` bodies in src, tests, scripts, configs.
- **Tiny E3 CLI run** (fresh, 20 steps): eval record in `results/review-e3-tiny.jsonl` contains `kappa` (4 norms × 3 components, drifting off ±1/0 init), `gate_usage` (rows sum to top_k=1), `router_entropy`, and train records carry live `aux_loss`.
- **Aux loss consumed**: `Trainer.train_step` computes `total = out.loss + out.aux_loss` and backprops `total`; `backbone.collect_aux_loss()` sums the per-module Switch losses populated by the same forward. Not computed-and-dropped. κ and all 1-D params correctly excluded from weight decay.
- **FP32 discipline structural**: `GatedKappaNorm.forward` wraps its entire body in `torch.autocast(enabled=False)`, casts `x.float()`, gate is declared fp32; κ clamp [−5,5] applied at every entry point; Taylor guard at |κ|<1e-4 with safe inputs on both `torch.where` branches; `projx` ball-boundary eps for κ<0.
- **√p Fisher init matches docs**: Laplace-smoothed rows → sqrt → exact renorm (unit rows, strictly positive orthant), seeded sign-fixed QR lift with `QQᵀ=I` for d_out≥256 (exact isometry, Bhattacharyya inner products preserved); tiny-config d<256 JL-projection fallback is documented and excluded from the isometry claim. Measured bytes/token path real: e0-bpe tiny reports 4.4287, not 4.0.
- **nGPT block (apart from F5)**: retraction residual `Norm(h + α(Norm(h_sub) − h))` with |α| eigen-LRs and base_scale storage matches the reference; sqk unit-q/k with √d_head softmax scale matches; SwiGLU suv scaling matches; `sz` logit scale present.
- **Integrity**: `.gitmodules` lists exactly 4 submodules matching `references/{holobyte,hnet,ngpt,geoopt}` (all populated, `git submodule status` clean); `.dvc/config` has `autostage = true` and is not gitignored (explicit NOTE in .gitignore; shows staged in `git status`); all 5 docs files exist; `ruff check` passes; experiment names in CLI/justfile/README all resolve to real files in `configs/` (verified programmatically).
- **Conventions**: pyproject mirrors dlt (line 100, py312, same select/ignore incl. RUF001-3, basedpyright strict excluding tests/references, strict-markers, unit/integration markers, `[tool.uv] exclude-newer = 2026-08-01`); justfile has all required recipes (`fmt fmt-check lint typecheck test test-all check train download-data container-build` + setup/sync/pc-ci in dlt tone); mise.toml matches the dlt tool set.

## Commands run (repo root, sandbox: Python 3.14 CPU, ruff 0.16.2)

| Command | Outcome |
|---|---|
| `PYTHONPATH=src python -m pytest tests -m unit -q` | **166 passed**, 75 warnings, 3.68s |
| `PYTHONPATH=src python -m pytest tests -q --collect-only` | 166 collected (⇒ every test is unit-marked) |
| `grep -rn "NotImplementedError\|TODO\|FIXME\|XXX" src tests scripts configs` + `...`-body grep | no hits |
| `PYTHONPATH=src python -m entropic_substrate.cli train --exp e3 --tiny --device cpu --steps 20 --results results/review-e3-tiny.jsonl` | completed; eval JSONL has `kappa`, `gate_usage`, `router_entropy`; train records have `aux_loss` |
| `PYTHONPATH=src python -m entropic_substrate.cli train --exp e0-bpe --tiny --device cpu --steps 10 --results results/review-e0bpe-tiny.jsonl` | completed; `bytes/token=4.4287` (measured, not 4.0) |
| `ruff check src tests scripts` | All checks passed |
| `ruff format --check .` | **FAIL: 10 files would be reformatted** (F3) |
| Minimal replication of Containerfile dep layer (`pip wheel` on pyproject-only dir with src layout) | **FAIL: `'src' does not exist`** (F1) |
| `git submodule status` / `git status --short` | 4 submodules populated; `.dvc/config` tracked (AM) |
| `PYTHONPATH=src python -c "resolve_experiment_config(...)"` over all 6 exp names | all resolve; block/norm/init per spec ladder |
| `grep -rn ENTROPIC_SUBSTRATE_DEVICE src tests` | zero hits (F11) |
| Analytic param recount (d=768) | baseline 7.08M/layer ×12 = 84.9M; nGPT 9.44M/layer ×9 = 84.9M — configs/_shared.md arithmetic correct |
| Reference diff: `references/ngpt/train.py:394-408` vs `blocks.py normalize_weights()` | c_proj axis divergence (F5) |
