#!/usr/bin/env bash
# Worktree post-create ceremony — invoked by the global WorktreeCreate hook's
# project-extension seam with CWD = the fresh worktree. Best-effort by contract:
# report failures loudly on stderr, never block worktree creation.
#
# Ported from sales-collateral-agx, whose version encoded ITS sprint-measured
# provisioning set (mise + a DUAL bun install, root and web/). This repo's set is
# different, and the third item below was measured the hard way during the
# observability sprint:
#
#   1. mise toolchain (python 3.12, uv, just, prek, shellcheck/shfmt/bats)
#   2. `uv sync --all-groups` — there is no bun and no web/ subproject here
#   3. .env — GITIGNORED, so `git worktree add` does NOT carry it over. A fresh
#      worktree therefore has no R2 / HF / RunPod credentials, and anything that
#      touches the DVC remote or the MLflow artifact store fails with an auth
#      error that looks like a code defect rather than a missing file. mise reads
#      .env via `_.file = ".env"`, so its absence is silent until something needs
#      a key.
set -u

log() { printf 'worktree-post-create: %s\n' "$*" >&2; }

if command -v mise >/dev/null 2>&1; then
  log "mise trust"
  mise trust >&2 2>&1 || log "mise trust FAILED (non-blocking)"
  log "mise install"
  mise install >&2 2>&1 || log "mise install FAILED (non-blocking)"
else
  log "mise not on PATH — skipping toolchain install"
fi

if command -v uv >/dev/null 2>&1; then
  log "uv sync --all-groups"
  uv sync --all-groups >&2 2>&1 || log "uv sync FAILED (non-blocking)"
else
  log "uv not on PATH — skipping dependency install"
fi

# Credentials: copy from the worktree's OWNING repository, resolved via git
# rather than assumed, so this keeps working if the checkout ever moves.
if [ -f .env ]; then
  log ".env already present — leaving it alone"
else
  main_root="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
  main_root="${main_root%/.git}"
  if [ -n "$main_root" ] && [ -f "$main_root/.env" ]; then
    cp "$main_root/.env" .env && log "copied .env from $main_root"
  else
    log "WARNING: no .env found at ${main_root:-<unresolved>} — R2/HF/RunPod-dependent commands will fail until one is provided"
  fi
fi

log "ceremony complete"
