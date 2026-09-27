#!/usr/bin/env bash
# Worktree post-create ceremony — invoked by the global WorktreeCreate hook's
# project-extension seam with CWD = the fresh worktree. Best-effort by contract:
# report failures loudly on stderr, never block worktree creation.
#
# Ported from sales-collateral-agx, whose version encoded ITS sprint-measured
# provisioning set (mise + a DUAL bun install, root and web/). This repo's set is
# different, and items 3-4 below were measured the hard way:
#
#   1. mise toolchain (python 3.12, uv, just, prek, shellcheck/shfmt/bats)
#   2. `uv sync --all-groups` — there is no bun and no web/ subproject here
#   3. .env — GITIGNORED, so `git worktree add` does NOT carry it over. A fresh
#      worktree therefore has no R2 / HF / RunPod credentials, and anything that
#      touches the DVC remote or the MLflow artifact store fails with an auth
#      error that looks like a code defect rather than a missing file. mise reads
#      .env via `_.file = ".env"`, so its absence is silent until something needs
#      a key.
#   4. justfile.local — also GITIGNORED (CLAUDE.local.md: "publish sequence:
#      just stack > just ship-it"), so a fresh worktree silently lacks the
#      personal `stack` / `commit-msg` recipes `import?` pulls in. MEASURED
#      2026-08-16: an agent mid-session reached for `git commit -F msg.txt`
#      here despite the sequence being written down, because the loaded
#      one-line memory hook named only `ship-it` and the live gate — the
#      absence of `stack` was never surfaced, it was just quietly not there.
#      Copying it here closes the same gap `.env` closes for credentials:
#      the worktree should provision the same commit path the owning repo has,
#      not a degraded one an agent has to notice by its absence.
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

# Both copies below come from the worktree's OWNING repository, resolved once
# via git rather than assumed, so this keeps working if the checkout ever moves.
main_root="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
main_root="${main_root%/.git}"

# Credentials: copy from the owning repository (see item 3 above).
if [ -f .env ]; then
  log ".env already present — leaving it alone"
else
  if [ -n "$main_root" ] && [ -f "$main_root/.env" ]; then
    cp "$main_root/.env" .env && log "copied .env from $main_root"
  else
    log "WARNING: no .env found at ${main_root:-<unresolved>} — R2/HF/RunPod-dependent commands will fail until one is provided"
  fi
fi

# Personal publish-sequence recipes: copy from the owning repository (see item 4
# above), exactly the same way as .env.
if [ -f justfile.local ]; then
  log "justfile.local already present — leaving it alone"
else
  if [ -n "$main_root" ] && [ -f "$main_root/justfile.local" ]; then
    cp "$main_root/justfile.local" justfile.local && log "copied justfile.local from $main_root"
  else
    log "WARNING: no justfile.local found at ${main_root:-<unresolved>} — 'just stack' will be unavailable until one is provided"
  fi
fi

log "ceremony complete"
