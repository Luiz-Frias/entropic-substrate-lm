#!/usr/bin/env bash
# AGX worktree post-create ceremony — invoked by the global WorktreeCreate hook's
# project-extension seam with CWD = the fresh worktree. Best-effort by contract:
# report failures loudly on stderr, never block worktree creation.
#
# Encodes the sprint-measured provisioning set: mise toolchain + the DUAL bun
# install (root AND web/) — a fresh worktree missing web/node_modules reds five
# web tests + lint in a way that mimics a base defect (2026-07 hex-sprint lesson).
set -u

log() { printf 'worktree-post-create(agx): %s\n' "$*" >&2; }

if command -v mise >/dev/null 2>&1; then
  log "mise trust"
  mise trust >&2 2>&1 || log "mise trust FAILED (non-blocking)"
  log "mise install"
  mise install >&2 2>&1 || log "mise install FAILED (non-blocking)"
else
  log "mise not on PATH — skipping toolchain install"
fi

if command -v bun >/dev/null 2>&1; then
  log "bun install (root)"
  bun install >&2 2>&1 || log "root bun install FAILED (non-blocking)"
  if [ -d web ]; then
    log "bun install (web/)"
    (cd web && bun install >&2 2>&1) || log "web/ bun install FAILED (non-blocking)"
  fi
else
  log "bun not on PATH — skipping dependency install"
fi

log "ceremony complete"
