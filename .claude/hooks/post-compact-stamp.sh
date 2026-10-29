#!/usr/bin/env bash
# =============================================================================
# post-compact-stamp.sh - Append session.compacted to the session-state ledger
# =============================================================================
# Event: PostCompact (matchers manual|auto). PostCompact cannot inject
# session context (only SessionStart/UserPromptSubmit/UserPromptExpansion/
# PostModelSwitch can — see session-start-snapshot.sh's header), so this
# hook's only honest job is to STAMP the ledger: it prints NOTHING on the
# success path, and exactly one line on the failure path:
#   [session-state] unavailable: <reason>
# Always exits 0. Ruling B-7's STAGED UNWIRED status held only through
# p7d: p7e's `helmsman init state` (src/init/adapters/state-install.ts,
# FOR-1354) now wires a PROJECT-SCOPE copy of this template into that
# project's own tracked <project>/.claude/settings.json via an additive
# merge — never helmsman's own hooks.json/settings.json, which stay
# unwired by this template (acceptance item 4; comment-only change, no
# executable line touched).
#
# Contract: session-state-hook-templates (FOR-1354). Never fabricates an
# event field: `trigger` is read from the hook's stdin JSON via jq (a
# missing or invalid trigger takes the unavailable path — never a defaulted
# "auto"); `at` is `date -u` in RFC 3339 form; `repo` is the git toplevel
# basename, or "global" when not inside a git repo (both within the
# hermetic tool floor, so reading the real values costs nothing).
#
# Env parameters: same four HELMSMAN_* as session-start-snapshot.sh, same
# derived defaults.
#
# Hermetic-bats tool floor (setup_minbin): bash sh cat jq git grep sed head
# dirname basename sort uname mkdir touch rm chmod tr ls cp date sleep ln
# awk, plus the stubbed bun. No realpath, readlink, mktemp, find, wc, cut,
# env, or timeout.
# =============================================================================
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# HELMSMAN_ROOT resolution order (FOR-1354 p7e amendment): the env var
# wins outright; else ${CLAUDE_PLUGIN_ROOT}; else the script-dir walk-up —
# taken ONLY when a src/state/adapters/state-cli.ts reader is actually
# found there (a project-scope COPY of this template resolves the
# walk-up to the PARENT of the project, which has no reader there — this
# guard is what keeps that case honest instead of silently pointing at a
# foreign root). Absent all three, the unavailable line NAMES the three
# sources tried and prints no resolved path (this is SessionStart stdout,
# injected into session context).
root_source=""
if [ -n "${HELMSMAN_ROOT:-}" ]; then
  root_source="HELMSMAN_ROOT"
elif [ -n "${CLAUDE_PLUGIN_ROOT:-}" ]; then
  HELMSMAN_ROOT="$CLAUDE_PLUGIN_ROOT"
  root_source="CLAUDE_PLUGIN_ROOT"
else
  candidate_root="$script_dir/../../.."
  if [ -f "$candidate_root/src/state/adapters/state-cli.ts" ]; then
    HELMSMAN_ROOT="$candidate_root"
    root_source="script-dir"
  fi
fi

if [ -z "$root_source" ]; then
  printf '[session-state] unavailable: no root from HELMSMAN_ROOT env, CLAUDE_PLUGIN_ROOT, or script-dir reader probe\n'
  exit 0
fi

# Ledger resolution (DN-66): explicit env -> the project's own state dir when
# it exists (the marker `helmsman init state` leaves; per-project ledger,
# DN-62) -> the XDG user path. The project wire is therefore the bare
# "$CLAUDE_PROJECT_DIR"/.claude/hooks/<script> and carries no value.
if [ -z "${HELMSMAN_STATE_LEDGER:-}" ] && [ -n "${CLAUDE_PROJECT_DIR:-}" ] && [ -d "$CLAUDE_PROJECT_DIR/.claude/state" ]; then
  HELMSMAN_STATE_LEDGER="$CLAUDE_PROJECT_DIR/.claude/state/session-state.jsonl"
fi
HELMSMAN_STATE_LEDGER="${HELMSMAN_STATE_LEDGER:-${XDG_STATE_HOME:-$HOME/.local/state}/helmsman/session-state.jsonl}"
HELMSMAN_PLANS_REGISTRY="${HELMSMAN_PLANS_REGISTRY:-$HOME/.claude/plans/_registry.yaml}"
HELMSMAN_RUNS_DIR="${HELMSMAN_RUNS_DIR:-$HOME/.claude/projects}"
export HELMSMAN_ROOT HELMSMAN_STATE_LEDGER HELMSMAN_PLANS_REGISTRY HELMSMAN_RUNS_DIR

reader="$HELMSMAN_ROOT/src/state/adapters/state-cli.ts"

unavailable() {
  printf '[session-state] unavailable: %s\n' "$1"
  exit 0
}

command -v bun >/dev/null 2>&1 || unavailable "bun not found on PATH"
[ -f "$reader" ] || unavailable "reader not found under $root_source"
command -v jq >/dev/null 2>&1 || unavailable "jq not found on PATH"

input="$(cat)"
trigger="$(printf '%s' "$input" | jq -r '.trigger // empty' 2>/dev/null)"
if [ "$trigger" != "manual" ] && [ "$trigger" != "auto" ]; then
  unavailable "missing or invalid trigger in hook input"
fi

at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
repo_toplevel="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -n "$repo_toplevel" ]; then
  repo="$(basename "$repo_toplevel")"
else
  repo="global"
fi

event="$(jq -cn --arg trigger "$trigger" --arg at "$at" --arg repo "$repo" \
  '{event: "session.compacted", at: $at, repo: $repo, trigger: $trigger}')"

bun "$reader" append "$event" >/dev/null 2>&1
rc=$?

if [ "$rc" -ne 0 ]; then
  unavailable "append failed (exit $rc)"
fi

exit 0
