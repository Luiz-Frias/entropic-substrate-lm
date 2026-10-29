#!/usr/bin/env bash
# =============================================================================
# session-start-snapshot.sh - Print the session-state snapshot at SessionStart
# =============================================================================
# Event: SessionStart. Per the official docs (code.claude.com/docs/en/hooks,
# measured 2026-09-09), SessionStart matchers are startup|resume|clear|
# compact|fork, and only SessionStart/UserPromptSubmit/UserPromptExpansion/
# PostModelSwitch have their plain-text stdout added to session context — a
# PostCompact hook cannot inject context at all. Consequence: the
# post-compaction snapshot rides THIS template under matcher `compact`.
# Ruling B-7's STAGED UNWIRED status held only through p7d: p7e's
# `helmsman init state` (src/init/adapters/state-install.ts, FOR-1354)
# now wires a PROJECT-SCOPE copy of this template into that project's own
# tracked <project>/.claude/settings.json via an additive merge — never
# helmsman's own hooks.json/settings.json, which stay unwired by this
# template (acceptance item 4; comment-only change, no executable line
# touched).
#
# Contract: session-state-hook-templates (FOR-1354). Always exits 0. stdout
# is EITHER the reader's snapshot text OR exactly one line:
#   [session-state] unavailable: <reason>
# never both, and never partial snapshot text plus the unavailable line —
# the reader's output is captured and emitted atomically at the end.
#
# Unavailability is decided from OBSERVABLE PRECONDITIONS (bun resolvable,
# reader file present) plus the reader's EXIT STATUS — never from stdout
# being empty (the hermetic bats bun stub exits 0 with empty stdout, which
# an emptiness-driven branch could not distinguish from success).
#
# Env parameters (derived defaults, never literals):
#   HELMSMAN_ROOT           resolution order: env -> ${CLAUDE_PLUGIN_ROOT} -> script-dir
#                           walk-up (only when a reader is found there) -> unavailable
#                           (FOR-1354 p7e; no resolved path is ever printed)
#   HELMSMAN_STATE_LEDGER   resolution order (DN-66): env -> the project's
#                           own .claude/state/ dir (only when
#                           CLAUDE_PROJECT_DIR is set and that directory
#                           exists) -> the XDG default
#                           ${XDG_STATE_HOME:-$HOME/.local/state}/helmsman/session-state.jsonl
#   HELMSMAN_PLANS_REGISTRY default $HOME/.claude/plans/_registry.yaml
#   HELMSMAN_RUNS_DIR       default $HOME/.claude/projects
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

if ! command -v bun >/dev/null 2>&1; then
  printf '[session-state] unavailable: bun not found on PATH\n'
  exit 0
fi

if [ ! -f "$reader" ]; then
  printf '[session-state] unavailable: reader not found under %s\n' "$root_source"
  exit 0
fi

out="$(bun "$reader" snapshot 2>/dev/null)"
rc=$?

if [ "$rc" -eq 0 ]; then
  printf '%s\n' "$out"
else
  printf '[session-state] unavailable: reader exited %d\n' "$rc"
fi

exit 0
