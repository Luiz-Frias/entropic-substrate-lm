#!/usr/bin/env bash
# =============================================================================
# workflow-run-stamp.sh - Append run.launched to the session-state ledger
# =============================================================================
# Event: PostToolUse (matcher Workflow). Fires after every Workflow tool
# call so an in-flight run shows up in the session-state snapshot without
# anyone remembering to record it. Prints NOTHING on the success path, and
# exactly one line on every other path:
#   [session-state] unavailable: <reason>
# Always exits 0. Ruling B-7's STAGED UNWIRED status held only through
# p7d: p7e's `helmsman init state` (src/init/adapters/state-install.ts,
# FOR-1354) now wires a PROJECT-SCOPE copy of this template into that
# project's own tracked <project>/.claude/settings.json via an additive
# merge — never helmsman's own hooks.json/settings.json, which stay
# unwired by this template (acceptance item 4; comment-only change, no
# executable line touched).
#
# Contract: session-state-hook-templates (FOR-1354, DN-58). Never
# fabricates an event field.
#
# tool_response source order (DN-58 — the object shape is MEASURED from a
# live session transcript; the labeled-text string shape below is an
# UNMEASURED, synthetic-shaped defensive fallback, not an observed hook
# payload):
#   - object  -> runId = .runId, taskId = .taskId (structured fields).
#   - string, or a labeled line embedded anywhere in the object's own text
#     (e.g. inside .summary) -> scraped via `Run ID: <id>` / `Task ID:
#     <id>`.
#   - both sources present and they DISAGREE on runId or taskId -> the
#     unavailable path (never pick one). Absence is never disagreement: a
#     source only counts once it yields a non-empty value, and the normal
#     object response (no labels anywhere in its text) stamps from the
#     structured fields alone.
# `script` is basename(tool_input.scriptPath), falling back to
# basename(tool_response.scriptPath) when tool_input carries none; this is
# the only fallback-ordered field and is never compared for disagreement.
# `phase` is tool_input.args.phase.id (args may be a JSON object or a JSON
# string). A hook input whose tool_name is not "Workflow", that yields no
# run id, no script, or no phase id, takes the unavailable path.
#
# Resume re-stamping (DN-58): a resumed run returns the same run id; this
# template stamps on EVERY Workflow result carrying a run id, append-only,
# with no ledger read-back. De-duplication is fold semantics behind the
# frozen reader (session-state.ts fold() keys `runs` by runId), never a
# template-side check.
#
# Env parameters: same four HELMSMAN_* as post-compact-stamp.sh /
# session-start-snapshot.sh, same derived defaults.
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

tool_name="$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null)"
[ "$tool_name" = "Workflow" ] || unavailable "not a Workflow tool_name"

# --- structured source (tool_response as an object) ------------------------
response_type="$(printf '%s' "$input" | jq -r '.tool_response | type' 2>/dev/null)"
struct_run_id=""
struct_task_id=""
struct_script_path=""
if [ "$response_type" = "object" ]; then
  struct_run_id="$(printf '%s' "$input" | jq -r '.tool_response.runId // empty' 2>/dev/null)"
  struct_task_id="$(printf '%s' "$input" | jq -r '.tool_response.taskId // empty' 2>/dev/null)"
  struct_script_path="$(printf '%s' "$input" | jq -r '.tool_response.scriptPath // empty' 2>/dev/null)"
fi

# --- labeled-text scrape (string response, or labels embedded in an
# object's own text, e.g. inside .summary) -----------------------------
response_text="$(printf '%s' "$input" | jq -r '.tool_response | if type == "string" then . else tostring end' 2>/dev/null)"
scraped_run_id="$(printf '%s' "$response_text" | grep -oE 'Run ID: [A-Za-z0-9_-]+' 2>/dev/null | head -1 | sed -E 's/^Run ID: //')"
scraped_task_id="$(printf '%s' "$response_text" | grep -oE 'Task ID: [A-Za-z0-9_-]+' 2>/dev/null | head -1 | sed -E 's/^Task ID: //')"

# --- resolve runId (structured vs scraped; disagreement is fatal) ---------
run_id=""
if [ -n "$struct_run_id" ] && [ -n "$scraped_run_id" ]; then
  [ "$struct_run_id" = "$scraped_run_id" ] || unavailable "disagreeing runId sources"
  run_id="$struct_run_id"
elif [ -n "$struct_run_id" ]; then
  run_id="$struct_run_id"
elif [ -n "$scraped_run_id" ]; then
  run_id="$scraped_run_id"
fi
[ -n "$run_id" ] || unavailable "no run id derivable"

# --- resolve taskId (optional; disagreement is still fatal) --------------
task_id=""
if [ -n "$struct_task_id" ] && [ -n "$scraped_task_id" ]; then
  [ "$struct_task_id" = "$scraped_task_id" ] || unavailable "disagreeing taskId sources"
  task_id="$struct_task_id"
elif [ -n "$struct_task_id" ]; then
  task_id="$struct_task_id"
elif [ -n "$scraped_task_id" ]; then
  task_id="$scraped_task_id"
fi

# --- resolve script (fallback-ordered; never compared for disagreement) --
tool_input_script_path="$(printf '%s' "$input" | jq -r '.tool_input.scriptPath // empty' 2>/dev/null)"
script_path="$tool_input_script_path"
[ -n "$script_path" ] || script_path="$struct_script_path"
[ -n "$script_path" ] || unavailable "missing script"
script="$(basename -- "$script_path")"

# --- resolve phase (tool_input.args.phase.id; args is object or JSON string)
args_type="$(printf '%s' "$input" | jq -r '.tool_input.args | type' 2>/dev/null)"
phase=""
if [ "$args_type" = "object" ]; then
  phase="$(printf '%s' "$input" | jq -r '.tool_input.args.phase.id // empty' 2>/dev/null)"
elif [ "$args_type" = "string" ]; then
  args_str="$(printf '%s' "$input" | jq -r '.tool_input.args' 2>/dev/null)"
  phase="$(printf '%s' "$args_str" | jq -r '.phase.id // empty' 2>/dev/null)"
fi
[ -n "$phase" ] || unavailable "missing phase id"

at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
startedAt="$at"
repo_toplevel="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -n "$repo_toplevel" ]; then
  repo="$(basename "$repo_toplevel")"
else
  repo="global"
fi

if [ -n "$task_id" ]; then
  event="$(jq -cn \
    --arg at "$at" --arg repo "$repo" --arg runId "$run_id" \
    --arg script "$script" --arg phase "$phase" --arg startedAt "$startedAt" \
    --arg taskId "$task_id" \
    '{event: "run.launched", at: $at, repo: $repo, runId: $runId, script: $script, phase: $phase, startedAt: $startedAt, taskId: $taskId}')"
else
  event="$(jq -cn \
    --arg at "$at" --arg repo "$repo" --arg runId "$run_id" \
    --arg script "$script" --arg phase "$phase" --arg startedAt "$startedAt" \
    '{event: "run.launched", at: $at, repo: $repo, runId: $runId, script: $script, phase: $phase, startedAt: $startedAt}')"
fi

bun "$reader" append "$event" >/dev/null 2>&1
rc=$?

if [ "$rc" -ne 0 ]; then
  unavailable "append failed (exit $rc)"
fi

exit 0
