#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Claude Code PostToolUse hook — auto-format + LOUD lint gate on edit.
#
# Ported from sales-collateral-agx (biome/dprint, TS toolchain) to this repo's
# Python toolchain. Same contract, same intent: run what the commit gate runs on
# the file just written, apply safe fixes, then FAIL LOUD (exit 2 -> stderr is fed
# back to the agent) if anything remains. Surfaces gate failures at EDIT time
# instead of at commit time.
#
# WHY prek RATHER THAN CALLING ruff DIRECTLY: prek executes the exact hook
# revisions pinned in .pre-commit-config.yaml, so "the same tools the commit gate
# runs" is exact rather than approximate -- there is no second invocation to drift
# out of sync. This hook READS that config and never writes it.
#
# The one deliberate exception is basedpyright: it is invoked through uv rather
# than prek so that it covers every path pyproject's [tool.basedpyright] include
# list claims, matching `just typecheck` (and therefore CI).
#
# Contract: receives the tool-call JSON on stdin; reads .tool_input.file_path.
#   exit 0 = clean   |   exit 2 = blocking (agent must fix before continuing)
# -----------------------------------------------------------------------------
set -uo pipefail

input="$(cat)"
file="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)"

[ -z "$file" ] && exit 0
[ -f "$file" ] || exit 0

# Skip writes outside this repo (plans, memory, scratchpad). The tools are scoped
# to the repo config, so an out-of-repo file yields "no files found" and would
# false-block the edit. Formatting those files is not this repo's concern.
#
# Resolve repo_root from the FILE's location, not the hook's own cwd. When the
# file lives in a linked worktree (.claude/worktrees/**), the hook's cwd is often
# the MAIN checkout -- resolving there yields the main checkout's root, and the
# tools' vcs-ignore handling then treats the worktree path as git-ignored (this
# repo's .gitignore excludes .claude/worktrees/), producing a false BLOCK.
# `git -C <file-dir> rev-parse --show-toplevel` always resolves to the git root
# that actually owns the file (worktree or main checkout).
abs_dir="$(cd "$(dirname "$file")" 2>/dev/null && pwd -P)" || exit 0
repo_root="$(git -C "$abs_dir" rev-parse --show-toplevel 2>/dev/null)" || exit 0
case "$abs_dir/" in
  "$repo_root"/*) ;; # inside the repo -> check it
  *) exit 0 ;;       # outside the repo -> skip silently
esac
# Run from repo_root so config + ignore-file resolution matches the git context
# that actually owns the file (see comment above).
cd "$repo_root" || exit 0

# prek matches its files:/exclude: patterns against REPO-RELATIVE paths, so an
# absolute path silently matches nothing and every hook no-ops into a false pass.
rel="${file#"$repo_root"/}"

# Run one prek hook against the file. A hook whose files:/types: filter excludes
# this path is a no-op that exits 0 -- an exclusion, never a block.
run_prek() {
  local hook=$1 out
  if ! out="$(prek run "$hook" --files "$rel" 2>&1)"; then
    printf 'prek:%s BLOCKED %s -- fix before continuing:\n%s\n' "$hook" "$rel" "$out" >&2
    return 1
  fi
  return 0
}

case "$rel" in
  *.py)
    # ruff-format first (rewrites), then ruff --fix --exit-non-zero-on-fix, which
    # deliberately fails when it changes something so the agent sees the edit.
    run_prek ruff-format || exit 2
    run_prek ruff || exit 2
    # Typecheck only the paths basedpyright is configured to own; anything else
    # would be analysed out of context and report noise.
    case "$rel" in
      src/* | scripts/*)
        if ! out="$(uv run basedpyright "$rel" 2>&1)"; then
          printf 'basedpyright BLOCKED %s -- fix before continuing:\n%s\n' "$rel" "$out" >&2
          exit 2
        fi
        ;;
    esac
    ;;
  *.md)
    run_prek dprint || exit 2
    run_prek markdownlint-cli2 || exit 2
    ;;
  *.sh)
    run_prek shfmt || exit 2
    run_prek shellcheck || exit 2
    ;;
  *.yaml | *.yml)
    run_prek yamllint || exit 2
    ;;
esac

exit 0
