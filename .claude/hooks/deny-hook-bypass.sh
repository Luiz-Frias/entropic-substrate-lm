#!/bin/sh
# PreToolUse(Bash) hook: refuse any git commit that skips the repo's hooks.
# FOR-1536 (MEASURED 2026-09-21): a lane fixer ran `git commit --no-verify`
# after the pre-commit unit suite stalled its commit twice, so gitlint never
# saw an 84-char title. Prose rules did not stop it; this channel can.
# Reads the tool call as JSON on stdin; prints a deny decision when the
# command carries a verification skip or re-points the hooks directory.
# The flag must belong to the SAME simple command as `commit`/`push`: a `;`,
# `&`, `|` or paren ends the match, and the hooks-path override must sit in a
# `git -c` form. MEASURED 2026-09-22: the first, broader form denied the
# conductor's own `git commit -F m; if [ -n "$(git status)" ]` and any command
# whose text merely MENTIONED the override key.
cmd="$(jq -r '.tool_input.command // ""' 2>/dev/null)"
[ -z "$cmd" ] && exit 0
if printf '%s' "$cmd" | grep -Eq '(^|[[:space:];&|(])git([[:space:]]+-[^[:space:];&|()]+)*[[:space:]]+(commit|push)([[:space:]]+[^[:space:];&|()]+)*[[:space:]]+(--no-verify|-n)([[:space:]]|$)|(^|[[:space:];&|(])git([[:space:]]+[^[:space:];&|()]+)*[[:space:]]+-c[[:space:]]*core\.hooksPath='; then
  printf '%s' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"FOR-1536: commits and pushes run with the repo hooks on; --no-verify / -n / core.hooksPath are refused. If a hook stalls, wait for it (the unit suite takes ~3 min) or report a DECISION."}}'
  exit 0
fi
exit 0
