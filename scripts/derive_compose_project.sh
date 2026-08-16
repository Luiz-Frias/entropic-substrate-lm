#!/usr/bin/env bash
# derive_compose_project.sh — the live-gate compose PROJECT identity (p5).
#
# Extracted out of run_container_test.sh so this derivation's own RUNTIME
# behaviour — same worktree, same second, two DIFFERENT project names — is
# directly testable by invoking this file as a subprocess twice
# (tests/test_derive_compose_project.py), rather than only provable by
# grepping run_container_test.sh's source text. A text assertion can show a
# token is present; it cannot show two invocations actually diverge.
#
# Sets two variables when SOURCED (`source scripts/derive_compose_project.sh`
# — this is how run_container_test.sh uses it):
#
#   PROJECT_FAMILY   stable per WORKTREE (basename + full-toplevel-path
#                     hash) — IDENTICAL across every invocation from the same
#                     worktree, concurrent or sequential. This is the
#                     identity tests/helpers.py's assert_no_foreign_mlflow_stack
#                     uses to recognise "another run of MINE" (same family,
#                     tolerate briefly) versus a genuinely foreign worktree's
#                     stack (different family, refuse immediately) — defects
#                     #2/#3.
#   PROJECT           PROJECT_FAMILY plus PER-RUN entropy (this process's PID
#                     and start-of-derivation epoch second) — unique to THIS
#                     invocation, so two concurrent `just container-test`
#                     runs from the same worktree can never derive the same
#                     compose project, and therefore never the same
#                     container names — defect #1.
#
# When EXECUTED directly (`bash scripts/derive_compose_project.sh`), also
# PRINTS both as `KEY=value` lines on stdout — the mechanism the acceptance
# test uses to observe two REAL, separate OS processes' derivations.
set -euo pipefail

# git rev-parse --show-toplevel, not $PWD: this resolves to the CURRENT
# worktree's own root regardless of nesting. An agent worktree created
# INSIDE another worktree (e.g.
# .../observability-sprint/.claude/worktrees/wf_*) still gets its own
# distinct toplevel here — VERIFIED EMPIRICALLY (2026-08-14/15, two real
# `git worktree add`-created worktrees nested under this one): each
# resolved `git rev-parse --show-toplevel` to its OWN full path, never the
# parent's, so PROJECT_FAMILY below is already distinct per nested worktree
# on the NAME side.
#
# What this derivation cannot catch is defect #4's OTHER half: a session
# whose cwd is not the worktree it believes it is in derives a name that is
# FAITHFUL to the path but the path itself is wrong — entropy (below)
# attaches to whatever path a session happens to be standing in, and no
# amount of it fixes standing in the wrong place. That is caught one layer
# up, in run_container_test.sh's LIVE_GATE_EXPECTED_TOPLEVEL guard, BEFORE
# this script is ever sourced.
TOPLEVEL="$(git rev-parse --show-toplevel)"
WORKTREE_NAME="$(basename "$TOPLEVEL")"
SANITIZED_WORKTREE="$(printf '%s' "$WORKTREE_NAME" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_-' '-')"
# Hash the FULL path, not just the basename: the Round C guard-teeth recipe
# mandates WT="$(mktemp -d)/wt", so every reviewer's worktree basename is
# literally "wt" and they would all resolve to one shared family otherwise —
# restoring the container-fusion hazard this derivation exists to remove.
# cksum is POSIX and always present; CRC32 over a handful of concurrent
# paths has negligible collision risk, and the readable basename is
# retained purely so `podman ps` stays human-legible.
PATH_HASH="$(printf '%s' "$TOPLEVEL" | cksum | cut -d' ' -f1)"
PROJECT_FAMILY="entropic-substrate-lm-mlflow-test-${SANITIZED_WORKTREE}-${PATH_HASH}"

# PER-RUN entropy — defect #1 (MEASURED 2026-08-14/15): two
# `run_container_test.sh` processes were observed sharing the SAME cwd
# (`lsof -p <pid> -d cwd`), and a fresh phase-1-shaped stack appeared under
# a worktree's exact project name that its own agent never started. The
# per-worktree PATH_HASH above closed CROSS-worktree fusion; it reopened the
# exact same hazard for two concurrent runs FROM the same worktree, because
# both derive byte-identical PROJECT_FAMILY and — before this fix —
# byte-identical PROJECT too.
#
# A RETRACTION of the "NOT $$ either" reasoning this derivation used to
# carry (run_container_test.sh, pre-p5): that comment was correct for a
# design where the project name had to be RE-DERIVED, independently, in a
# DIFFERENT shell invocation than the one that ran `up` — a PID captured
# during `up` is meaningless to a `down` invoked later from a fresh process.
# That is not how this repo's live gate actually works, and was not even
# when that comment was written: PROJECT is computed ONCE per
# run_container_test.sh invocation and EXPORTED as MLFLOW_COMPOSE_PROJECT —
# the EXIT trap (same shell), pytest's phase-2/3 fixtures, and
# tests/test_mlflow_live.py's own `podman compose -p` calls all READ that
# exported value; nothing re-derives it from scratch. So this process's own
# PID is safe to fold in.
#
# PID + start-of-derivation epoch second, not either alone: two concurrent
# invocations landing in the same wall-clock second is the exact near-miss
# this fix exists to rule out — acceptance proves this directly, by
# comparing two derivations issued in the same second and showing they
# still differ (via PID) even when the timestamp does not.
PROJECT="${PROJECT_FAMILY}-$$-$(date +%s)"

# Print only on DIRECT execution — a SOURCED invocation (run_container_test.sh's
# normal use) must not pollute the caller's stdout with these lines; it gets
# the shell variables above instead. `${BASH_SOURCE[0]}` is this file's own
# path regardless of how it was reached; it equals `$0` only when this file
# IS the process's entry point.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  printf 'PROJECT_FAMILY=%s\n' "$PROJECT_FAMILY"
  printf 'PROJECT=%s\n' "$PROJECT"
fi
