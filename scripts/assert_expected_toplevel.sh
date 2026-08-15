#!/usr/bin/env bash
# assert_expected_toplevel.sh — the live-gate's identity guard (p5, defects
# #4 / #5-fix-1).
#
# scripts/derive_compose_project.sh's per-run entropy fixes a session
# deriving the WRONG NAME for the worktree it is genuinely standing in — it
# cannot fix a session standing in the WRONG WORKTREE altogether. Agent
# worktrees are created INSIDE another worktree
# (.../observability-sprint/.claude/worktrees/wf_*), so a stale cwd or a
# nested-worktree mix-up derives a project identity that is FAITHFUL to the
# path but the path itself is wrong — no amount of entropy catches that,
# because entropy attaches to whatever path a session happens to be
# standing in.
#
# EXTRACTED to its own file (rather than inlined in run_container_test.sh)
# for the SAME reason derive_compose_project.sh was: this makes the guard's
# own runtime behaviour — refuses on a mismatch, permits a match, permits an
# unset expectation — directly testable in isolation
# (tests/test_run_container_test_config.py), with NO dependency on .env,
# AWS credentials, or podman. A grep of run_container_test.sh's source text
# could prove a token is present; it could not prove the guard actually
# fires only when it should.
#
# OPTIONAL: an orchestrator that knows which worktree it launched this gate
# in can pin LIVE_GATE_EXPECTED_TOPLEVEL; a manual `just container-test` run
# has nothing external to pin against and this degrades to a no-op — the
# same best-effort posture every other guard in this script takes when it
# cannot check something (see assert_no_foreign_mlflow_stack's own probe
# tolerance in tests/helpers.py).
#
# DISTINCT EXIT CODE (3, not the generic preflight 1 run_container_test.sh's
# AWS/S3 checks use) so a caller scripting around this gate can tell "wrong
# worktree" apart from "missing credentials" without parsing stderr text.
#
# When SOURCED (run_container_test.sh's normal use), also leaves TOPLEVEL
# set for the caller's own identity header — one `git rev-parse` call, not
# two.
set -euo pipefail

TOPLEVEL="$(git rev-parse --show-toplevel)"
if [[ -n "${LIVE_GATE_EXPECTED_TOPLEVEL:-}" && "$TOPLEVEL" != "${LIVE_GATE_EXPECTED_TOPLEVEL}" ]]; then
  echo "[container-test] REFUSING: this session's toplevel ($TOPLEVEL) does not match LIVE_GATE_EXPECTED_TOPLEVEL (${LIVE_GATE_EXPECTED_TOPLEVEL}) — proceeding would derive a compose project identity for the WRONG worktree." >&2
  exit 3
fi
