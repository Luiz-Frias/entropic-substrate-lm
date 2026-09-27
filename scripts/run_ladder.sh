#!/usr/bin/env bash
# =============================================================================
# run_ladder.sh — sequential experiment ladder: E1 (both arms) → E2 → E3.
#
# Gates encode SANITY + SELECTION only: run finished, final eval present and
# finite, pick the E1 winner. Judgment points STOP LOUDLY with a
# DECISION-NEEDED block instead of deciding:
#   * an E1 winner that contradicts the fisher assumption pre-committed in the
#     e2/e3 configs stops the ladder before GPU hours are spent on it;
#   * everything after E3 — the anchored arms (e3-anchor / e3-vantage, whose
#     execution gate is the E3 per-head kappa readout) and E4 — is a human
#     go/no-go call by design (docs/e-3-instruction.md). The script emits the
#     launch commands; it never creates pods and never spends money on its own.
#
# Pod usage:    bash scripts/run_ladder.sh --device cuda
# Local smoke:  just ladder --tiny        (CPU, minutes; skips dvc sync)
# Extra args are passed through to every `train` invocation.
# LADDER_SKIP_DVC=1 skips artifact sync explicitly.
# LADDER_GPU_ID / LADDER_IMAGE set the GPU and image in the emitted (never
# executed) pod commands.
# =============================================================================
set -euo pipefail
umask 077

cd "$(dirname "$0")/.."

# Deliverable 10 (dreamy-questing-scroll.md, E-4 ruling): the dotted `python -m`
# entrypoint names are parameterized from ONE source, scripts/entrypoints.env,
# shared with the justfile and the worker allowlists in provider_launch.py /
# d1_runpod_adapter.py / h_probe_launch.py / d1_external_analysis_adapter.py
# (see patches/entrypoints-python.md). `set -u` (above) means a key missing
# from that file fails loud here, not silently as an empty string.
source scripts/entrypoints.env

RESULTS_DIR=${RESULTS_DIR:-results}
PY=${LADDER_PYTHON:-python}
GPU_ID=${LADDER_GPU_ID:-NVIDIA H100 80GB HBM3}
IMAGE=${LADDER_IMAGE:-ghcr.io/luiz-frias/entropic-substrate-lm:latest}
# DVC's current DiskCache dependency has no patched release for its local
# pickle-cache advisory. Restrict the site cache to this pod user before DVC
# opens it; chmod also repairs a directory persisted from an older image.
DVC_SITE_CACHE_DIR=${DVC_SITE_CACHE_DIR:-.cache/dvc-site-cache}
export DVC_SITE_CACHE_DIR
install -d -m 0700 "$DVC_SITE_CACHE_DIR"
chmod 0700 "$DVC_SITE_CACHE_DIR"
# The .dvc pointers are md5 addresses into the remote cache; without them the
# pushed objects are unreachable. They are pod-local files, so they are also
# collected here to ride along with any results retrieval.
POINTER_MANIFEST="${RESULTS_DIR}/dvc-pointers.txt"
declare -a DVC_FAILURES=()

SUF=""
for arg in "$@"; do
  [[ "$arg" == "--tiny" ]] && SUF="-tiny"
done

say() { printf '\n[ladder] %s\n' "$*"; }

rname() { printf '%s%s' "$1" "$SUF"; }

run_rung() {
  local exp=$1
  shift
  say "rung ${exp}: training..."
  "$PY" -m "$ENTRYPOINT_CLI" train --exp "$exp" "$@"
}

# Measurement path — fail LOUD: no defaults, no fallbacks, no "probably fine".
# Admissibility lives in entropic_substrate.telemetry, shared with the CLI
# preflight; an inline copy here is how the two came to disagree once already.
final_npb() {
  "$PY" -m "$ENTRYPOINT_TELEMETRY" final-eval \
    "${RESULTS_DIR}/$(rname "$1").jsonl" --metric nats_per_byte
}

dvc_sync() {
  local exp
  exp=$(rname "$1")
  if [[ "${LADDER_SKIP_DVC:-0}" == "1" || -n "$SUF" ]]; then
    say "dvc sync skipped for ${exp} (tiny run or LADDER_SKIP_DVC=1)"
    return 0
  fi
  say "dvc: pushing ${exp} artifacts..."
  local failed=0
  dvc add "models/${exp}-final.pt" "${RESULTS_DIR}/${exp}.jsonl" || failed=1
  dvc push || failed=1
  if [[ "$failed" -eq 1 ]]; then
    say "WARNING: dvc sync FAILED for ${exp} — artifacts are pod-local only"
    DVC_FAILURES+=("$exp")
    return 0
  fi
  # Pointer capture: `dvc add` writes the .dvc files on the POD, and nothing
  # here commits them to git. Echo them into the log and append them to a
  # manifest so the md5 addresses survive the pod even if the files don't.
  local pointer missing=0
  for pointer in "models/${exp}-final.pt.dvc" "${RESULTS_DIR}/${exp}.jsonl.dvc"; do
    if [[ ! -f "$pointer" ]]; then
      say "WARNING: ${pointer} missing after a successful dvc add — those objects are unaddressable"
      missing=1
      continue
    fi
    printf '\n----- %s -----\n' "$pointer"
    cat "$pointer"
    {
      printf '\n----- %s -----\n' "$pointer"
      cat "$pointer"
    } >>"$POINTER_MANIFEST"
  done
  if [[ "$missing" -eq 1 ]]; then
    DVC_FAILURES+=("$exp")
  fi
}

# ----------------------------------------------------------------- E1 rung
say "=== E1: substrate init swap (both arms) ==="
run_rung e1-random "$@"
dvc_sync e1-random
run_rung e1-fisher "$@"
dvc_sync e1-fisher

RAND_NPB=$(final_npb e1-random)
FISH_NPB=$(final_npb e1-fisher)
say "E1 gate: e1-random ${RAND_NPB} vs e1-fisher ${FISH_NPB} nats/byte"

WINNER=$("$PY" -c "import sys; r, f = float(sys.argv[1]), float(sys.argv[2]); print('e1-fisher' if f <= r else 'e1-random')" "$RAND_NPB" "$FISH_NPB")
say "E1 winner: ${WINNER}"

if [[ "$WINNER" != "e1-fisher" ]]; then
  cat <<EOF

=============================== DECISION-NEEDED ===============================
E1 winner is ${WINNER}, contradicting the fisher_sqrt assumption pre-committed
in configs/e2.yaml and configs/e3.yaml (see the ASSUMPTION note there).
Stopping BEFORE E2/E3 so no GPU hours are spent on a falsified premise.
Decide: keep fisher anyway (rerun with LADDER_ACCEPT_FISHER=1), or repoint the
e2/e3 configs at the winner and rerun the ladder.
===============================================================================
EOF
  if [[ "${LADDER_ACCEPT_FISHER:-0}" != "1" ]]; then
    exit 2
  fi
  say "LADDER_ACCEPT_FISHER=1 set — continuing on the pre-committed fisher arm"
fi

# The E2/E3 configs are pinned to the fisher init, so with LADDER_ACCEPT_FISHER
# the winner and the arm downstream rungs build on are DIFFERENT numbers; the
# gate line must not conflate them.
WINNER_NPB=$FISH_NPB
FISHER_NOTE=""
if [[ "$WINNER" != "e1-fisher" ]]; then
  WINNER_NPB=$RAND_NPB
  FISHER_NOTE=" [E2/E3 still build on the fisher arm: ${FISH_NPB}]"
fi

# ----------------------------------------------------------------- E2 rung
say "=== E2: hypersphere-native backbone ==="
run_rung e2 "$@"
dvc_sync e2
E2_NPB=$(final_npb e2)
say "E2 gate: ${E2_NPB} nats/byte vs E1 winner ${WINNER} ${WINNER_NPB}${FISHER_NOTE}; if E2 converged slower, see the triage table in docs/e-3-instruction.md"

# ----------------------------------------------------------------- E3 rung
say "=== E3: kappa-attention kernel ==="
run_rung e3 "$@"
dvc_sync e3
E3_NPB=$(final_npb e3)
say "E3 gate: ${E3_NPB} nats/byte vs E2 ${E2_NPB}"

say "per-head kappa divergence report (the e3-anchor / e3-vantage execution gate):"
"$PY" -m "$ENTRYPOINT_TELEMETRY" kappa-report "${RESULTS_DIR}/$(rname e3).jsonl"

cat <<EOF

=============================== DECISION-NEEDED ===============================
The ladder STOPS here by design. Review the numbers above, then decide:

1) Anchored arms — execution gate is the per-head kappa divergence just
   printed. If it fires for you, run these to launch each in its own pod
   (parallel to any E3 seed reruns). They are PRINTED, never executed here:

     runpodctl pod create --name e3-anchor --image ${IMAGE} \\
       --gpu-id "${GPU_ID}" \\
       --docker-args "python -m ${ENTRYPOINT_CLI} train --exp e3-anchor --device cuda"

     runpodctl pod create --name e3-vantage --image ${IMAGE} \\
       --gpu-id "${GPU_ID}" \\
       --docker-args "python -m ${ENTRYPOINT_CLI} train --exp e3-vantage --device cuda"

   NOTE: each arm's CLI preflight requires results/e3.jsonl ON ITS OWN POD —
   transfer this pod's copy first (runpodctl send ${RESULTS_DIR}/$(rname e3).jsonl
   here, runpodctl receive <code> there) so the gate evaluates the real E3
   readout. Do NOT reach for ENTROPIC_ALLOW_UNGATED=1 to skip the transfer;
   that override exists for deliberately ungated runs, not for convenience.

2) Multi-seed: 3 seeds before believing any gap < 0.05 nats/byte
   (--seed N --results results/<exp>-sN.jsonl, sequentially on this pod).

3) E4 (docs/e-4-instruction.md) — precondition: E3 concluded WITH the 3-seed
   numbers. Do not start it from a single-seed readout.

4) Artifact recovery BEFORE terminating this pod: the .dvc pointers printed
   above are the only route back to the pushed cache objects, and they live
   only here. They are collected in ${POINTER_MANIFEST} (absent on --tiny /
   LADDER_SKIP_DVC runs) — pull it off (runpodctl send) and commit the
   pointers to git, or the pushed objects are orphaned.
===============================================================================
EOF

if [[ ${#DVC_FAILURES[@]} -gt 0 ]]; then
  say "PARTIAL RESULT: dvc sync failed for: ${DVC_FAILURES[*]} — pull them off this pod before it dies (runpodctl send / rsync)"
  exit 3
fi
say "ladder complete; all artifacts synced"
