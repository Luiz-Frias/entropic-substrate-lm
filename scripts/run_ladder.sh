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
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

RESULTS_DIR=${RESULTS_DIR:-results}
PY=${LADDER_PYTHON:-python}
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
  "$PY" -m entropic_substrate.cli train --exp "$exp" "$@"
}

# Measurement path — fail LOUD: no defaults, no fallbacks, no "probably fine".
final_npb() {
  "$PY" - "${RESULTS_DIR}/$(rname "$1").jsonl" <<'PYEOF'
import json
import math
import sys

path = sys.argv[1]
last = None
with open(path, encoding="utf-8") as fh:
    for line in fh:
        rec = json.loads(line)
        if rec.get("event") == "eval":
            last = rec
if last is None:
    sys.exit(f"{path}: no eval record — run never reached an eval; refusing to gate on nothing")
npb = last["nats_per_byte"]
if not math.isfinite(npb):
    sys.exit(f"{path}: non-finite nats/byte {npb!r}")
print(npb)
PYEOF
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

# ----------------------------------------------------------------- E2 rung
say "=== E2: hypersphere-native backbone ==="
run_rung e2 "$@"
dvc_sync e2
E2_NPB=$(final_npb e2)
say "E2 gate: ${E2_NPB} nats/byte (E1 winner was ${FISH_NPB}); if E2 converged slower, see the triage table in docs/e-3-instruction.md"

# ----------------------------------------------------------------- E3 rung
say "=== E3: kappa-attention kernel ==="
run_rung e3 "$@"
dvc_sync e3
E3_NPB=$(final_npb e3)
say "E3 gate: ${E3_NPB} nats/byte vs E2 ${E2_NPB}"

say "per-head kappa divergence report (the e3-anchor / e3-vantage execution gate):"
"$PY" - "${RESULTS_DIR}/$(rname e3).jsonl" <<'PYEOF'
import json
import sys

last = None
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        rec = json.loads(line)
        if rec.get("event") == "eval":
            last = rec
if last is None or not last.get("kappa"):
    sys.exit("no kappa telemetry in the final eval — cannot report divergence")
spreads = []
for i, row in enumerate(last["kappa"]):
    spread = max(row) - min(row)
    spreads.append(spread)
    print(f"  layer {i}: {row}  (head spread {spread:.4f})")
print(f"  max within-layer head spread: {max(spreads):.4f}")
PYEOF

cat <<EOF

=============================== DECISION-NEEDED ===============================
The ladder STOPS here by design. Review the numbers above, then decide:

1) Anchored arms — execution gate is the per-head kappa divergence just
   printed. If it fires for you, launch each in its own pod (parallel to any
   E3 seed reruns):
     runpodctl pod create --image ghcr.io/luiz-frias/entropic-substrate-lm:latest \\
       # run command: python -m entropic_substrate.cli train --exp e3-anchor --device cuda
     runpodctl pod create --image ghcr.io/luiz-frias/entropic-substrate-lm:latest \\
       # run command: python -m entropic_substrate.cli train --exp e3-vantage --device cuda

2) Multi-seed: 3 seeds before believing any gap < 0.05 nats/byte
   (--seed N --results results/<exp>-sN.jsonl, sequentially on this pod).

3) E4 (docs/e-4-instruction.md) — precondition: E3 concluded WITH the 3-seed
   numbers. Do not start it from a single-seed readout.
===============================================================================
EOF

if [[ ${#DVC_FAILURES[@]} -gt 0 ]]; then
  say "PARTIAL RESULT: dvc sync failed for: ${DVC_FAILURES[*]} — pull them off this pod before it dies (runpodctl send / rsync)"
  exit 3
fi
say "ladder complete; all artifacts synced"
