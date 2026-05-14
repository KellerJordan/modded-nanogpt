#!/usr/bin/env bash
# PR-only continuation sweep. Used after the interleaved sweep is stopped early.
# Runs N additional xsa-gated-layers runs and parks logs in logs/runpod_2026-05-07/pr/.

set -uo pipefail
cd /workspace/modded-nanogpt

: "${HF_TOKEN:?must export HF_TOKEN before running}"
SWEEPLOG=/tmp/sweep.log
PR_DIR=logs/runpod_2026-05-07/pr

N="${1:-5}"

run_one() {
  local idx="$1"
  echo "=== [pr #$idx] checkout xsa-gated-layers ===" | tee -a "$SWEEPLOG"
  git checkout -q xsa-gated-layers
  local before
  before=$(ls logs/*.txt 2>/dev/null | sort -u)
  echo "=== [pr #$idx] starting torchrun at $(date -Is) ===" | tee -a "$SWEEPLOG"
  local t0=$(date +%s)
  ./run.sh >>"$SWEEPLOG" 2>&1
  local rc=$?
  local t1=$(date +%s)
  local newlog
  newlog=$(comm -13 <(echo "$before") <(ls logs/*.txt 2>/dev/null | sort -u) | head -1)
  if [ -z "$newlog" ]; then
    echo "=== [pr #$idx] FAILED rc=$rc at $(date -Is) (no new log file)" | tee -a "$SWEEPLOG"
    return 1
  fi
  mv "$newlog" "$PR_DIR/"
  local moved="$PR_DIR/$(basename "$newlog")"
  local final_line
  final_line=$(grep -E "step:[0-9]+/[0-9]+ val_loss:" "$moved" | tail -1)
  echo "=== [pr #$idx] DONE rc=$rc wall=$((t1-t0))s -> $moved" | tee -a "$SWEEPLOG"
  echo "    $final_line" | tee -a "$SWEEPLOG"
}

# Find next available PR index by counting existing PR logs + 1.
existing_pr=$(ls "$PR_DIR" 2>/dev/null | wc -l)
for i in $(seq 1 "$N"); do
  run_one $((existing_pr + i)) || echo "(continuing)" | tee -a "$SWEEPLOG"
done

echo "=== PR-ONLY CONTINUATION COMPLETE at $(date -Is) ===" | tee -a "$SWEEPLOG"
ls "$PR_DIR" | wc -l | xargs -I{} echo "PR logs total: {}" | tee -a "$SWEEPLOG"
