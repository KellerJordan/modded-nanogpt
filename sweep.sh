#!/usr/bin/env bash
# Interleaved sweep: alternates xsa-gated-layers and master to reach 10 runs each.
# - The smoke run already produced PR run #1 (in logs/runpod_2026-05-07/pr/).
# - We start with master, then alternate.
# - Per-run logs are emitted as logs/{uuid}.txt by train_gpt.py and moved into
#   logs/runpod_2026-05-07/{pr,master}/ after each run.
# - Sweep log: /tmp/sweep.log . Per-run wall-times also tracked there.

set -uo pipefail
cd /workspace/modded-nanogpt

: "${HF_TOKEN:?must export HF_TOKEN before running}"
SWEEPLOG=/tmp/sweep.log
PR_DIR=logs/runpod_2026-05-07/pr
MASTER_DIR=logs/runpod_2026-05-07/master

run_one() {
  local label="$1"           # "pr" or "master"
  local target_branch="$2"
  local idx="$3"

  echo "=== [$label #$idx] checkout $target_branch ===" | tee -a "$SWEEPLOG"
  git checkout -q "$target_branch"

  # snapshot existing log uuids so we can find the new one
  local before
  before=$(ls logs/*.txt 2>/dev/null | sort -u)

  echo "=== [$label #$idx] starting torchrun at $(date -Is) ===" | tee -a "$SWEEPLOG"
  local t0=$(date +%s)
  # run.sh = torchrun --standalone --nproc_per_node=8 train_gpt.py
  ./run.sh >>"$SWEEPLOG" 2>&1
  local rc=$?
  local t1=$(date +%s)

  local newlog
  newlog=$(comm -13 <(echo "$before") <(ls logs/*.txt 2>/dev/null | sort -u) | head -1)

  if [ -z "$newlog" ]; then
    echo "=== [$label #$idx] FAILED rc=$rc at $(date -Is) (no new log file)" | tee -a "$SWEEPLOG"
    return 1
  fi

  local dest
  if [ "$label" = "pr" ]; then dest="$PR_DIR/"; else dest="$MASTER_DIR/"; fi
  mv "$newlog" "$dest"

  local moved
  moved="$dest$(basename "$newlog")"
  local final_line
  final_line=$(grep -E "step:[0-9]+/[0-9]+ val_loss:" "$moved" | tail -1)
  echo "=== [$label #$idx] DONE rc=$rc wall=$((t1-t0))s -> $moved" | tee -a "$SWEEPLOG"
  echo "    $final_line" | tee -a "$SWEEPLOG"
}

# Already have 1 PR run from smoke. Need 10 master + 9 more PR. Start with master to populate its compile cache.
for i in $(seq 1 10); do
  run_one master master "$i" || echo "(continuing)" | tee -a "$SWEEPLOG"
  if [ "$i" -le 9 ]; then
    run_one pr xsa-gated-layers $((i+1)) || echo "(continuing)" | tee -a "$SWEEPLOG"
  fi
done

git checkout -q xsa-gated-layers
echo "=== SWEEP COMPLETE at $(date -Is) ===" | tee -a "$SWEEPLOG"
ls "$PR_DIR" | wc -l | xargs -I{} echo "PR logs: {}"   | tee -a "$SWEEPLOG"
ls "$MASTER_DIR" | wc -l | xargs -I{} echo "MASTER logs: {}" | tee -a "$SWEEPLOG"
