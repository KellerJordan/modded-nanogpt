#!/usr/bin/env bash
set -uo pipefail

source /home/karan/speedrun/venv/bin/activate
export PYTHONUNBUFFERED=1
export DATA_PATH=/home/karan/speedrun/candidate

timing_dir=/home/karan/speedrun/logs/timing-confirmation
run_timeout="${RUN_TIMEOUT:-10m}"
start_block="${START_BLOCK:-1}"
mkdir -p "$timing_dir"

if (( start_block < 1 || start_block > 8 )); then
  echo "START_BLOCK must be between 1 and 8" >&2
  exit 89
fi

if [[ -e "$timing_dir/TIMING_CONFIRMATION_COMPLETE" ]]; then
  echo "Refusing to rerun a completed timing confirmation: $timing_dir" >&2
  exit 91
fi

run_one() {
  local block_number="$1"
  local position="$2"
  local arm="$3"
  local total_steps
  local label
  total_steps=1300
  [[ "$arm" == "candidate" ]] && total_steps=1288
  label="$(printf 'block%02d-%s-%s' "$block_number" "$position" "$arm")"

  local console_log="$timing_dir/${label}.console.txt"
  local hardware_log="$timing_dir/${label}.hardware.txt"
  local result_log="$timing_dir/${label}.result.txt"

  if [[ -e "$console_log" || -e "$result_log" ]]; then
    echo "Refusing to overwrite an existing outcome for $label" >&2
    return 92
  fi

  local gpu_pids
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "${gpu_pids//[[:space:]]/}" ]]; then
    echo "INFRASTRUCTURE_FAILURE $label dirty-gpu pids=$gpu_pids" >&2
    return 95
  fi

  {
    date --iso-8601=seconds
    hostname
    printf 'arm=%s expected_total_steps=%s\n' "$arm" "$total_steps"
    sha256sum /home/karan/speedrun/baseline/train_gpt.py
    sha256sum /home/karan/speedrun/candidate/train_gpt.py
    nvidia-smi --query-gpu=index,name,uuid,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader
  } > "$hardware_log"

  echo "BEGIN $label expected_total_steps=$total_steps"
  cd "/home/karan/speedrun/$arm" || return 1
  mkdir -p logs
  local before
  before="$(find logs -maxdepth 1 -type f -name '*.txt' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2- || true)"

  timeout --signal=TERM --kill-after=30s "$run_timeout" \
    torchrun --standalone --nproc_per_node=8 \
    /home/karan/speedrun/bin/preload-inductor.py train_gpt.py 2>&1 | tee "$console_log"
  local status=${PIPESTATUS[0]}
  if (( status != 0 )); then
    echo "INFRASTRUCTURE_FAILURE $label exit=$status" | tee -a "$console_log"
    return "$status"
  fi

  local after
  after="$(find logs -maxdepth 1 -type f -name '*.txt' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  if [[ -z "$after" || "$after" == "$before" ]]; then
    echo "INFRASTRUCTURE_FAILURE $label no-new-run-log" | tee -a "$console_log"
    return 90
  fi

  cp "$after" "$timing_dir/${label}.full.txt"
  grep -E 'Terminal-blended|step:[0-9]+/[0-9]+ val_loss:' "$console_log" | tail -n 8 | tee "$result_log"

  if ! grep -q "^step:${total_steps}/${total_steps} val_loss:" "$result_log"; then
    echo "INVALID_OUTCOME $label final-validation-missing" | tee -a "$console_log"
    return 94
  fi
  if [[ "$arm" == "candidate" ]] && ! grep -q '^Terminal-blended sharded TailEMA' "$result_log"; then
    echo "INVALID_OUTCOME $label terminal-ema-assertion-missing" | tee -a "$console_log"
    return 93
  fi

  echo "PASS $label"
}

# Frozen before the accuracy batch returned: four ABBA and four BAAB blocks,
# alternating order to balance monotonic node drift.
block_orders=(ABBA BAAB ABBA BAAB ABBA BAAB ABBA BAAB)

for index in "${!block_orders[@]}"; do
  block_number=$((index + 1))
  if (( block_number < start_block )); then
    continue
  fi
  order="${block_orders[$index]}"
  echo "BEGIN_BLOCK $(printf '%02d' "$block_number") order=$order"
  for position_index in 0 1 2 3; do
    position=$((position_index + 1))
    symbol="${order:$position_index:1}"
    arm=baseline
    [[ "$symbol" == "B" ]] && arm=candidate
    if run_one "$block_number" "$position" "$arm"; then
      :
    else
      status=$?
      printf '%s\n' "$(date --iso-8601=seconds) block=$block_number order=$order exit=$status" \
        > "$timing_dir/$(printf 'block%02d' "$block_number").BLOCK_REJECTED"
      echo "REJECT_BLOCK $(printf '%02d' "$block_number") order=$order exit=$status"
      exit "$status"
    fi
  done
  date --iso-8601=seconds > "$timing_dir/$(printf 'block%02d' "$block_number").BLOCK_ACCEPTED"
  echo "PASS_BLOCK $(printf '%02d' "$block_number") order=$order"
done

date --iso-8601=seconds > "$timing_dir/TIMING_CONFIRMATION_COMPLETE"
echo "TIMING_CONFIRMATION_COMPLETE"
