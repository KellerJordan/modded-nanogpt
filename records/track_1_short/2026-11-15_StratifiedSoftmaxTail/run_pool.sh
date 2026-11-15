#!/usr/bin/env bash
# Interleaved run pool on one 8xH100 node: this branch (candidate) and the untouched record (baseline).
#
#   bash records/track_1_short/2026-11-15_StratifiedSoftmaxTail/run_pool.sh <candidate runs> <baseline runs>
#
# Run from the repository root, inside the environment whose torchrun launches the run. The baseline
# legs run the record's own commit from a second checkout (../modded-nanogpt-record), so they are the
# record's code byte for byte; both checkouts read the same data. Legs alternate, the candidate first,
# so both pools see the same machine state. Each finished log is copied into this folder's candidate/
# or baseline/ directory. A failed leg stops the pool; logs already copied stay.
set -euo pipefail

candidate_runs=${1:?candidate runs}
baseline_runs=${2:?baseline runs}
record_commit=4ea6b937337a4889b8cfe3f38a93d120048d8f71
root=$(pwd)
here=$root/records/track_1_short/2026-11-15_StratifiedSoftmaxTail
record=$root/../modded-nanogpt-record
[[ -f train_gpt.py && -d track_1_short ]] || { echo "run from the repository root" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "commit or drop local changes first" >&2; exit 1; }
git worktree prune  # a checkout made in an earlier container is gone; forget it
[[ -d $record ]] || git worktree add --detach "$record" "$record_commit"
[[ "$(git -C "$record" rev-parse HEAD)" == "$record_commit" ]] || { echo "$record is not the record commit" >&2; exit 1; }
export DATA_PATH=${DATA_PATH:-$root}
mkdir -p "$here/candidate" "$here/baseline"

leg() {  # leg <name> <checkout>
    local name=$1 dir=$2 before after log
    before=$(ls "$dir/logs" 2>/dev/null | sort || true)
    echo "=== $(date -u +%FT%TZ) $name leg in $dir"
    (cd "$dir" && torchrun --standalone --nproc_per_node=8 train_gpt.py)
    after=$(ls "$dir/logs" | sort)
    log=$(comm -13 <(echo "$before") <(echo "$after") | grep '\.txt$')
    [[ $(echo "$log" | wc -l) -eq 1 ]] || { echo "expected one new log in $dir/logs" >&2; exit 1; }
    cp "$dir/logs/$log" "$here/$name/$log"
    grep -h "val_loss" "$here/$name/$log" | tail -1
}

done_candidate=0
done_baseline=0
while (( done_candidate < candidate_runs || done_baseline < baseline_runs )); do
    if (( done_candidate < candidate_runs )); then
        leg candidate "$root"
        done_candidate=$((done_candidate + 1))
    fi
    # Spread the baseline legs evenly through the candidate legs.
    if (( done_baseline < baseline_runs && done_baseline * candidate_runs < done_candidate * baseline_runs )); then
        leg baseline "$record"
        done_baseline=$((done_baseline + 1))
    fi
done
echo "=== $(date -u +%FT%TZ) POOL DONE candidate=$done_candidate baseline=$done_baseline"
