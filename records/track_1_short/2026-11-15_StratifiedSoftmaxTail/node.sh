#!/usr/bin/env bash
# Everything that runs inside the record's Docker image on the 8xH100 node, from the repository root.
#
#   node.sh smoke                     setup, the GPU kernel check, one candidate run
#   node.sh pool <candidate> <base>   setup, the interleaved pool (run_pool.sh), the statistics
#
# Setup is idempotent: the stack check, the data, and the attention kernel pre-seeded into a cache next
# to the repository so the runs go offline (anonymous get_kernel can return 401: the record's README,
# "Five things that will bite you", item 2).
set -euo pipefail

here=records/track_1_short/2026-11-15_StratifiedSoftmaxTail
[[ -f train_gpt.py && -d $here ]] || { echo "run from the repository root" >&2; exit 1; }
mode=${1:?smoke or pool}

git config --global --add safe.directory '*'
python - <<'PY'
import torch
assert torch.__version__ == "2.10.0+cu128", f"torch {torch.__version__}: not the record's pinned build"
assert torch.cuda.device_count() == 8, f"{torch.cuda.device_count()} GPUs visible, the run needs 8"
names = {torch.cuda.get_device_name(i) for i in range(8)}
assert all("H100" in name for name in names), f"not H100s: {names}"
print(f"stack ok: torch {torch.__version__}, 8x {names.pop()}")
PY
nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 | awk -F. '{ if ($1 < 580) { print "driver " $0 " is below 580: the record README says the run is then ~35% slower"; exit 1 } else print "driver " $0 " ok" }'

python data/cached_fineweb10B.py 9

export HF_HOME=${HF_HOME:-$(cd .. && pwd)/hf-cache}
hub=$HF_HOME/hub
kernel=devenpzak--flash-attn3-12864
if [[ ! -d $hub/kernels--$kernel ]]; then
    hf download devenpzak/flash-attn3-12864 --revision 64c1e6d1f2780e7931839f41426ddcdb564a7cb9
    cp -rl "$hub/models--$kernel" "$hub/kernels--$kernel"
fi
export HF_HUB_OFFLINE=1
echo "setup ok: data present, attention kernel cached in $HF_HOME, runs go offline"

case $mode in
    smoke)
        python "$here/gpu_check.py"
        bash run.sh
        ;;
    pool)
        bash "$here/run_pool.sh" "${2:?candidate runs}" "${3:?baseline runs}"
        pip install -q scipy
        python "$here/pool_statistics.py"
        ;;
    *)
        echo "unknown mode $mode" >&2; exit 1 ;;
esac
