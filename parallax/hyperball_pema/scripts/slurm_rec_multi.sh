#!/usr/bin/env bash
# Generalized multi-seed launcher for any Parallax-patched Track-3 port.
# Submits one slurm job array (one seed per array task) for the script named by
# $SCRIPT. Clone of slurm_rec27_soaph_multi.sh with the training script path and
# the Phase-5 tuning knobs (R_MULT / SOAPH_LR) exposed as env-var overrides.
#
# Usage:
#   JOB_NAME=bench-rec34 SEEDS=1-2 SCRIPT=rec34_rre.py \
#     PARALLAX_PATH=/home/guests/yifei/parallax-repo \
#     bash parallax/slurm_rec_multi.sh
#
# Env-var overrides:  SCRIPT (default rec27_soaph.py), SEEDS (default "1-2"),
#                     R_MULT (default 1.0), SOAPH_LR (default 0.018),
#                     JOB_NAME, NGPU, QOS, PARTITION, TIME, PARALLAX_PATH

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
OUTPUT_DIR="${REPO_DIR}/parallax/outputs/slurm"
mkdir -p "${OUTPUT_DIR}"

SCRIPT=${SCRIPT:-"rec27_soaph.py"}
JOB_NAME=${JOB_NAME:-"bench-${SCRIPT%.py}"}
NGPU=${NGPU:-8}
QOS=${QOS:-"guest-dev"}
PARTITION=${PARTITION:-"main"}
TIME=${TIME:-"01:30:00"}
SEEDS=${SEEDS:-"1-2"}
R_MULT=${R_MULT:-"1.0"}
SOAPH_LR=${SOAPH_LR:-"0.018"}
QK_SCALE=${QK_SCALE:-"0.12"}
LR_MUOWN=${LR_MUOWN:-"0.0045"}
TRAIN_STEPS=${TRAIN_STEPS:-"3125"}
LR_STEPS=${LR_STEPS:-"3200"}
MICRO_BATCH_SEQUENCES=${MICRO_BATCH_SEQUENCES:-"16"}
LOG_PLX_RATIO=${LOG_PLX_RATIO:-"0"}
RHO_SCALE=${RHO_SCALE:-"1.0"}
EMA_STEPSIZE=${EMA_STEPSIZE:-"0.3"}
EMA_DECAY=${EMA_DECAY:-"0.99"}
EMA_PREFILL=${EMA_PREFILL:-"323"}
PARALLAX_PATH=${PARALLAX_PATH:-"/home/guests/yifei/parallax-repo"}

# Heredoc interpolates submit-time values (REPO_DIR, PARALLAX_PATH, NGPU, SCRIPT,
# R_MULT, SOAPH_LR) while leaving \${SLURM_ARRAY_TASK_ID} literal so SLURM expands
# it per-task on the worker. Outer `bash -lc` ensures bash (for set -o pipefail).
WRAP_CMD=$(cat <<EOF
bash -lc '
set -euo pipefail
cd ${REPO_DIR}
export PYTHONUNBUFFERED=1
export PARALLAX_PATH=${PARALLAX_PATH}
export LOG_PLX_RATIO=${LOG_PLX_RATIO}
export RHO_SCALE=${RHO_SCALE}
export EMA_STEPSIZE=${EMA_STEPSIZE}
export EMA_DECAY=${EMA_DECAY}
export EMA_PREFILL=${EMA_PREFILL}
export R_MULT=${R_MULT}
export SOAPH_LR=${SOAPH_LR}
export QK_SCALE=${QK_SCALE}
export LR_MUOWN=${LR_MUOWN}
export TRAIN_STEPS=${TRAIN_STEPS}
export LR_STEPS=${LR_STEPS}
export MICRO_BATCH_SEQUENCES=${MICRO_BATCH_SEQUENCES}
export SEED=\${SLURM_ARRAY_TASK_ID}
uv run torchrun --standalone --nproc_per_node=${NGPU} parallax/${SCRIPT}
'
EOF
)

sbatch \
    --job-name "${JOB_NAME}" \
    --nodes 1 \
    --gpus "${NGPU}" \
    --qos "${QOS}" \
    --partition "${PARTITION}" \
    --time "${TIME}" \
    --array "${SEEDS}" \
    --output "${OUTPUT_DIR}/slurm-%A_%a.out" \
    --wrap "${WRAP_CMD}"
