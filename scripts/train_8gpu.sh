#!/bin/bash
# 8-GPU DDP training launcher for LHP-CZSL.
# Usage:
#   bash scripts/train_8gpu.sh                                    # default v2 config
#   bash scripts/train_8gpu.sh config/lhp_czsl_v2_mit_8gpu.yml    # explicit
set -euo pipefail

cd "$(dirname "$0")/.."

CFG="${1:-config/lhp_czsl_v2_mit_8gpu.yml}"

# Activate conda env (script must be run from a login-style shell or use conda init)
CONDA_BASE="/data1/workspaces/jgshin22/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate llm_cluspro

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

# Pick a free master port to avoid clashes with other tmux sessions.
MASTER_PORT="${MASTER_PORT:-29501}"

echo "[launch] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nproc_per_node=$NPROC config=$CFG port=$MASTER_PORT"

torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" \
    train.py --yml_path "$CFG"
