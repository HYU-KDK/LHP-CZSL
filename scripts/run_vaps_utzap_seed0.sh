#!/bin/bash
# Single-seed UT-Zap VAPS run.
set -e
cd /data1/workspaces/jgshin22/LHP-CZSL
PY=/data1/workspaces/jgshin22/miniconda3/envs/llm_cluspro/bin/python
TS=$(date +%Y%m%d_%H%M%S)
GPU=${GPU:-0}
SEED=${SEED:-0}

mkdir -p logs checkpoint
YML=config/vaps_utzap_l14_seed${SEED}.yml
LOG=logs/train_vaps_utzap_seed${SEED}_${TS}.log
echo "=== [seed=${SEED}] start $(date) -> ${LOG}"
CUDA_VISIBLE_DEVICES=${GPU} ${PY} train.py --yml_path ${YML} > ${LOG} 2>&1
echo "=== [seed=${SEED}] end   $(date)"
