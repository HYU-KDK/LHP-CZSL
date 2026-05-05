#!/bin/bash
# Single-seed UT-Zap VP-CMJL runner (used by parallel launcher).
# Usage: SEED=<n> GPU=<n> TS=<timestamp> bash scripts/run_vpcmjl_utzap_one_seed.sh
set -e
cd /data1/workspaces/jgshin22/LHP-CZSL/VP-CMJL
PY=/data1/workspaces/jgshin22/miniconda3/envs/llm_cluspro/bin/python

: "${SEED:?SEED required}"
: "${GPU:?GPU required}"
: "${TS:?TS required}"

SAVE=../checkpoint/vpcmjl_l14_utzap_seed${SEED}
LOG=../logs/train_vpcmjl_utzap_seed${SEED}_${TS}.log
mkdir -p "${SAVE}"

echo "=== [seed=${SEED} gpu=${GPU}] start $(date) -> ${LOG}"
CUDA_VISIBLE_DEVICES=${GPU} ${PY} train_multi_proxy.py \
    --dataset ut-zappos \
    --seed ${SEED} \
    --save_path "${SAVE}" \
    > "${LOG}" 2>&1
echo "=== [seed=${SEED} gpu=${GPU}] end   $(date)"
