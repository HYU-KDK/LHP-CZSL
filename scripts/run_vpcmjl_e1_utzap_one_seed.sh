#!/bin/bash
# E1: VP-CMJL + Phase 2 (LLM contextual prototype gate). Single seed.
# Usage: SEED=<n> GPU=<n> TS=<timestamp> bash scripts/run_vpcmjl_e1_utzap_one_seed.sh
set -e
cd /data1/workspaces/jgshin22/LHP-CZSL/VP-CMJL
PY=/data1/workspaces/jgshin22/miniconda3/envs/llm_cluspro/bin/python

: "${SEED:?SEED required}"
: "${GPU:?GPU required}"
: "${TS:?TS required}"

LLM_DIR=../data/llm_descriptions/ut-zappos
SAVE=../checkpoint/vpcmjl_e1_l14_utzap_seed${SEED}
LOG=../logs/train_vpcmjl_e1_utzap_seed${SEED}_${TS}.log
mkdir -p "${SAVE}"

echo "=== [E1 seed=${SEED} gpu=${GPU}] start $(date) -> ${LOG}"
CUDA_VISIBLE_DEVICES=${GPU} ${PY} train_multi_proxy.py \
    --dataset ut-zappos \
    --seed ${SEED} \
    --save_path "${SAVE}" \
    --use_llm_desc \
    --llm_attr_in_context_pt "${LLM_DIR}/attr_in_context_emb.pt" \
    > "${LOG}" 2>&1
echo "=== [E1 seed=${SEED} gpu=${GPU}] end   $(date)"
