#!/bin/bash
# E2: VP-CMJL + Phase 3 (alpha-blend visual proxy init + L_unseen). Single seed.
# Usage: SEED=<n> GPU=<n> TS=<timestamp> bash scripts/run_vpcmjl_e2_utzap_one_seed.sh
# Optional: ALPHA, GAMMA, SAMPLE_K
set -e
cd /data1/workspaces/jgshin22/LHP-CZSL/VP-CMJL
PY=/data1/workspaces/jgshin22/miniconda3/envs/llm_cluspro/bin/python

: "${SEED:?SEED required}"
: "${GPU:?GPU required}"
: "${TS:?TS required}"

ALPHA=${ALPHA:-0.3}
GAMMA=${GAMMA:-0.3}
SAMPLE_K=${SAMPLE_K:-32}

LLM_DIR=../data/llm_descriptions/ut-zappos
SAVE=../checkpoint/vpcmjl_e2_l14_utzap_seed${SEED}
LOG=../logs/train_vpcmjl_e2_utzap_seed${SEED}_${TS}.log
mkdir -p "${SAVE}"

echo "=== [E2 seed=${SEED} gpu=${GPU} alpha=${ALPHA} gamma=${GAMMA}] start $(date) -> ${LOG}"
CUDA_VISIBLE_DEVICES=${GPU} ${PY} train_multi_proxy.py \
    --dataset ut-zappos \
    --seed ${SEED} \
    --save_path "${SAVE}" \
    --proxy_alpha ${ALPHA} \
    --llm_proxy_init_pt "${LLM_DIR}/proxy_init_emb.pt" \
    --unseen_alignment_weight ${GAMMA} \
    --unseen_alignment_sample_k ${SAMPLE_K} \
    --llm_unseen_pt "${LLM_DIR}/unseen_comp_emb.pt" \
    > "${LOG}" 2>&1
echo "=== [E2 seed=${SEED} gpu=${GPU}] end   $(date)"
