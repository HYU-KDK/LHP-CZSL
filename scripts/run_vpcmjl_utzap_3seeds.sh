#!/bin/bash
# Sequential 3-seed UT-Zap VP-CMJL (ICCV 2025) baseline runs.
# Protocol from upstream config/ut-zappos.yml: ViT-L/14, lr=5e-4, BS=16, 20 epochs,
# StepLR(step=3, gamma=0.5), Adam, weight_decay=1e-5, fusion=BiFusion.
# Same dataset directory as our LHP-CZSL UT-Zap runs.
set -e
cd /data1/workspaces/jgshin22/LHP-CZSL/VP-CMJL
PY=/data1/workspaces/jgshin22/miniconda3/envs/llm_cluspro/bin/python
TS=$(date +%Y%m%d_%H%M%S)
GPU=${GPU:-1}

mkdir -p ../logs ../checkpoint

for SEED in 0 1 2; do
  SAVE=../checkpoint/vpcmjl_l14_utzap_seed${SEED}
  LOG=../logs/train_vpcmjl_utzap_seed${SEED}_${TS}.log
  mkdir -p "${SAVE}"
  echo "=== [seed=${SEED}] start $(date) -> ${LOG}" | tee -a ../logs/run_vpcmjl_utzap_3seeds_${TS}.log
  CUDA_VISIBLE_DEVICES=${GPU} ${PY} train_multi_proxy.py \
      --dataset ut-zappos \
      --seed ${SEED} \
      --save_path "${SAVE}" \
      > "${LOG}" 2>&1
  echo "=== [seed=${SEED}] end   $(date)" | tee -a ../logs/run_vpcmjl_utzap_3seeds_${TS}.log
done

echo "=== ALL SEEDS DONE $(date)" | tee -a ../logs/run_vpcmjl_utzap_3seeds_${TS}.log
