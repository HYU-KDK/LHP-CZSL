#!/bin/bash
# Sequential 3-seed UT-Zap LHP-CZSL v3_text (sub-meaning text-side ensemble) runs.
# Same backbone/conditions as v1_init_only: ViT-L/14, fp16+GradScaler+guard, 15 ep, lr=1e-4, bs=8 grad_accum=8.
set -e
cd /home/student/dongki/LHP-CZSL
PY=/home/student/anaconda3/envs/lhp_czsl/bin/python
TS=$(date +%Y%m%d_%H%M%S)

for SEED in 0 1 2; do
  YML=config/lhp_czsl_v3_text_utzap_l14_seed${SEED}.yml
  LOG=logs/train_v3_text_utzap_seed${SEED}_${TS}.log
  echo "=== [seed=${SEED}] start $(date) -> ${LOG}" | tee -a logs/run_v3_text_utzap_3seeds_${TS}.log
  CUDA_VISIBLE_DEVICES=1 ${PY} train.py --yml_path ${YML} > ${LOG} 2>&1
  echo "=== [seed=${SEED}] end   $(date)" | tee -a logs/run_v3_text_utzap_3seeds_${TS}.log
done

echo "=== ALL SEEDS DONE $(date)" | tee -a logs/run_v3_text_utzap_3seeds_${TS}.log
