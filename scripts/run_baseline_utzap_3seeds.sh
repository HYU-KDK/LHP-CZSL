#!/bin/bash
# Sequential 3-seed UT-Zap cluspro_baseline runs (post AMP/NaN fix)
set -e
cd /home/student/dongki/LHP-CZSL
PY=/home/student/anaconda3/envs/lhp_czsl/bin/python
TS=$(date +%Y%m%d_%H%M%S)

for SEED in 0 1 2; do
  YML=config/cluspro_baseline_utzap_l14_v2_seed${SEED}.yml
  LOG=logs/train_baseline_utzap_v2_seed${SEED}_${TS}.log
  echo "=== [seed=${SEED}] start $(date) -> ${LOG}" | tee -a logs/run_baseline_utzap_3seeds_${TS}.log
  CUDA_VISIBLE_DEVICES=1 ${PY} train.py --yml_path ${YML} > ${LOG} 2>&1
  echo "=== [seed=${SEED}] end   $(date)" | tee -a logs/run_baseline_utzap_3seeds_${TS}.log
done

echo "=== ALL SEEDS DONE $(date)" | tee -a logs/run_baseline_utzap_3seeds_${TS}.log
