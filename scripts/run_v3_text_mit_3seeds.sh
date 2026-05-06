#!/bin/bash
# Sequential 3-seed mit-states LHP-CZSL v3_text run with LLM-generated visual
# descriptions (data/descriptions_mit.json). ViT-L/14, fp16+GradScaler+guard.
set -e
cd /home/student/dongki/LHP-CZSL
PY=/home/student/anaconda3/envs/lhp_czsl/bin/python
TS=$(date +%Y%m%d_%H%M%S)

for SEED in 0 1 2; do
  YML=config/lhp_czsl_v3_text_mit_l14_seed${SEED}.yml
  LOG=logs/train_v3_text_mit_seed${SEED}_${TS}.log
  echo "=== [seed=${SEED}] start $(date) -> ${LOG}" | tee -a logs/run_v3_text_mit_3seeds_${TS}.log
  CUDA_VISIBLE_DEVICES=1 ${PY} train.py --yml_path ${YML} > ${LOG} 2>&1
  echo "=== [seed=${SEED}] end   $(date)" | tee -a logs/run_v3_text_mit_3seeds_${TS}.log
done

echo "=== ALL SEEDS DONE $(date)" | tee -a logs/run_v3_text_mit_3seeds_${TS}.log
