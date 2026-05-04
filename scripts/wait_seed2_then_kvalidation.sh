#!/bin/bash
# Wait for a training PID to finish, then run K-validation (LLM K vs visual K)
# on UT-Zap and mit-states (v1 + v2 sub_meanings) sequentially on GPU 1.
#
# Usage:
#   bash scripts/wait_seed2_then_kvalidation.sh <PID>
#   PID=12345 bash scripts/wait_seed2_then_kvalidation.sh
#
# Datasets are read from $DATA_ROOT (default: /home/student/dongki/DPAS/data).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
TARGET_PID=${1:-${PID:-}}
DATA_ROOT=${DATA_ROOT:-/home/student/dongki/DPAS/data}
if [[ -z "${TARGET_PID}" ]]; then
  echo "usage: $0 <PID>   or PID=<pid> $0" >&2; exit 1
fi
TS=$(date +%Y%m%d_%H%M%S)
WAIT_LOG=logs/k_validation/wait_${TS}.log
mkdir -p logs/k_validation
echo "[wait] target PID=${TARGET_PID} start $(date)" | tee -a ${WAIT_LOG}
while kill -0 ${TARGET_PID} 2>/dev/null; do sleep 60; done
echo "[wait] PID ${TARGET_PID} ended at $(date)" | tee -a ${WAIT_LOG}

run_one() {
  local DSPATH=$1 SUB=$2 TAG=$3
  local LOG=logs/k_validation/run_${TAG}_${TS}.log
  echo "[run] ${TAG}: ${SUB} → ${LOG}" | tee -a ${WAIT_LOG}
  CUDA_VISIBLE_DEVICES=1 ${PY} tools/k_validation.py \
    --dataset_path ${DSPATH} \
    --sub_meanings_path ${SUB} \
    --output_dir logs/k_validation \
    > ${LOG} 2>&1
  echo "[run] ${TAG} done at $(date) (exit=$?)" | tee -a ${WAIT_LOG}
}

run_one ${DATA_ROOT}/ut-zap50k   data/sub_meanings_utzap.json   utzap
run_one ${DATA_ROOT}/mit-states  data/sub_meanings_mit.json     mit_v1
run_one ${DATA_ROOT}/mit-states  data/sub_meanings_mit_v2.json  mit_v2

echo "[done] all k-validation runs complete at $(date)" | tee -a ${WAIT_LOG}
