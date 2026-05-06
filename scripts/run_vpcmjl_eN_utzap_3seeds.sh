#!/bin/bash
# Generic 3-seed parallel launcher for E1/E2/E3 on UT-Zappos.
# Uses GPUs 4/5/6 (1/2/3 reserved for baseline; 0 reserved by user).
#
# Usage:
#   E=1 bash scripts/run_vpcmjl_eN_utzap_3seeds.sh
#   E=2 ALPHA=0.3 GAMMA=0.3 bash scripts/run_vpcmjl_eN_utzap_3seeds.sh
#   E=3 bash scripts/run_vpcmjl_eN_utzap_3seeds.sh
#
# Each seed is dispatched to a separate tmux session. Master log at
# logs/run_vpcmjl_e${E}_utzap_3seeds_<TS>.log; per-seed logs separate.
set -e
cd /data1/workspaces/jgshin22/LHP-CZSL

: "${E:?E required (1, 2, or 3)}"
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG=logs/run_vpcmjl_e${E}_utzap_3seeds_${TS}.log
mkdir -p logs

ONE_SEED_SCRIPT="scripts/run_vpcmjl_e${E}_utzap_one_seed.sh"
if [[ ! -f "${ONE_SEED_SCRIPT}" ]]; then
    echo "[ERR] missing ${ONE_SEED_SCRIPT}" | tee -a "${MASTER_LOG}"
    exit 1
fi

# Pass-through env vars (ALPHA / GAMMA / SAMPLE_K) for E2/E3
EXTRA_ENV=""
[[ -n "${ALPHA}" ]]    && EXTRA_ENV="${EXTRA_ENV} ALPHA=${ALPHA}"
[[ -n "${GAMMA}" ]]    && EXTRA_ENV="${EXTRA_ENV} GAMMA=${GAMMA}"
[[ -n "${SAMPLE_K}" ]] && EXTRA_ENV="${EXTRA_ENV} SAMPLE_K=${SAMPLE_K}"

echo "=== launch E${E} 3-seed @ ${TS} (gpus 4/5/6) ${EXTRA_ENV}" | tee "${MASTER_LOG}"

for s in 0 1 2; do
    GPU=$((4 + s))
    SESSION="vpcmjl-e${E}-s${s}"
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    tmux new-session -d -s "${SESSION}" \
        "SEED=${s} GPU=${GPU} TS=${TS} ${EXTRA_ENV} bash ${ONE_SEED_SCRIPT}"
    echo "  spawned ${SESSION} on GPU ${GPU}" | tee -a "${MASTER_LOG}"
done

echo "=== all 3 seeds dispatched. attach: tmux attach -t vpcmjl-e${E}-s0" | tee -a "${MASTER_LOG}"
echo "=== logs: logs/train_vpcmjl_e${E}_utzap_seed{0,1,2}_${TS}.log" | tee -a "${MASTER_LOG}"
