#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

LOGDIR="logs/openworld"
mkdir -p "$LOGDIR"

PY=/home/student/anaconda3/envs/lhp_czsl/bin/python
export CUDA_VISIBLE_DEVICES=1

declare -a RUNS=(
  "cluspro_baseline_l14_utzap_v2_seed0  config/cluspro_baseline_utzap_l14_v2_seed0.yml"
  "cluspro_baseline_l14_utzap_v2_seed1  config/cluspro_baseline_utzap_l14_v2_seed1.yml"
  "cluspro_baseline_l14_utzap_v2_seed2  config/cluspro_baseline_utzap_l14_v2_seed2.yml"
  "lhp_czsl_v1_init_only_l14_utzap_seed0  config/lhp_czsl_v1_init_only_utzap_l14_seed0.yml"
  "lhp_czsl_v1_init_only_l14_utzap_seed1  config/lhp_czsl_v1_init_only_utzap_l14_seed1.yml"
  "lhp_czsl_v1_init_only_l14_utzap_seed2  config/lhp_czsl_v1_init_only_utzap_l14_seed2.yml"
  "lhp_czsl_v3_text_l14_utzap_seed0  config/lhp_czsl_v3_text_utzap_l14_seed0.yml"
  "lhp_czsl_v3_text_l14_utzap_seed1  config/lhp_czsl_v3_text_utzap_l14_seed1.yml"
  "lhp_czsl_v3_text_l14_utzap_seed2  config/lhp_czsl_v3_text_utzap_l14_seed2.yml"
)

TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOGDIR/summary_openworld_utzap_${TS}.txt"
echo "tag,mode,best_seen,best_unseen,best_hm,AUC,attr_acc,obj_acc" > "$SUMMARY"

extract_metrics () {
  # last 'best_seen ... obj_acc ...' line
  grep -aE "best_seen" "$1" | tail -1 | \
    sed -E 's/.*best_seen[[:space:]]+([0-9.]+).*best_unseen[[:space:]]+([0-9.]+).*best_hm[[:space:]]+([0-9.]+).*AUC[[:space:]]+([0-9.]+).*attr_acc[[:space:]]+([0-9.]+).*obj_acc[[:space:]]+([0-9.]+).*/\1,\2,\3,\4,\5,\6/'
}

for entry in "${RUNS[@]}"; do
  TAG=$(echo "$entry" | awk '{print $1}')
  YML=$(echo "$entry" | awk '{print $2}')
  CKPT="checkpoint/${TAG}/val_best.pt"

  if [[ ! -f "$CKPT" ]]; then
    echo "[skip] $TAG: missing $CKPT"
    continue
  fi

  for MODE in closed open; do
    LOG="$LOGDIR/${TAG}_${MODE}_${TS}.log"
    echo "[eval] $TAG ($MODE) -> $LOG"
    if [[ "$MODE" == "open" ]]; then
      $PY test.py --yml_path "$YML" --load_model "$CKPT" --open_world True > "$LOG" 2>&1
    else
      $PY test.py --yml_path "$YML" --load_model "$CKPT" > "$LOG" 2>&1
    fi
    METRICS=$(extract_metrics "$LOG")
    if [[ -n "$METRICS" ]]; then
      echo "${TAG},${MODE},${METRICS}" >> "$SUMMARY"
    else
      echo "${TAG},${MODE},NO_METRICS" >> "$SUMMARY"
    fi
  done

  # Failure-case analysis (closed-world test set)
  FAIL_DIR="logs/failure_analysis/${TS}"
  mkdir -p "$FAIL_DIR"
  FAIL_LOG="$FAIL_DIR/${TAG}.log"
  echo "[failure] $TAG -> $FAIL_LOG"
  $PY tools/failure_analysis.py --yml_path "$YML" --load_model "$CKPT" \
      --output_dir "$FAIL_DIR" > "$FAIL_LOG" 2>&1
done

# Aggregate failure analyses (top confused attrs across all checkpoints)
$PY tools/failure_aggregate.py "$FAIL_DIR" > "$FAIL_DIR/_aggregate.txt" 2>&1 || true

echo "[done] eval summary  -> $SUMMARY"
echo "[done] failure jsons -> $FAIL_DIR/"
column -t -s, "$SUMMARY"
