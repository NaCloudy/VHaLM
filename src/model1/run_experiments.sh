#!/bin/bash

# Batch experiment runner for Model1: baseline, improved, ablations, bootstrap
# Usage:
#   ./run_experiments.sh

set -euo pipefail

# 环境配置（避免每次重新下载NLTK/HF缓存）
export NLTK_DATA=${NLTK_DATA:-/root/autodl-tmp/nltk_data}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf_cache}

# =================== 配置区域 ===================
# 模型 checkpoint 目录
V0_DIR="/root/autodl-tmp/myprj/src/model1/outputs/image2haiku_20251106_125849(midway-300-attention)"
V1_DIR="/root/autodl-tmp/myprj/src/model1/outputs/exp2_att2_tag6_l.3r1e-4_ws150_20251124_134150(final_best)"

# 输出与评估配置
OUT_DIR="/root/autodl-tmp/myprj/outputs"
PRED_FILENAME="preds.json"
REF_FILENAME="refs.json"
BOOTSTRAP_B=200 #2
BOOTSTRAP_SEED=42

# 留空=全量测试集；设置为数值可抽样子集（例：1000）
MAX_SAMPLES=""

ORDER=(v0_multi v1_multi v1_single v1_zero)
declare -A MODE_MAP=(
  [v0_multi]="multi"
  [v1_multi]="multi"
  [v1_single]="single"
  [v1_zero]="zero-fill"
)
declare -A MODEL_MAP=(
  [v0_multi]="v0"
  [v1_multi]="v1"
  [v1_single]="v1"
  [v1_zero]="v1"
)
declare -A DIR_MAP=(
  [v0_multi]="$V0_DIR"
  [v1_multi]="$V1_DIR"
  [v1_single]="$V1_DIR"
  [v1_zero]="$V1_DIR"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TEST="$SCRIPT_DIR/run_test.sh"
BOOTSTRAP_SCRIPT="$SCRIPT_DIR/scripts/bootstrap_ci.py"

mkdir -p "$OUT_DIR"

run_variant() {
  local tag="$1"
  local model="$2"
  local mode="$3"
  local exp_dir="$4"
  local save_dir="$OUT_DIR/$tag"

  echo "[Run] $tag ($model, $mode)"

  mkdir -p "$save_dir"
  local args=("$exp_dir")
  if [[ -n "$MAX_SAMPLES" ]]; then
    args+=("$MAX_SAMPLES")
  else
    args+=("")
  fi
  args+=("$save_dir" "$mode")

  "$RUN_TEST" "${args[@]}" || {
    echo "$tag run failed" >&2; exit 2;
  }

  local pred_path="$save_dir/$PRED_FILENAME"
  local ref_path="$save_dir/$REF_FILENAME"
  if [[ -f "$pred_path" && -f "$ref_path" ]]; then
    echo "  [Bootstrap] $tag"
    python "$BOOTSTRAP_SCRIPT" \
      --pred_path "$pred_path" \
      --ref_path "$ref_path" \
      --B "$BOOTSTRAP_B" \
      --seed "$BOOTSTRAP_SEED" \
      > "$save_dir/metrics_ci.json"
  else
    echo "  [Warn] Missing $pred_path or $ref_path, skip bootstrap" >&2
  fi
}

for tag in "${ORDER[@]}"; do
  run_variant "$tag" "${MODEL_MAP[$tag]}" "${MODE_MAP[$tag]}" "${DIR_MAP[$tag]}"
done

echo "All runs completed. Check outputs under: $OUT_DIR"
