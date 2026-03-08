#!/bin/bash

# Image2Haiku Model Testing Script
# Usage: ./run_test.sh <experiment_dir> [max_samples] [save_dir] [ablation_mode]

set -euo pipefail

# =================== 配置区域 ===================
export NLTK_DATA=${NLTK_DATA:-/root/autodl-tmp/nltk_data}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf_cache}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}  # 避免多进程冲突

EXPERIMENT_DIR=${1:-${EXPERIMENT_DIR:-}}  # 优先使用命令行参数，其次环境变量
MAX_SAMPLES_ARG=${2:-${MAX_SAMPLES:-}}
# 额外可选参数：保存输出目录与消融模式（multi/single/zero-fill）
SAVE_DIR_ARG=${3:-${SAVE_DIR:-}}
ABLATION_MODE_ARG=${4:-${ABLATION_MODE:-}}

if [[ -z "$EXPERIMENT_DIR" ]]; then
    echo "Usage: ./run_test.sh <experiment_dir> [max_samples] [save_dir] [ablation_mode]"
    echo "Provide the experiment directory that contains checkpoints."
    exit 1
fi

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "Error: experiment directory not found: $EXPERIMENT_DIR" >&2
    exit 1
fi

# 数据路径可通过环境变量覆盖；默认指向完整训练集
DEFAULT_FEATURES_PATH="../../data/model1_data/image_features.npy"
DEFAULT_IMAGE_IDS_PATH="../../data/model1_data/image_ids.json"
DEFAULT_ANNOTATIONS_PATH="../../data/model1_data/merged_captions_async_augmented.json"

FEATURES_PATH=${FEATURES_PATH:-$DEFAULT_FEATURES_PATH}
IMAGE_IDS_PATH=${IMAGE_IDS_PATH:-$DEFAULT_IMAGE_IDS_PATH}
ANNOTATIONS_PATH=${ANNOTATIONS_PATH:-$DEFAULT_ANNOTATIONS_PATH}

ARGS=(
    --experiment_dir "$EXPERIMENT_DIR"
    --features_path "$FEATURES_PATH"
    --image_ids_path "$IMAGE_IDS_PATH"
    --annotations_path "$ANNOTATIONS_PATH"
)

if [[ -n "$MAX_SAMPLES_ARG" ]]; then
    ARGS+=(--max_samples "$MAX_SAMPLES_ARG")
fi

# 保存预测与参考到指定目录，便于后续bootstrap评估
if [[ -n "$SAVE_DIR_ARG" ]]; then
    mkdir -p "$SAVE_DIR_ARG"
    ARGS+=(--save_dir "$SAVE_DIR_ARG")
fi

# 消融模式：multi-image(默认)/single-image/zero-fill，占位保持结构一致
if [[ -n "$ABLATION_MODE_ARG" ]]; then
    ARGS+=(--ablation_mode "$ABLATION_MODE_ARG")
fi

python scripts/test_best_model.py "${ARGS[@]}"
