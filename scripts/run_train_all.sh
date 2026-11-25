#!/usr/bin/env bash
# Phase 2: sequentially train the three LoRA adapters for one backbone/dataset.
# Usage: scripts/run_train_all.sh [model_key] [dataset]
#   model_key: qwen (default) | llama | laya | clef-flash
#   dataset:   docred (default) | redocred
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODEL="${1:-qwen}"
DATASET="${2:-docred}"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

for stage in stage1_ner stage2_coref stage3_re; do
  if [ -s "weights/$stage/${MODEL}_${DATASET}/calibration.json" ]; then
    echo "=== Skip $stage [$MODEL / $DATASET]: adapter already saved ==="
    continue
  fi
  echo "=== Training $stage [$MODEL / $DATASET] ==="
  "$PY" -m src.training.train_$stage --model "$MODEL" --dataset "$DATASET" \
    2>&1 | tee "$LOG_DIR/train_${stage}_${MODEL}_${DATASET}.log"
done
echo "=== all three adapters trained for $MODEL/$DATASET ==="
