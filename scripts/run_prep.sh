#!/usr/bin/env bash
# Phase 1: build all processed JSONL datasets for both corpora and all stages.
set -euo pipefail
source "$(dirname "$0")/env.sh"

for ds in docred redocred; do
  echo "=== Stage 1 (NER) [$ds] ==="
  "$PY" -m src.data_prep.make_stage1_ner --dataset "$ds"
  echo "=== Stage 2 (Coref) [$ds] ==="
  "$PY" -m src.data_prep.make_stage2_coref --dataset "$ds"
  echo "=== Stage 3 (RE) [$ds] ==="
  "$PY" -m src.data_prep.make_stage3_re --dataset "$ds"
done

echo "=== done. Row counts: ==="
wc -l data/processed/*/*.jsonl
