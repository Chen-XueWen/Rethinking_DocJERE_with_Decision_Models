#!/usr/bin/env bash
# Full end-to-end experiment for answerdotai/ModernBERT-large (+ a fresh
# Laya-style decision head; see src/models/build_encoder_decision.py), run
# strictly in this order on the FULL test splits:
#   1. Fine-tuned  DocRED    (train 3 stages on ALL DocRED train rows, then test)
#   2. Fine-tuned  Re-DocRED (train 3 stages on ALL Re-DocRED train rows, then test)
#
# Zero-shot is NOT run: the head is freshly initialised (untrained), and it
# measured 648 s/doc (~11 days for both test sets) because it over-tags ~80
# entities/doc and an encoder pays one forward pass per question
# (FINDINGS.md 2026-11-14). Untrained-head scores are reported as 0.
#
#   mkdir -p results/mbert_large && nohup bash scripts/run_mbert_large.sh > results/mbert_large/run.log 2>&1 &
#
# Expected wall time on the GB10 (measured 2026-11-14, FINDINGS.md): training
# ~30-35 h for all rows x 1 epoch on both datasets (Laya-style models score ONE
# question per sequence, so a packed row becomes ~10-30 training items:
# ~0.9M DocRED + ~1.1M Re-DocRED items at 30-100 ms each). EPOCHS=3 is ~3x.
# Each trained stage saves a merged 16-bit checkpoint (~0.8 GB; 6 = ~5 GB).
#
# Resumable: each step writes its artifact and is skipped when it exists
# (adapters: weights/stage*/mbert_large_<ds>/calibration.json; predictions are
# written to *.partial and renamed only when complete). Delete to re-run.
#
# Env overrides (defaults = the full run):
#   MODEL=mbert_large        model key in configs/models.yaml
#   EVAL_LIMIT=0         docs per inference run (0 = whole test split)
#   TRAIN_MAX_ROWS=0     rows per stage for training (0 = every train row)
#   EPOCHS=              override the per-stage epochs in lora_params.yaml (default 1)
#   GOLD_ENTS=1          also run gold-entity RE eval (micro/Ign-F1) per run
set -uo pipefail
source "$(dirname "$0")/env.sh"

MODEL="${MODEL:-mbert_large}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-0}"
EPOCHS="${EPOCHS:-}"
GOLD_ENTS="${GOLD_ENTS:-1}"
OUT="results/$MODEL"
LOG="$OUT/logs"
mkdir -p "$LOG"

LIMIT_ARG=(); [ "$EVAL_LIMIT" != "0" ] && LIMIT_ARG=(--limit "$EVAL_LIMIT")
EPOCH_ARG=(); [ -n "$EPOCHS" ] && EPOCH_ARG=(--epochs "$EPOCHS")

stamp() { date '+%F %T'; }
fail() { echo "!!! [$(stamp)] FAILED: $1 (see $LOG/$2.log)" | tee -a "$LOG/failures.log"; exit 1; }

run_step() {  # run_step <artifact> <log-name> <cmd...>; aborts the script on failure
  local artifact="$1" log="$2"; shift 2
  if [ -s "$artifact" ]; then echo "--- [$(stamp)] skip (exists): $artifact"; return 0; fi
  echo "--- [$(stamp)] $log: $*"
  "$@" > "$LOG/$log.log" 2>&1 || fail "$log" "$log"
  grep -E "^(timing summary|wrote|saved|\[stage)" "$LOG/$log.log" | tail -3
}

infer_and_eval() {  # infer_and_eval <tag: zeroshot|finetuned> <dataset>
  local tag="$1" ds="$2" zs=()
  [ "$tag" = "zeroshot" ] && zs=(--zero-shot)
  local P="$OUT/preds_${MODEL}_${tag}_${ds}_test_e2e.jsonl"
  run_step "$P" "${tag}_${ds}_e2e" \
    "$PY" -m src.inference.pipeline --model "$MODEL" "${zs[@]}" \
      --dataset "$ds" --split test --mode e2e "${LIMIT_ARG[@]}" --out "$P"
  # strict E2E + stage-isolated NER/coref scores from the same predictions
  run_step "${P%.jsonl}.strict_e2e.json" "${tag}_${ds}_e2e_eval" \
    "$PY" -m src.evaluation.strict_e2e_eval --preds "$P" --dataset "$ds" --split test
  run_step "${P%.jsonl}.isolated_eval.json" "${tag}_${ds}_e2e_isolated" \
    "$PY" -m src.evaluation.isolated_eval --preds "$P" --dataset "$ds" --split test \
      --tasks ner coref
  if [ "$GOLD_ENTS" = "1" ]; then
    local G="$OUT/preds_${MODEL}_${tag}_${ds}_test_gold-ents.jsonl"
    run_step "$G" "${tag}_${ds}_gold" \
      "$PY" -m src.inference.pipeline --model "$MODEL" "${zs[@]}" \
        --dataset "$ds" --split test --mode gold-ents "${LIMIT_ARG[@]}" --out "$G"
    run_step "${G%.jsonl}.isolated_eval.json" "${tag}_${ds}_gold_eval" \
      "$PY" -m src.evaluation.isolated_eval --preds "$G" --dataset "$ds" --split test \
        --tasks re
  fi
}

train_all() {  # train_all <dataset>
  local ds="$1"
  for stage in stage1_ner stage2_coref stage3_re; do
    run_step "weights/$stage/${MODEL}_${ds}/calibration.json" "train_${stage}_${ds}" \
      "$PY" -m src.training.train_$stage --model "$MODEL" --dataset "$ds" \
        --max-rows "$TRAIN_MAX_ROWS" "${EPOCH_ARG[@]}"
  done
}

echo "=============================================================="
echo " $MODEL full run — started $(stamp)"
echo " EVAL_LIMIT=$EVAL_LIMIT (0=all) TRAIN_MAX_ROWS=$TRAIN_MAX_ROWS (0=all)"
echo " EPOCHS=${EPOCHS:-from config} GOLD_ENTS=$GOLD_ENTS"
echo "=============================================================="

[ -s data/processed/stage3_re/redocred_train.jsonl ] || bash scripts/run_prep.sh

# Raw ModernBERT-large can't be loaded as a decision model by unsloth; build
# the Laya-layout base checkpoint (pretrained encoder + fresh, seeded head) once.
run_step "weights/base/$MODEL/model.safetensors" "build_base" \
  "$PY" -m src.models.build_encoder_decision --model "$MODEL"

echo; echo ">>> [1/2] Fine-tuned DocRED";    train_all docred;   infer_and_eval finetuned docred
echo; echo ">>> [2/2] Fine-tuned Re-DocRED"; train_all redocred; infer_and_eval finetuned redocred

"$PY" -m src.evaluation.summarize --results-dir "$OUT" > "$OUT/BENCHMARK.md"
cat "$OUT/BENCHMARK.md"
echo "=============================================================="
echo " $MODEL full run — finished $(stamp). Summary: $OUT/BENCHMARK.md"
echo "=============================================================="
