#!/usr/bin/env bash
# First full end-to-end experiment, one backbone (unsloth/Qwen3.5-0.8B), run
# strictly in this order on the FULL test splits:
#   1. Zero-shot   DocRED    (DocRED local test = original dev.json, 998 docs)
#   2. Fine-tuned  DocRED    (train 3 adapters on ALL DocRED train rows, then test)
#   3. Zero-shot   Re-DocRED (test_revised.json, 500 docs)
#   4. Fine-tuned  Re-DocRED (train 3 adapters on ALL Re-DocRED train rows, then test)
#
#   mkdir -p results/qwen08 && nohup bash scripts/run_qwen08.sh > results/qwen08/run.log 2>&1 &
#
# Expected wall time on the GB10 (measured 2026-11-12, FINDINGS.md): training
# ~27 h for all rows x 1 epoch (the config default) on both datasets;
# inference ~2-7 s/doc (zero-shot) per e2e pass. EPOCHS=3 is ~80 h.
#
# Resumable: each step writes its artifact and is skipped when it exists
# (adapters: weights/stage*/qwen08_<ds>/calibration.json; predictions are
# written to *.partial and renamed only when complete). Delete to re-run.
#
# Env overrides (defaults = the full run):
#   MODEL=qwen08         model key in configs/models.yaml
#   EVAL_LIMIT=0         docs per inference run (0 = whole test split)
#   TRAIN_MAX_ROWS=0     rows per stage for training (0 = every train row)
#   EPOCHS=              override the per-stage epochs in lora_params.yaml (default 1)
#   GOLD_ENTS=1          also run gold-entity RE eval (micro/Ign-F1) per run
set -uo pipefail
source "$(dirname "$0")/env.sh"

MODEL="${MODEL:-qwen08}"
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

echo; echo ">>> [1/4] Zero-shot DocRED";   infer_and_eval zeroshot docred
echo; echo ">>> [2/4] Fine-tuned DocRED";  train_all docred;   infer_and_eval finetuned docred
echo; echo ">>> [3/4] Zero-shot Re-DocRED"; infer_and_eval zeroshot redocred
echo; echo ">>> [4/4] Fine-tuned Re-DocRED"; train_all redocred; infer_and_eval finetuned redocred

"$PY" -m src.evaluation.summarize --results-dir "$OUT" > "$OUT/BENCHMARK.md"
cat "$OUT/BENCHMARK.md"
echo "=============================================================="
echo " $MODEL full run — finished $(stamp). Summary: $OUT/BENCHMARK.md"
echo "=============================================================="
