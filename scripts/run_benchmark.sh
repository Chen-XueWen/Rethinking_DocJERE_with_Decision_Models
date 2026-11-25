#!/usr/bin/env bash
# Phase 3: full experimental sweep (EXP-01..07) + evaluation + summary table.
#
# This is the end-of-day batch script. It is resumable: every step writes its
# artifact and is skipped if that artifact already exists (delete to re-run).
#
# Env overrides:
#   EVAL_LIMIT   cap on #docs per inference run (default 100; 0 = all docs)
#   RUNS         model:dataset pairs to fine-tune
#                (default "qwen:docred qwen:redocred llama:redocred laya:redocred")
#   SKIP_ZS=1    skip zero-shot runs
#   SKIP_FT=1    skip fine-tuned runs (training + inference)
set -uo pipefail
source "$(dirname "$0")/env.sh"

# Measured on the GB10: ~5 min/doc for full e2e with the 4B backbone, so 25
# docs ≈ 2h per e2e run. Raise EVAL_LIMIT (or set 0) only with time to spare.
EVAL_LIMIT="${EVAL_LIMIT:-25}"
# Experiment matrix (ImplementationPlan §9):
#   EXP-03 qwen/docred   EXP-04 qwen/redocred
#   EXP-05 llama/redocred   EXP-07 laya/redocred (16-bit LoRA)
# Default = the two headline runs; set RUNS to the full matrix when there is
# GPU time to spare (each run trains 3 adapters: hours per run even with the
# max_train_rows cap in configs/lora_params.yaml).
RUNS="${RUNS:-qwen:docred qwen:redocred}"
FULL_RUNS="qwen:docred qwen:redocred llama:redocred laya:redocred"
[ "${RUNS}" = "all" ] && RUNS="$FULL_RUNS"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR" results

LIMIT_ARG=()
[ "$EVAL_LIMIT" != "0" ] && LIMIT_ARG=(--limit "$EVAL_LIMIT")

run_step() {  # run_step <artifact> <log-name> <cmd...>
  local artifact="$1"; local log="$2"; shift 2
  if [ -s "$artifact" ]; then
    echo "--- skip (exists): $artifact"
    return 0
  fi
  echo "--- running: $* (log: $LOG_DIR/$log.log)"
  if ! "$@" 2>&1 | tee "$LOG_DIR/$log.log"; then
    echo "!!! FAILED: $log (continuing with remaining steps)" | tee -a "$LOG_DIR/failures.log"
    return 1
  fi
}

# ---------------------------------------------------------------- data prep
if [ ! -s data/processed/stage3_re/redocred_train.jsonl ]; then
  bash scripts/run_prep.sh
fi

# ------------------------------------------------- EXP-01/02: clef-flash ZS
# ------------------------------------------------- EXP-06:    laya ZS
if [ "${SKIP_ZS:-0}" != "1" ]; then
  for zs_model in clef-flash laya; do
    for ds in docred redocred; do
      P="results/preds_${zs_model}_zeroshot_${ds}_test_e2e.jsonl"
      run_step "$P" "zs_${zs_model}_${ds}" \
        "$PY" -m src.inference.pipeline --model "$zs_model" --zero-shot \
          --dataset "$ds" --split test --mode e2e "${LIMIT_ARG[@]}" --out "$P"
      [ -s "$P" ] && run_step "${P%.jsonl}.strict_e2e.json" "zs_eval_${zs_model}_${ds}" \
        "$PY" -m src.evaluation.strict_e2e_eval --preds "$P" --dataset "$ds" --split test
    done
  done
fi

# --------------------------------- EXP-03/04/05/07: fine-tuned backbones
if [ "${SKIP_FT:-0}" != "1" ]; then
  for run in $RUNS; do
      model="${run%%:*}"
      ds="${run##*:}"
      # 1) train all three adapters (skipped if adapters already saved)
      if [ ! -s "weights/stage3_re/${model}_${ds}/calibration.json" ]; then
        bash scripts/run_train_all.sh "$model" "$ds" || {
          echo "!!! training failed for $model/$ds" | tee -a "$LOG_DIR/failures.log"
          continue
        }
      else
        echo "--- skip training (weights exist): ${model}_${ds}"
      fi

      # 2) strict end-to-end inference + eval on the labelled test split
      P="results/preds_${model}_finetuned_${ds}_test_e2e.jsonl"
      run_step "$P" "ft_e2e_${model}_${ds}" \
        "$PY" -m src.inference.pipeline --model "$model" \
          --dataset "$ds" --split test --mode e2e "${LIMIT_ARG[@]}" --out "$P"
      [ -s "$P" ] && run_step "${P%.jsonl}.strict_e2e.json" "ft_e2e_eval_${model}_${ds}" \
        "$PY" -m src.evaluation.strict_e2e_eval --preds "$P" --dataset "$ds" --split test

      # 3) gold-entity upper-bound RE eval (DocRED micro-F1 / Ign-F1)
      G="results/preds_${model}_finetuned_${ds}_test_gold-ents.jsonl"
      run_step "$G" "ft_gold_${model}_${ds}" \
        "$PY" -m src.inference.pipeline --model "$model" \
          --dataset "$ds" --split test --mode gold-ents "${LIMIT_ARG[@]}" --out "$G"
      [ -s "$G" ] && run_step "${G%.jsonl}.isolated_eval.json" "ft_gold_eval_${model}_${ds}" \
        "$PY" -m src.evaluation.isolated_eval --preds "$G" --dataset "$ds" --split test
  done
fi

# ---------------------------------------------------------------- summary
"$PY" -m src.evaluation.summarize --results-dir results | tee results/BENCHMARK.md
echo
echo "=== Benchmark sweep finished. Summary: results/BENCHMARK.md ==="
[ -s "$LOG_DIR/failures.log" ] && { echo "Some steps FAILED:"; cat "$LOG_DIR/failures.log"; }
exit 0
