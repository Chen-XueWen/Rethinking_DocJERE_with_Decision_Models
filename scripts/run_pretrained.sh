#!/usr/bin/env bash
# Zero-shot evaluation of a hosted PRE-TRAINED decision model (no fine-tuning),
# through the LiteLLM proxy in .env. Tests generalization: the exact stage-1/2/3
# schemas the fine-tuned models used, sent as-is.
#
#   bash scripts/run_pretrained.sh [model_key]      # default azure_ms_decision_1
#   mkdir -p results/azure_ms_decision_1 && nohup bash scripts/run_pretrained.sh \
#       > results/azure_ms_decision_1/run.log 2>&1 &
#
# Order (full test splits):
#   1. DocRED    e2e (strict E2E + NER/coref eval) -> gold-entity RE (micro/Ign-F1)
#   2. Re-DocRED e2e (strict E2E + NER/coref eval) -> gold-entity RE (micro/Ign-F1)
#
# Rate limits (FINDINGS.md 2026-11-18): LiteLLM allows 1,000 RPM, but the proxy's
# AWS WAF blocks everything for ~2-4 min once a rolling ~5-min window sees
# ~800-1,500 requests, counted across ALL clients of the proxy (including
# Claude Code itself). Paced at 150 RPM by default (configs/pretrained.yaml;
# override with PRETRAINED_RPM=...). A block triggers a 5-min cooldown and the
# run continues. ~160k requests total -> ~17.5 h, ~20 h with occasional blocks.
# Cost: ~$0.04 / M input tokens, ~$12 for all four runs.
#
# Resumable: finished docs are kept in *.partial and skipped on restart;
# finished steps are skipped (delete an artifact to redo it).
set -uo pipefail
source "$(dirname "$0")/env.sh"

MODEL="${1:-azure_ms_decision_1}"
# MODEL="${1:-h2o_lightning_4b}"
# MODEL="${1:-strands_decider_2b}"
OUT="results/$MODEL"
LOG="$OUT/logs"
mkdir -p "$LOG"
[ -f .env ] || { echo "!!! .env with LITELLM_API_BASE / LITELLM_KEY not found"; exit 1; }

stamp() { date '+%F %T'; }
fail() { echo "!!! [$(stamp)] FAILED: $1 (see $LOG/$1.log)" | tee -a "$LOG/failures.log"; exit 1; }

run_step() {  # run_step <artifact> <log-name> <cmd...>; aborts the script on failure
  local artifact="$1" log="$2"; shift 2
  if [ -s "$artifact" ]; then echo "--- [$(stamp)] skip (exists): $artifact"; return 0; fi
  echo "--- [$(stamp)] $log: $*"
  "$@" >> "$LOG/$log.log" 2>&1 || fail "$log"
  grep -E "^(timing summary|wrote|saved|api usage|\[.*WAF)" "$LOG/$log.log" | tail -4
}

run_ds() {  # run_ds <dataset>
  local ds="$1"
  local P="$OUT/preds_${MODEL}_zeroshot_${ds}_test_e2e.jsonl"
  run_step "$P" "${ds}_e2e" \
    "$PY" -m src.pretrained.run_pretrained --model "$MODEL" --dataset "$ds" --split test --mode e2e --out "$P"
  run_step "${P%.jsonl}.strict_e2e.json" "${ds}_e2e_eval" \
    "$PY" -m src.evaluation.strict_e2e_eval --preds "$P" --dataset "$ds" --split test
  run_step "${P%.jsonl}.isolated_eval.json" "${ds}_e2e_isolated" \
    "$PY" -m src.evaluation.isolated_eval --preds "$P" --dataset "$ds" --split test --tasks ner coref

  local G="$OUT/preds_${MODEL}_zeroshot_${ds}_test_gold-ents.jsonl"
  run_step "$G" "${ds}_gold" \
    "$PY" -m src.pretrained.run_pretrained --model "$MODEL" --dataset "$ds" --split test --mode gold-ents --out "$G"
  run_step "${G%.jsonl}.isolated_eval.json" "${ds}_gold_eval" \
    "$PY" -m src.evaluation.isolated_eval --preds "$G" --dataset "$ds" --split test --tasks re
}

echo "=============================================================="
echo " $MODEL zero-shot (pre-trained, no fine-tuning) — started $(stamp)"
echo " pace: ${PRETRAINED_RPM:-from configs/pretrained.yaml} req/min"
echo "=============================================================="
echo; echo ">>> [1/2] DocRED";    run_ds docred
echo; echo ">>> [2/2] Re-DocRED"; run_ds redocred

"$PY" -m src.evaluation.summarize --results-dir "$OUT" > "$OUT/BENCHMARK.md"
cat "$OUT/BENCHMARK.md"
echo "=============================================================="
echo " $MODEL finished $(stamp). Summary: $OUT/BENCHMARK.md"
echo "=============================================================="
