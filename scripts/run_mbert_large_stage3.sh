#!/usr/bin/env bash
# Re-run ONLY stage 3 (relation extraction) for ModernBERT-large, reusing the
# already-trained stage 1 (NER) and stage 2 (coref) checkpoints.
#
# Why: the first stage-3 checkpoints were trained at head_max_len=256, which
# cut the 97-way relation question so every entity pair got the identical
# input -> zero triples (FINDINGS.md 2026-11-16). head_max_len is now 768.
#
# Order:
#   1. DocRED:    train stage 3 -> e2e test + eval -> gold-entity RE test + eval
#   2. Re-DocRED: train stage 3 -> e2e test + eval -> gold-entity RE test + eval
#
#   mkdir -p results/mbert_large && nohup bash scripts/run_mbert_large_stage3.sh > results/mbert_large/run_stage3.log 2>&1 &
#
# Expected wall time (GB10, measured on the first run): stage-3 training
# ~3 h DocRED + ~7 h Re-DocRED (items are ~10% longer at the new budget),
# plus e2e and gold-entity inference on both full test sets.
#
# Resumable: finished steps are skipped (delete an artifact to redo it).
set -uo pipefail
source "$(dirname "$0")/env.sh"

MODEL=mbert_large
OUT="results/$MODEL"
LOG="$OUT/logs"
mkdir -p "$LOG"

stamp() { date '+%F %T'; }
fail() { echo "!!! [$(stamp)] FAILED: $1 (see $LOG/$1.log)" | tee -a "$LOG/failures.log"; exit 1; }

run_step() {  # run_step <artifact> <log-name> <cmd...>; aborts the script on failure
  local artifact="$1" log="$2"; shift 2
  if [ -s "$artifact" ]; then echo "--- [$(stamp)] skip (exists): $artifact"; return 0; fi
  echo "--- [$(stamp)] $log: $*"
  "$@" > "$LOG/$log.log" 2>&1 || fail "$log"
  grep -E "^(timing summary|wrote|saved|\[stage)" "$LOG/$log.log" | tail -3
}

# --- preflight: stage 1-2 checkpoints must exist, base must carry the new budget
for ds in docred redocred; do
  for stage in stage1_ner stage2_coref; do
    [ -s "weights/$stage/${MODEL}_${ds}/calibration.json" ] \
      || { echo "!!! missing weights/$stage/${MODEL}_${ds}; run scripts/run_mbert_large.sh instead"; exit 1; }
  done
done
"$PY" -c "
import json, sys
h = json.load(open('weights/base/$MODEL/rl_agent_config.json'))['head_max_len']
sys.exit(0 if h >= 768 else f'!!! base head_max_len is {h}, expected >= 768')" || exit 1

run_ds() {  # run_ds <dataset>
  local ds="$1"
  run_step "weights/stage3_re/${MODEL}_${ds}/calibration.json" "train_stage3_re_${ds}" \
    "$PY" -m src.training.train_stage3_re --model "$MODEL" --dataset "$ds" --max-rows 0

  local P="$OUT/preds_${MODEL}_finetuned_${ds}_test_e2e.jsonl"
  run_step "$P" "finetuned_${ds}_e2e" \
    "$PY" -m src.inference.pipeline --model "$MODEL" --dataset "$ds" --split test --mode e2e --out "$P"
  run_step "${P%.jsonl}.strict_e2e.json" "finetuned_${ds}_e2e_eval" \
    "$PY" -m src.evaluation.strict_e2e_eval --preds "$P" --dataset "$ds" --split test
  run_step "${P%.jsonl}.isolated_eval.json" "finetuned_${ds}_e2e_isolated" \
    "$PY" -m src.evaluation.isolated_eval --preds "$P" --dataset "$ds" --split test --tasks ner coref

  local G="$OUT/preds_${MODEL}_finetuned_${ds}_test_gold-ents.jsonl"
  run_step "$G" "finetuned_${ds}_gold" \
    "$PY" -m src.inference.pipeline --model "$MODEL" --dataset "$ds" --split test --mode gold-ents --out "$G"
  run_step "${G%.jsonl}.isolated_eval.json" "finetuned_${ds}_gold_eval" \
    "$PY" -m src.evaluation.isolated_eval --preds "$G" --dataset "$ds" --split test --tasks re

  # sanity: the bug this run fixes produced zero triples everywhere
  "$PY" -c "
import json
n = sum(len(json.loads(l)['triples']) for l in open('$G'))
print(f'[sanity] $ds gold-ents: {n} predicted triples')
if n == 0: print('!!! still ZERO triples on gold entities: stage 3 is not working')"
}

echo "=============================================================="
echo " $MODEL stage-3 re-run — started $(stamp)"
echo "=============================================================="
echo; echo ">>> [1/2] DocRED";    run_ds docred
echo; echo ">>> [2/2] Re-DocRED"; run_ds redocred

"$PY" -m src.evaluation.summarize --results-dir "$OUT" > "$OUT/BENCHMARK.md"
cat "$OUT/BENCHMARK.md"
echo "=============================================================="
echo " $MODEL stage-3 re-run — finished $(stamp). Summary: $OUT/BENCHMARK.md"
echo "=============================================================="
