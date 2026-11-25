#!/usr/bin/env bash
# Quick end-to-end validation (~15 min): tiny train on stage-3 rows (packed
# gate questions + 97-way relation rows), save, reload the adapter, run the
# full 3-stage pipeline zero-shot on 2 documents, evaluate.
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "=== 1/3 tiny train + calibrate + save (stage 3 mixed rows, 48 rows, 1 epoch) ==="
"$PY" - <<'EOF'
from src.training.train_common import train_stage
from src.common.utils import ROOT
d = ROOT / "data/processed/stage3_re"
train_stage("smoke_stage3", "qwen",
            d / "docred_train.jsonl", d / "docred_dev.jsonl",
            ROOT / "weights/smoke_test/qwen", epochs=1, max_rows=48)
print("SMOKE TRAIN OK")
EOF

echo "=== 2/3 reload adapter + predict one gate row and one relation row ==="
"$PY" - <<'EOF'
from src.models.decision_engine import DecisionEngine
from src.common.utils import ROOT, read_jsonl
rows = read_jsonl(ROOT / "data/processed/stage3_re/docred_dev.jsonl")
gate = next(r for r in rows if r["meta"]["kind"] == "gate")
rel = next(r for r in rows if r["meta"]["kind"] == "relation")
eng = DecisionEngine("qwen", adapter_path=ROOT / "weights/smoke_test/qwen")
out = eng.predict(gate["state"], gate["questions"])
assert set(out) == set(gate["questions"]), "every packed question must be answered"
print(f"gate row: {len(out)} questions answered in one pass")
out = eng.predict(rel["state"], rel["questions"])
print("relation row: gold", rel["gold"]["relation"], "| pred", out["relation"]["answer"])
print("SMOKE RELOAD+PREDICT OK")
EOF

echo "=== 3/3 zero-shot pipeline on 2 docs + strict eval ==="
"$PY" -m src.inference.pipeline --model qwen --zero-shot \
  --dataset docred --split test --mode e2e --limit 2 \
  --out results/preds_smoke.jsonl
"$PY" -m src.evaluation.strict_e2e_eval --preds results/preds_smoke.jsonl \
  --dataset docred --split test
echo "=== SMOKE TEST PASSED ==="
