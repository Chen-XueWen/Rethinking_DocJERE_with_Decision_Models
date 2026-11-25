# JEREX with a Jev-Like Model Twist

Document-level joint entity and relation extraction on DocRED/Re-DocRED via
non-autoregressive decision scoring (Unsloth `FastDecisionModel`, Clef-style
decision heads).

## Environment

Everything runs in the dedicated conda env **`jevdocred`** (Python 3.12,
torch cu130, unsloth). All scripts source `scripts/env.sh`, which pins the
interpreter and sets `PYTHONNOUSERSITE=1` (the machine has a leaky
`~/.local/lib/python3.12` that must stay out of this env).

```bash
conda activate jevdocred   # optional; scripts don't need an activated env
```

## Pipeline

| Stage | Task | Formulation (one forward pass answers many questions) | Post-processing |
|---|---|---|---|
| 1 | Mention detection | ≤14 per-token 13-way BIO choices per row | deterministic BIO repair |
| 2 | Coreference | mentions tagged once `[M7]`; ~24–36 binary pair questions per row | complete-linkage clustering (α) |
| 3 | Relation extraction | entities tagged once `[E3]`; packed binary gate rows, then one 97-way row per gate survivor | τ_gate / τ_rel thresholds |

The original one-row-per-pair layout was 5.5 docs/hour and has been removed.
`FINDINGS.md` records why, and also logs every approach that did not work.

Evaluation protocol (per the resolved plan):
- **JEREX protocol**: DocRED `train_annotated` is re-split 90/10 into train/dev
  (seed 3407); the original `dev.json` is the labelled local **test** set.
  Re-DocRED uses its own `*_revised.json` splits.
- **JEREX-strict E2E**: a triple counts iff head and tail entities match a gold
  entity's full mention-span set AND entity type, each gold entity used once.
- Ign-F1 excludes relational facts (name-level) seen in the training annotation.

## Quick start

```bash
# 0) fast validation that everything still works (~10-15 min)
bash scripts/run_smoke.sh

# 1) build all processed datasets (already done if data/processed is populated)
bash scripts/run_prep.sh

# 2) train the three adapters for one backbone+dataset
bash scripts/run_train_all.sh qwen docred

# 3) full benchmark sweep — THE end-of-day batch (resumable)
bash scripts/run_benchmark.sh                      # default: 100 docs/run eval cap
EVAL_LIMIT=0 bash scripts/run_benchmark.sh         # full splits (much slower)
RUNS="qwen:docred" SKIP_ZS=1 bash scripts/run_benchmark.sh   # single experiment
```

The sweep is resumable: each step writes its artifact (weights, predictions,
eval JSON) and is skipped on re-run if the artifact exists; failures are
logged to `results/logs/failures.log` and don't abort the sweep. The final
table lands in `results/BENCHMARK.md`.

Single pieces:

```bash
source scripts/env.sh
$PY -m src.inference.pipeline --model qwen --dataset docred --split test --mode e2e --limit 5
$PY -m src.evaluation.strict_e2e_eval --preds results/preds_... .jsonl --dataset docred --split test
$PY -m src.evaluation.summarize --results-dir results
```

## Repository layout

See `ImplementationPlan.md` §4. Key entry points:
- `src/data_prep/make_stage{1,2,3}_*.py` — JSONL builders (Unsloth row schema)
- `src/training/train_stage{1,2,3}_*.py` — LoRA + decision-head training
- `src/inference/pipeline.py` — end-to-end orchestrator (`--mode e2e|gold-ents`)
- `src/evaluation/{isolated_eval,strict_e2e_eval,summarize}.py`
- `scripts/run_benchmark.sh` — the end-of-day batch sweep

## Known caveats

- `Cloudflare/clef-flash` is 9B (not 2.5B): zero-shot runs load it 4-bit via
  Unsloth. `convaiinnovations/laya` zero-shot is expected to be near-chance
  (its card says it is a base to specialise); it is included for the latency
  comparison and additionally fine-tuned (EXP-07 analogue) via the same path.
- Laya has tight input/option budgets; if its runs fail on the 97-way relation
  question, re-run with the relation question chunked (see plan notes).
- Unsloth #13160: an evaluate() pass runs before each training as a warm-up.
- Multi-adapter hot-swap also swaps the per-stage decision head; if the head
  can't be located, the manager falls back to full reload per stage.
