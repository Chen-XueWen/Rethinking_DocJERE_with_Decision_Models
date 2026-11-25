#!/usr/bin/env bash
# THE end-of-day batch entry point. Launch and walk away:
#
#   nohup bash scripts/run_overnight.sh > results/overnight.log 2>&1 &
#
# or in the foreground: bash scripts/run_overnight.sh
#
# It smoke-checks first (fails fast if the env broke), then runs the full
# benchmark sweep. Everything is resumable — re-running skips finished steps.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=============================================================="
echo " JevDocRED overnight batch — started $(date)"
echo "=============================================================="

echo; echo ">>> Step 1/2: smoke test (fail-fast validation)"
if ! bash scripts/run_smoke.sh; then
  echo "!!! SMOKE TEST FAILED — aborting the batch. See output above."
  exit 1
fi

echo; echo ">>> Step 2/2: full benchmark sweep"
# EVAL_LIMIT=100 docs/run by default; export EVAL_LIMIT=0 beforehand for full splits.
bash scripts/run_benchmark.sh

echo "=============================================================="
echo " Batch finished $(date). Summary: results/BENCHMARK.md"
echo " Failures (if any): results/logs/failures.log"
echo "=============================================================="
