#!/usr/bin/env bash
# Shared environment setup for all scripts. Source this, don't execute it.
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT"
# Keep the leaked ~/.local site-packages out of the dedicated env:
export PYTHONNOUSERSITE=1
export HF_XET_HIGH_PERFORMANCE=1
PY="$HOME/anaconda3/envs/jevdocred/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: conda env 'jevdocred' not found at $PY" >&2
  exit 1
fi
export PY
cd "$REPO_ROOT"
