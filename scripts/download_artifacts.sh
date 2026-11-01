#!/usr/bin/env bash
# Run on a preparation/login node. Fetches pinned tokenizer assets and streams
# only the bounded FineWeb-Edu selection; no full-corpus download or GPU use.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/load_cluster_env.sh
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN. Set PYTHON_BIN to your environment's bin/python." >&2
    exit 1
fi
export HF_HOME="${HF_HOME:-${PROJECT:-.}/hf_cache}"
DATA_ROOT="${REPRO_DATA_DIR:-${SCRATCH:-.}/looplm/data/reproduction_fineweb_edu}"

mkdir -p "$HF_HOME"
echo "==> Preparing the bounded pilot corpus at $DATA_ROOT"
"$PYTHON_BIN" scripts/prepare_reproduction_data.py --output-dir "$DATA_ROOT" "$@"

echo "==> Use $DATA_ROOT/train and $DATA_ROOT/validation on the GPU node"
