#!/usr/bin/env bash
# Usage: bash scripts/submit_array.sh profile|reproduce [--dry-run] [sbatch options]
set -euo pipefail
kind="${1:-}"
if [[ "$kind" != profile && "$kind" != reproduce ]]; then
    echo "Usage: bash scripts/submit_array.sh profile|reproduce [--dry-run] [sbatch options]" >&2
    exit 1
fi
shift
dry_run=0
extra=()
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=1 ;;
        --array*|-a|-a[0-9]*) echo "Array indices are computed from the matrix; use ARRAY_CONCURRENCY=1 or 2." >&2; exit 1 ;;
        *) extra+=("$argument") ;;
    esac
done
cd "${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source scripts/load_cluster_env.sh
PYTHON_BIN="${PYTHON_BIN:-python3}"
concurrency="${ARRAY_CONCURRENCY:-2}"
if [[ "$concurrency" != 1 && "$concurrency" != 2 ]]; then
    echo "ARRAY_CONCURRENCY must be 1 or 2 for your two-GPU limit" >&2; exit 1
fi
manifest=$("$PYTHON_BIN" scripts/array_jobs.py create --kind "$kind" --concurrency "$concurrency")
count=$("$PYTHON_BIN" scripts/array_jobs.py count --manifest "$manifest")
export OURO_ARRAY_MANIFEST="$manifest"
export OURO_ARRAY_REPO="$PWD"
export OURO_ARRAY_PYTHON
OURO_ARRAY_PYTHON=$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')
submit_args=(--export=ALL --chdir="$REPO_DIR" --array="0-$((count - 1))%$concurrency" --job-name="ouro_${kind}_array")
if [[ "$kind" == reproduce ]]; then submit_args+=(--time=72:00:00); fi
if [[ -n "${PILOT_ACCOUNT:-}" ]]; then submit_args+=(--account="$PILOT_ACCOUNT"); fi
if [[ -n "${PILOT_PARTITION:-}" ]]; then submit_args+=(--partition="$PILOT_PARTITION"); fi
mkdir -p "$REPO_DIR/logs"
echo "Array manifest: $manifest"
"$PYTHON_BIN" scripts/array_jobs.py describe --manifest "$manifest"
if (( dry_run )); then
    printf 'Submission command: '
    printf '%q ' sbatch "${submit_args[@]}" "${extra[@]}" "$REPO_DIR/scripts/array_task.sh"
    printf '\n'
    exit 0
fi
exec sbatch "${submit_args[@]}" "${extra[@]}" "$REPO_DIR/scripts/array_task.sh"
