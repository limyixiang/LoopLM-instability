#!/bin/bash
# Bounded profiling job. Existing reproduce.sh resources and runs are unchanged.
#SBATCH --job-name=ouro_profile
#SBATCH --partition=gpu-long
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100-96:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

set -euo pipefail
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
source scripts/load_cluster_env.sh
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${TRAIN_DATA_DIR:?Set TRAIN_DATA_DIR to the prepared training directory}"
: "${VAL_DATA_DIR:?Set VAL_DATA_DIR to the separate prepared validation directory}"
# Space-separated cases are intentional. No shell evaluation of their contents.
read -r -a profile_cases <<< "${PROFILE_CASES:-512:1}"
read -r -a profile_depths <<< "${PROFILE_DEPTHS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec "$PYTHON_BIN" -u scripts/profile_reproduction.py \
    --model-config "${PROFILE_MODEL_CONFIG:-ouro_1_4b_repro}" \
    --data-dir "$TRAIN_DATA_DIR" --val-data-dir "$VAL_DATA_DIR" \
    --output-root "${PROFILE_ROOT:-${OUTPUT_ROOT:-runs/reproduction}/profiles}" \
    --cases "${profile_cases[@]}" --depths "${profile_depths[@]}" \
    --effective-tokens "${PROFILE_EFFECTIVE_TOKENS:-16384}" \
    --warmup-updates "${PROFILE_WARMUP_UPDATES:-2}" --measure-updates "${PROFILE_MEASURE_UPDATES:-4}" \
    --budget-seconds "${PROFILE_BUDGET_SECONDS:-3600}" \
    --case-timeout-seconds "${PROFILE_CASE_TIMEOUT_SECONDS:-1800}" "$@"
