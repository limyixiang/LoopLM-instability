#!/bin/bash
# One frozen experiment cell per array task; submit via submit_array.sh.
#SBATCH --job-name=ouro_array
#SBATCH --partition=gpu-long
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

set -euo pipefail
: "${OURO_ARRAY_MANIFEST:?Submit with scripts/submit_array.sh}"
: "${OURO_ARRAY_REPO:?Missing saved repository directory}"
: "${OURO_ARRAY_PYTHON:?Missing saved Python interpreter}"
: "${SLURM_ARRAY_TASK_ID:?This script requires a SLURM array task}"
cd "$OURO_ARRAY_REPO"
source scripts/load_cluster_env.sh  # Load any site modules before frozen dispatch.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec "$OURO_ARRAY_PYTHON" "$OURO_ARRAY_REPO/scripts/array_jobs.py" run \
    --manifest "$OURO_ARRAY_MANIFEST" --task-id "$SLURM_ARRAY_TASK_ID"
