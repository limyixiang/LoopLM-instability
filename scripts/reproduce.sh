#!/bin/bash
# Submit through submit_reproduction.sh (adjust resources on the command line), or run
# inside an existing single-GPU allocation. One process runs all cells serially.
#SBATCH --job-name=ouro_reproduce
#SBATCH --partition=gpu-long
#SBATCH --time=72:00:00
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
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN. Set PYTHON_BIN to your environment's bin/python." >&2
    exit 1
fi
: "${TRAIN_DATA_DIR:?Set TRAIN_DATA_DIR to the prepared training .npy directory}"
: "${VAL_DATA_DIR:?Set VAL_DATA_DIR to a separate validation .npy directory}"
MODE="${MODE:-screen}"
case "$MODE" in
    screen) SEEDS="${SEEDS:-0}"; LRS="${LRS:-1e-4 3e-4}"; MAX_STEPS="${MAX_STEPS:-512}" ;;
    reproduce) SEEDS="${SEEDS:-101 202 303}"; LRS="${LRS:?Set LRS to the LR selected after screening}"; MAX_STEPS="${MAX_STEPS:-3072}" ;;
    *) echo "MODE must be screen or reproduce" >&2; exit 1 ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/reproduction}"
MODEL_CONFIG="${MODEL_CONFIG:-small_repro}"
SEQ_LEN="${SEQ_LEN:-512}"
case "$MODEL_CONFIG" in
    small_repro|ouro_1_4b_repro|depth24_repro) ;;
    *) echo "Use a validated reproduction preset: small_repro, ouro_1_4b_repro, or depth24_repro" >&2; exit 1 ;;
esac
if [[ ! "$SEQ_LEN" =~ ^[1-9][0-9]*$ ]] || (( SEQ_LEN > 4096 )); then
    echo "SEQ_LEN must be an integer from 1 to 4096" >&2; exit 1
fi
if [[ "$MODEL_CONFIG" != small_repro ]]; then
    : "${KEEP_CHECKPOINTS:=1}"
    : "${KEEP_EVENT_CHECKPOINTS:=1}"
fi
# Preserve historical small-run paths. Scale/context experiments have their own
# namespace and still pass strict checkpoint/config/provenance resume checks.
RUN_ROOT="$OUTPUT_ROOT"
if [[ "$MODEL_CONFIG" != small_repro || "$SEQ_LEN" != 512 ]]; then
    RUN_ROOT="$OUTPUT_ROOT/models/$MODEL_CONFIG/seq_$SEQ_LEN"
fi
repack=()
if [[ "${REPACK_STREAM:-0}" == 1 ]]; then repack=(--repack-stream); fi
if [[ ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEPS must be a positive integer" >&2; exit 1
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Install dependencies before the reservation; the job uses this interpreter as-is.
for seed in $SEEDS; do
    for lr in $LRS; do
        for depth in ${DEPTHS:-4 8}; do
            run_dir="$RUN_ROOT/$MODE/seed_${seed}/lr_${lr}/T${depth}"
            mkdir -p "$run_dir"
            resume=()
            if [[ -d "$run_dir/checkpoints" ]]; then
                latest=$(find "$run_dir/checkpoints" -maxdepth 1 -name 'step_*.pt' -print | sort | tail -n 1)
                if [[ -n "$latest" ]]; then
                    completed=${latest##*/step_}
                    completed=${completed%.pt}
                    if (( 10#$completed >= MAX_STEPS )); then
                        echo "Requested observed horizon already reached: $run_dir"
                        continue
                    fi
                    resume=(--resume-from "$latest")
                fi
            fi
            if "$PYTHON_BIN" scripts/train.py --reproduction --device cuda \
                --model-config "$MODEL_CONFIG" --num-recurrent-steps "$depth" \
                --seed "$seed" --data-seed "$seed" \
                --data-dir "$TRAIN_DATA_DIR" --val-data-dir "$VAL_DATA_DIR" \
                --batch-size "${MICRO_BATCH:-2}" --grad-accum-steps "${GRAD_ACCUM:-16}" \
                --seq-len "$SEQ_LEN" --max-steps "$MAX_STEPS" --lr "$lr" --warmup-steps 200 \
                --beta-kl 0.1 --log-every 1 --save-every 100 --keep-checkpoints "${KEEP_CHECKPOINTS:-5}" \
                --keep-event-checkpoints "${KEEP_EVENT_CHECKPOINTS:-3}" \
                --val-every 100 --val-tokens 131072 --gradient-checkpointing \
                --output-dir "$run_dir" "${repack[@]}" "${resume[@]}"; then
                echo "Completed observed horizon: $run_dir"
            else
                code=$?
                if [[ "$code" == 42 ]]; then
                    echo "Numerical event recorded: $run_dir; continuing other cells"
                else
                    exit "$code"
                fi
            fi
        done
    done
done
echo "Review result.json and metrics.jsonl for every cell before choosing further experiments."
