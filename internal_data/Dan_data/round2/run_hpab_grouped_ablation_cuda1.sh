#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
RUNNER="$REPO_ROOT/internal_data/dan_grouped_single_ablation.py"
DATA_FILE="$REPO_ROOT/internal_data/Dan_data/round2/hpab_data2_enumerated.csv"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/internal_data/Dan_data/round2/random_family_holdout_seed101}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
GPU="${GPU:-1}"
LR="${LR:-1e-3}"
STEPS="${STEPS:-100}"
SEED="${SEED:-101}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-${BATCH_SIZE:-32}}"
NUM_SPLITS="${NUM_SPLITS:-10}"
HOLDOUT_FAMILIES_PER_SPLIT="${HOLDOUT_FAMILIES_PER_SPLIT:-5}"
MAX_SPLITS="${MAX_SPLITS:-}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

CMD=(
    "$PYTHON_BIN" "$RUNNER"
    --data_file "$DATA_FILE"
    --protein hpab
    --output_root "$OUTPUT_ROOT"
    --gpu "$GPU"
    --lr "$LR"
    --steps "$STEPS"
    --seed "$SEED"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --inference_batch_size "$INFERENCE_BATCH_SIZE"
    --num_splits "$NUM_SPLITS"
    --holdout_families_per_split "$HOLDOUT_FAMILIES_PER_SPLIT"
)

if [[ -n "$MAX_SPLITS" ]]; then
    CMD+=(--max_splits "$MAX_SPLITS")
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting HPAB random family-holdout run on cuda:$GPU with seed $SEED"
"${CMD[@]}" 2>&1 | tee "$LOG_ROOT/hpab_grouped_ablation.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished HPAB random family-holdout run"