#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
RUNNER="$REPO_ROOT/internal_data/internal_peft_multi_background.py"
DATA_FILE="$REPO_ROOT/internal_data/Dan_data/round2/hpab_data2_enumerated.csv"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/internal_data/Dan_data/round2/predict_runs_enumerated_singlePref}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
GPU="${GPU:-1}"
LR="${LR:-1e-3}"
STEPS="${STEPS:-100}"
SEED="${SEED:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-${BATCH_SIZE:-32}}"

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
    --inference_batch_size "$INFERENCE_BATCH_SIZE"
    --background "DE_ML_combination=I157L:N159H:T201A:S210L:A211R:Q212L:S237C"
    --background "DE_only=R0094H:S0210D:A0211R:Q0212K:A0156S:I0157L:T0398A"
    --background "ML_only=I29V:A199P:S237C:D354K:H372P:H454Q"
)

if [[ -n "$TRAIN_BATCH_SIZE" ]]; then
    CMD+=(--train_batch_size "$TRAIN_BATCH_SIZE")
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting HPAB multi-background run on cuda:$GPU with seed $SEED"
"${CMD[@]}" 2>&1 | tee "$LOG_ROOT/hpab_multi_background.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished HPAB multi-background run"