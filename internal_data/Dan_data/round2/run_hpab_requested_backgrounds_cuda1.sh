#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
RUNNER="$REPO_ROOT/internal_data/internal_peft_multi_background.py"
DATA_FILE="$REPO_ROOT/internal_data/Dan_data/round2/hpab_data2_enumerated.csv"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/internal_data/Dan_data/round2/predict_runs_requested_backgrounds}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
GPU="${GPU:-1}"
LR="${LR:-1e-3}"
STEPS="${STEPS:-100}"
SEED="${SEED:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
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
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --inference_batch_size "$INFERENCE_BATCH_SIZE"
    --background "L303H_A199P_I29V=L303H:A199P:I29V"
    --background "R0094H_S0210D_A0211R_Q0212K=R0094H:S0210D:A0211R:Q0212K"
    --background "A199P_H454Q_H372P=A199P:H454Q:H372P"
)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting HPAB requested-background run on cuda:$GPU with seed $SEED"
"${CMD[@]}" 2>&1 | tee "$LOG_ROOT/hpab_requested_backgrounds.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished HPAB requested-background run"
