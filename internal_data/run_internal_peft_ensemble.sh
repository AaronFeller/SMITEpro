#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

DATA_FILE="${1:?Usage: run_internal_peft_ensemble.sh TRAIN_CSV PROTEIN OUTPUT_DIR [CANDIDATE_CSV]}"
PROTEIN="${2:?Usage: run_internal_peft_ensemble.sh TRAIN_CSV PROTEIN OUTPUT_DIR [CANDIDATE_CSV]}"
OUTPUT_DIR="${3:?Usage: run_internal_peft_ensemble.sh TRAIN_CSV PROTEIN OUTPUT_DIR [CANDIDATE_CSV]}"
CANDIDATE_FILE="${4:-}"

GPU="${GPU:-0}"
LR="${LR:-1e-3}"
STEPS="${STEPS:-250}"
MODEL="${MODEL:-esmc_300m}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-${BATCH_SIZE:-16}}"
PREDICT_FROM_MUTANT="${PREDICT_FROM_MUTANT:-}"
PREDICT_WT_SEQUENCE="${PREDICT_WT_SEQUENCE:-}"

CMD=(
    "$PYTHON_BIN" "$REPO_ROOT/internal_data/internal_peft_ensemble.py"
    --data_file "$DATA_FILE"
    --protein "$PROTEIN"
    --output_dir "$OUTPUT_DIR"
    --gpu "$GPU"
    --lr "$LR"
    --steps "$STEPS"
    --model "$MODEL"
    --inference_batch_size "$INFERENCE_BATCH_SIZE"
)

if [[ -n "$TRAIN_BATCH_SIZE" ]]; then
    CMD+=(--train_batch_size "$TRAIN_BATCH_SIZE")
fi

if [[ -n "$PREDICT_FROM_MUTANT" ]]; then
    CMD+=(--predict_from_mutant "$PREDICT_FROM_MUTANT")
fi

if [[ -n "$PREDICT_WT_SEQUENCE" ]]; then
    CMD+=(--predict_wt_sequence "$PREDICT_WT_SEQUENCE")
fi

if [[ -n "$CANDIDATE_FILE" ]]; then
    CMD+=(--candidate_file "$CANDIDATE_FILE")
fi

"${CMD[@]}"