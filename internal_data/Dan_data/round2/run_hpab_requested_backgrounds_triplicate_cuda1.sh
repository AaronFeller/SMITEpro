#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RUN_SCRIPT="$REPO_ROOT/internal_data/Dan_data/round2/run_hpab_requested_backgrounds_cuda1.sh"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-$REPO_ROOT/internal_data/Dan_data/round2}"
GPU="${GPU:-1}"
LR="${LR:-1e-3}"
STEPS="${STEPS:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-${BATCH_SIZE:-32}}"

run_seed() {
    local seed="$1"
    local suffix="$2"
    local output_root="$BASE_OUTPUT_ROOT/predict_runs_requested_backgrounds_seed${suffix}"
    local log_root="$output_root/logs"

    SEED="$seed" \
    OUTPUT_ROOT="$output_root" \
    LOG_ROOT="$log_root" \
    GPU="$GPU" \
    LR="$LR" \
    STEPS="$STEPS" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    INFERENCE_BATCH_SIZE="$INFERENCE_BATCH_SIZE" \
    bash "$RUN_SCRIPT"
}

run_seed 101 1
run_seed 202 2
run_seed 303 3