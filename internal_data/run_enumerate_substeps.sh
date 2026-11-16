#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

INPUT_CSV="${1:?Usage: run_enumerate_substeps.sh INPUT_CSV [OUTPUT_CSV] [DERIVED_ONLY_CSV]}"
OUTPUT_CSV="${2:-}"
DERIVED_ONLY_CSV="${3:-}"

CMD=(
    "$PYTHON_BIN" "$REPO_ROOT/internal_data/enumerate_substeps.py"
    --input_csv "$INPUT_CSV"
)

if [[ -n "$OUTPUT_CSV" ]]; then
    CMD+=(--output_csv "$OUTPUT_CSV")
fi

if [[ -n "$DERIVED_ONLY_CSV" ]]; then
    CMD+=(--derived_only_csv "$DERIVED_ONLY_CSV")
fi

"${CMD[@]}"