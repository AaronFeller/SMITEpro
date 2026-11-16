#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
"$PYTHON" -m pip install --no-deps --target baselines/.deps -r baselines/requirements.txt
PYTHONPATH="$PWD/baselines/.deps${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -c 'import torch, numpy, pandas, sklearn, gpytorch; print("CPU baseline dependencies ready")'