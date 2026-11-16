#!/bin/bash
set -euo pipefail

# Set variables
OUTPUT_DIR="proteinGym/singleMutant/2026-03-09_softSqLoss_16batch_1e-3_cosineannealing_PEFT_contiguous"
GPU=1
LR=1e-3
EPOCHS=100

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Use an array to safely count and iterate through files
FILES=(datasets/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/*.csv)
FILE_NUMBER=${#FILES[@]}
echo "Running SMITEpro on ${OUTPUT_DIR} -- number of files: ${FILE_NUMBER}"


# init file counter
i=0

# loop through the file array
for file in "${FILES[@]}"; do
    i=$((i+1)) 
    base=$(basename "$file" .csv) # Moved to outer loop

    # print current file number out of total
    echo "Processing file ${i} out of ${FILE_NUMBER}: $base"
    outfile="$OUTPUT_DIR/${base}_contiguous_split.csv"

    if [ -f "$outfile" ]; then
        echo "Skipping $base, largest K results already exist"
        continue
    fi
    
    echo "Processing $file"
    python proteinGym/rebuild_eval_pipeline_PEFT_contiguous.py \
        --data_file    "$file" \
        --protein      "$base" \
        --output_dir   "$OUTPUT_DIR" \
        --gpu          "$GPU" \
        --epochs       "$EPOCHS" \
        --lr           "$LR" > "$LOG_DIR/${base}.log" 2>&1 || {
            echo "Warning: Python script failed for $base. Moving to next file."
            continue
        }

done