#!/usr/bin/env bash
# Reproduce thesis plots. Run from the repository root.
# Full pipeline: see src/scripts/run_pipeline.sh

set -euo pipefail

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python src/plot.py \
    --data_file "data/Dataset.csv" \
    --output_dir "figure/" \
    --model_number 2 \
    --model "BERTopic" \
    --save \
    --avg
