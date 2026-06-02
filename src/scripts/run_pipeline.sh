#!/usr/bin/env bash
# Full analysis pipeline: load data → tokenize → analyze
# Run from the repository root:  bash src/scripts/run_pipeline.sh

set -euo pipefail

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "=== Step 1: Load and merge data ==="
python src/pipeline/load_data.py --data_dir data

echo "=== Step 2: Tokenize (requires CKIP_TAGGER model) ==="
python src/pipeline/tokenize.py --data_dir data

echo "=== Step 3a: Segment-level analysis (BERTopic + LDA) ==="
python src/pipeline/analyze_segment.py \
    --data_file data/Manifesto_Dataset.csv \
    --output_dir output \
    --figure_dir figure \
    --version 1 \
    --save

echo "=== Step 3b: Manifesto-level analysis (LDA) ==="
python src/pipeline/analyze_manifesto.py \
    --data_file data/Manifesto_Dataset_Origin.csv \
    --output_dir output \
    --figure_dir figure \
    --version 1 \
    --save

echo "=== Step 4: Generate plots ==="
bash src/scripts/run_plots.sh

echo "Pipeline complete."
