#!/usr/bin/env bash
# Reproduce thesis plots from a completed result dataset.
# Run from the repository root:  bash src/scripts/run_plots.sh
# Requires data/Dataset.csv with PORK_* columns already computed.

set -euo pipefail

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

DATA_FILE="${1:-data/Dataset.csv}"

for party in 0 1 2; do
    for model_number in 1 2 3; do
        python src/plot.py \
            --data_file "$DATA_FILE" \
            --model_number "$model_number" \
            --model "BERTopic" \
            --party "$party" \
            --save \
            --avg
    done
done

echo "Plots saved to figure/"
