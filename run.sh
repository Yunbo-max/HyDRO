#!/bin/bash
# Quick start script for HyDRO on Linux
# Usage: ./run.sh [dataset] [reduction_rate]
# Example: ./run.sh cora 0.5

set -e

DATASET=${1:-cora}
RATE=${2:-0.5}

echo "Running HyDRO on ${DATASET} with reduction rate ${RATE}"

cd run
python train_all.py -M hydro -D "$DATASET" -R "$RATE"
