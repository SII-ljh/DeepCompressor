#!/bin/bash
# Final evaluation script with full eval set (54K samples)
# Use this after training is complete for comprehensive evaluation

set -e

echo "========================================================================"
echo "Final Evaluation - Full Eval Set (54K samples)"
echo "========================================================================"
echo ""

# Model to evaluate
if [ -z "$1" ]; then
    echo "Usage: bash scripts/final_eval_8gpu.sh <checkpoint_dir>"
    echo ""
    echo "Example:"
    echo "  bash scripts/final_eval_8gpu.sh outputs/qa_q128_8gpu/checkpoint-final"
    echo ""
    exit 1
fi

CHECKPOINT_DIR="$1"

# Check checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_DIR"
    exit 1
fi

# Data paths
EVAL_DATA_PATH="data/qa_large_dev.json"

if [ ! -f "$EVAL_DATA_PATH" ]; then
    echo "Error: Eval data not found at $EVAL_DATA_PATH"
    exit 1
fi

# Disable wandb
export WANDB_MODE=disabled

echo "Configuration:"
echo "  Checkpoint:  $CHECKPOINT_DIR"
echo "  Eval data:   $EVAL_DATA_PATH"
echo "  Using:       Full eval set (no sampling)"
echo ""
echo "Starting evaluation..."
echo ""

# Run evaluation
python -m deep_compressor.eval \
    --checkpoint "$CHECKPOINT_DIR" \
    --eval_data "$EVAL_DATA_PATH" \
    --stage 2 \
    --output "results/$(basename $CHECKPOINT_DIR)_full_eval.json"

echo ""
echo "========================================================================"
echo "✓ Evaluation complete!"
echo "Results saved to: results/$(basename $CHECKPOINT_DIR)_full_eval.json"
echo "========================================================================"
