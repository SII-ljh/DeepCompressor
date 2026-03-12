#!/bin/bash
# Training script for Qwen3-1.7B with Q=512 on 8 GPUs

set -e

echo "========================================================================"
echo "Deep Compressor QA Training - Qwen3-1.7B, Q=512, 8 GPUs"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

OUTPUT_DIR="outputs/qa_qwen1.7b_q512_8gpu"
DATA_PATH="data/qa_large_train.json"
EVAL_DATA_PATH="data/qa_large_dev.json"

if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data not found at $DATA_PATH"
    echo "Please run: python scripts/prepare_large_qa_data.py"
    exit 1
fi

if [ ! -f "$EVAL_DATA_PATH" ]; then
    echo "Error: Eval data not found at $EVAL_DATA_PATH"
    exit 1
fi

echo "Configuration:"
echo "  Model:                Qwen3-1.7B (hidden=2048, layers=28)"
echo "  Q value:              512"
echo "  GPUs:                 8"
echo "  Batch size (per GPU): 12"
echo "  Gradient accum:       2"
echo "  Effective batch:      $((8 * 12 * 2))"
echo "  Max steps:            10000 (no early stopping, select best checkpoint)"
echo "  Learning rate:        1e-4"
echo "  Output dir:           $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"
export WANDB_MODE=offline

accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision bf16 \
    -m deep_compressor.train \
    --config configs/qa_qwen1.7b_q512_8gpu.yaml \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --max_eval_samples 5000 \
    --wandb --wandb_project deep-compressor --wandb_offline \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Qwen3-1.7B Q=512 training completed successfully!"
else
    echo "Qwen3-1.7B Q=512 training failed with exit code $EXIT_CODE"
fi
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
