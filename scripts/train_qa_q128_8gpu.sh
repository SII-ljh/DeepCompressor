#!/bin/bash
# Training script for Q=128 with 8 GPUs (Compression ratio 32:1, early stopping)

set -e  # Exit on error

echo "========================================================================"
echo "Deep Compressor QA Training - Q=128 (Compression 32:1, 8 GPUs, early stopping)"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# Configuration
Q_VALUE=128
OUTPUT_DIR="outputs/qa_q128_8gpu"

# Data paths
DATA_PATH="data/qa_large_train.json"
EVAL_DATA_PATH="data/qa_large_dev.json"

# Check data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data not found at $DATA_PATH"
    echo "Please run: python scripts/prepare_large_qa_data.py"
    exit 1
fi

if [ ! -f "$EVAL_DATA_PATH" ]; then
    echo "Error: Eval data not found at $EVAL_DATA_PATH"
    echo "Please run: python scripts/prepare_large_qa_data.py"
    exit 1
fi

# Training hyperparameters (optimized for 8 GPUs)
BATCH_SIZE=30           # Per GPU batch size
GRAD_ACCUM=2            # Gradient accumulation steps
# Effective batch size = 8 GPUs × 30 batch × 2 accum = 480
MAX_STEPS=10000
WARMUP_STEPS=500
LEARNING_RATE=1e-4
EVAL_EVERY=252
SAVE_EVERY=1008

echo "Configuration:"
echo "  Q value:              $Q_VALUE (压缩比 32:1)"
echo "  GPUs:                 8"
echo "  Batch size (per GPU): $BATCH_SIZE"
echo "  Gradient accum:       $GRAD_ACCUM"
echo "  Effective batch:      $((8 * BATCH_SIZE * GRAD_ACCUM))"
echo "  Max steps:            $MAX_STEPS (early stopping patience=5)"
echo "  Learning rate:        $LEARNING_RATE"
echo "  Output dir:           $OUTPUT_DIR"
echo ""
echo "Starting training..."
echo ""

# Run training with accelerate
export WANDB_MODE=disabled

accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision bf16 \
    -m deep_compressor.train \
    --config configs/qa_q128_8gpu.yaml \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --max_eval_samples 5000 \
    --stage 2 \
    2>&1 | tee "${OUTPUT_DIR}_training.log"

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
fi
echo "End time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: ${OUTPUT_DIR}_training.log"
echo "========================================================================"

exit $EXIT_CODE
