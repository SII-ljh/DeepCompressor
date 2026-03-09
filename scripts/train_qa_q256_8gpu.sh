#!/bin/bash
# Training script for Q=256 with 8 GPUs
# Full-scale QA training on large dataset (~800K samples)

set -e  # Exit on error

echo "========================================================================"
echo "Deep Compressor QA Training - Q=256 (8 GPUs)"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# Configuration
Q_VALUE=256
OUTPUT_DIR="outputs/qa_q256_8gpu"
WANDB_PROJECT="deep-compressor-qa"
WANDB_RUN_NAME="qa_q256_8gpu_full"

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
BATCH_SIZE=4           # Per GPU batch size (smaller due to large Q)
GRAD_ACCUM=4           # Gradient accumulation steps
# Effective batch size = 8 GPUs × 4 batch × 4 accum = 128
MAX_STEPS=50000
WARMUP_STEPS=2000
LEARNING_RATE=1e-4
EVAL_EVERY=1000
SAVE_EVERY=5000

echo "Configuration:"
echo "  Q value:              $Q_VALUE"
echo "  GPUs:                 8"
echo "  Batch size (per GPU): $BATCH_SIZE"
echo "  Gradient accum:       $GRAD_ACCUM"
echo "  Effective batch:      $((8 * BATCH_SIZE * GRAD_ACCUM))"
echo "  Max steps:            $MAX_STEPS"
echo "  Learning rate:        $LEARNING_RATE"
echo "  Output dir:           $OUTPUT_DIR"
echo ""
echo "Starting training..."
echo ""

# Run training with accelerate
accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision bf16 \
    -m deep_compressor.train \
    --config configs/qa_q256_8gpu.yaml \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --stage 2 \
    --wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
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
