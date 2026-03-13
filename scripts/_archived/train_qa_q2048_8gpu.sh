#!/bin/bash
# Training script for Q=2048 with 8 GPUs (Compression ratio 2:1, early stopping)

set -e  # Exit on error

echo "========================================================================"
echo "Deep Compressor QA Training - Q=2048 (Compression 2:1, 8 GPUs, early stopping)"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# Configuration
Q_VALUE=2048
OUTPUT_DIR="outputs/qa_q2048_8gpu"

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
BATCH_SIZE=5            # Per GPU (reduced from 10 due to larger Q)
GRAD_ACCUM=4            # Increased accum to compensate smaller batch
# Effective batch size = 8 GPUs × 5 batch × 4 accum = 160
MAX_STEPS=10000
WARMUP_STEPS=500
LEARNING_RATE=1e-4
EVAL_EVERY=756
SAVE_EVERY=3024

echo "Configuration:"
echo "  Q value:              $Q_VALUE (压缩比 2:1，接近无损压缩)"
echo "  GPUs:                 8"
echo "  Batch size (per GPU): $BATCH_SIZE"
echo "  Gradient accum:       $GRAD_ACCUM"
echo "  Effective batch:      $((8 * BATCH_SIZE * GRAD_ACCUM))"
echo "  Max steps:            $MAX_STEPS (early stopping patience=5)"
echo "  Learning rate:        $LEARNING_RATE"
echo "  Grad checkpointing:  enabled (Q=2048显存压力大)"
echo "  Output dir:           $OUTPUT_DIR"
echo ""
echo "预期性能提升："
echo "  - Q=1024: EM ~15-20%, F1 ~0.30-0.40 (压缩比4:1)"
echo "  - Q=2048: EM ~25-35%, F1 ~0.45-0.55 (压缩比2:1) ← 当前配置"
echo ""
echo "Starting training..."
echo ""

# Run training with accelerate
export WANDB_MODE=offline

accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision bf16 \
    -m deep_compressor.train \
    --config configs/qa_q2048_8gpu.yaml \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --max_eval_samples 5000 \
    --wandb --wandb_project deep-compressor --wandb_offline \
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
