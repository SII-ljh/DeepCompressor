#!/bin/bash
# Train all Q values sequentially with 8 GPUs
# Runs Q=64, 128, 256, 512 one after another
# All models trained for 2.5 epochs

set -e  # Exit on error

echo "========================================================================"
echo "Deep Compressor - Train All Q Values (8 GPUs)"
echo "========================================================================"
echo "This script will train 4 models sequentially (2.5 epochs each):"
echo "  1. Q=64   ( 9,450 steps, ~2 hours)"
echo "  2. Q=128  (12,600 steps, ~3 hours)"
echo "  3. Q=256  (18,900 steps, ~4.5 hours)"
echo "  4. Q=512  (37,800 steps, ~8 hours)"
echo ""
echo "Total estimated time: ~17.5 hours"
echo "All models will see the same amount of data (2.5 epochs)"
echo "========================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if data exists
DATA_PATH="data/qa_large_train.json"
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data not found at $DATA_PATH"
    echo "Please run: python scripts/prepare_large_qa_data.py"
    exit 1
fi

echo "Starting sequential training..."
echo ""

# Train Q=64
echo "========================================="
echo "[1/4] Training Q=64 (9,450 steps)..."
echo "========================================="
bash "${SCRIPT_DIR}/train_qa_q64_8gpu.sh"
if [ $? -ne 0 ]; then
    echo "✗ Q=64 training failed. Stopping."
    exit 1
fi
echo "✓ Q=64 completed"
echo ""

# Train Q=128
echo "========================================="
echo "[2/4] Training Q=128 (12,600 steps)..."
echo "========================================="
bash "${SCRIPT_DIR}/train_qa_q128_8gpu.sh"
if [ $? -ne 0 ]; then
    echo "✗ Q=128 training failed. Stopping."
    exit 1
fi
echo "✓ Q=128 completed"
echo ""

# Train Q=256
echo "========================================="
echo "[3/4] Training Q=256 (18,900 steps)..."
echo "========================================="
bash "${SCRIPT_DIR}/train_qa_q256_8gpu.sh"
if [ $? -ne 0 ]; then
    echo "✗ Q=256 training failed. Stopping."
    exit 1
fi
echo "✓ Q=256 completed"
echo ""

# Train Q=512
echo "========================================="
echo "[4/4] Training Q=512 (37,800 steps)..."
echo "========================================="
bash "${SCRIPT_DIR}/train_qa_q512_8gpu.sh"
if [ $? -ne 0 ]; then
    echo "✗ Q=512 training failed. Stopping."
    exit 1
fi
echo "✓ Q=512 completed"
echo ""

echo "========================================================================"
echo "✓ All training completed successfully!"
echo "========================================================================"
echo ""
echo "Training Summary:"
echo "  Q=64:  9,450 steps (2.5 epochs)"
echo "  Q=128: 12,600 steps (2.5 epochs)"
echo "  Q=256: 18,900 steps (2.5 epochs)"
echo "  Q=512: 37,800 steps (2.5 epochs)"
echo ""
echo "Output directories:"
echo "  - outputs/qa_q64_8gpu/"
echo "  - outputs/qa_q128_8gpu/"
echo "  - outputs/qa_q256_8gpu/"
echo "  - outputs/qa_q512_8gpu/"
echo ""
echo "Next step: Evaluate all models"
echo "  python scripts/evaluate_all_checkpoints.py \\"
echo "      --eval_data data/qa_large_dev.json \\"
echo "      --stage 2 \\"
echo "      --output results/qa_8gpu_results.csv"
echo "========================================================================"
