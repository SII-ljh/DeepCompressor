#!/bin/bash
# ── Stage 2 QA: Optimized Single-Stage Long-Sequence Training ──
#
# This replaces the three-stage training (2a → 2b → 2c) with a single
# efficient run on long documents (4096 tokens).
#
# Why this is better:
#   1. Simpler: One training run instead of three
#   2. Faster: No checkpoint loading/saving between stages
#   3. Better: Perceiver is designed for long sequences
#   4. More coverage: 4096 tokens covers 92% of QA data
#
# Usage:
#   bash scripts/train_h200_stage2_long_optimized.sh
#
# Prerequisites:
#   1. conda activate deep_compressor
#   2. Stage 1 checkpoint: outputs/h200_stage1/checkpoint-final/
#   3. Full QA data: data/qa_train.json and data/qa_dev.json
#   4. Model: models/Qwen3-0.6B/

set -euo pipefail

# ── Sanity checks ──
if [ ! -f "data/qa_train.json" ]; then
    echo "ERROR: data/qa_train.json not found."
    echo "Run: python scripts/prepare_data.py"
    exit 1
fi

if [ ! -f "data/qa_dev.json" ]; then
    echo "ERROR: data/qa_dev.json not found."
    exit 1
fi

if [ ! -d "models/Qwen3-0.6B" ]; then
    echo "ERROR: models/Qwen3-0.6B/ not found."
    exit 1
fi

if [ ! -f "outputs/h200_stage1/checkpoint-final/trainable_weights.pt" ]; then
    echo "ERROR: Stage 1 checkpoint not found."
    echo "Run: bash scripts/train_h200_stage1.sh"
    exit 1
fi

echo "✓ All prerequisites satisfied"
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Stage 2: Single-Stage Long-Sequence Training"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Document length: 4096 tokens (8× longer than baseline)"
echo "  - Data coverage: ~92% of QA dataset"
echo "  - Training: Single stage (no 2a/2b/2c)"
echo "  - Batch size: 2 per GPU × 4 accum = 8 effective"
echo "  - Total effective batch: 64 (8 GPU × 8)"
echo "  - Training steps: 10K (~2.4 epochs)"
echo "  - Estimated time: 6-8 hours on 8×H200"
echo ""
echo "Why this works:"
echo "  - Perceiver cross-attention: O(queries × doc_len) - linear!"
echo "  - Perceiver self-attention: O(queries²) - constant!"
echo "  - Qwen encoder: O(doc_len²) but Flash Attention helps"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5
echo ""

# ── NCCL tuning for H200 ──
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export TOKENIZERS_PARALLELISM=false

# ── Offline mode ──
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# ── Launch ──
accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision bf16 \
    -m deep_compressor.train \
    --config configs/h200_stage2_long_optimized.yaml \
    --data_path data/qa_train.json \
    --eval_data_path data/qa_dev.json \
    --resume_from outputs/h200_stage1/checkpoint-final \
    --stage 2 \
    --max_eval_samples 1000 \
    --wandb \
    --wandb_project deep-compressor \
    --wandb_run_name "stage2-long-optimized-4096tok"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Training complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Checkpoint: outputs/h200_stage2_long/checkpoint-final/"
echo ""
echo "Next steps:"
echo "  1. Evaluate on test set:"
echo "     python scripts/benchmark.py \\"
echo "       --checkpoint outputs/h200_stage2_long/checkpoint-final/trainable_weights.pt \\"
echo "       --eval_data data/qa_test.json"
echo ""
