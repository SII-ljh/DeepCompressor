#!/bin/bash
# ── Stage 2b: QA Fine-tuning on MEDIUM documents (512-2048 tokens) ──
#
# Usage:
#   bash scripts/train_h200_stage2b_medium.sh
#
# Prerequisites: Stage 2a must be completed first!

set -euo pipefail

# ── Sanity checks ──
if [ ! -f "data/qa_train_filtered_512_2048.json" ]; then
    echo "ERROR: Medium-length filtered data not found."
    echo "Run: python scripts/filter_qa_by_length.py --min_tokens 512 --max_tokens 2048"
    exit 1
fi

if [ ! -d "models/Qwen3-0.6B" ]; then
    echo "ERROR: models/Qwen3-0.6B/ not found."
    exit 1
fi

if [ ! -f "outputs/h200_stage2a/checkpoint-final/trainable_weights.pt" ]; then
    echo "ERROR: Stage 2a checkpoint not found at outputs/h200_stage2a/checkpoint-final/"
    echo "Run: bash scripts/train_h200_stage2a_short.sh"
    exit 1
fi

echo "✓ All prerequisites satisfied"
echo "  - Stage 2a checkpoint: outputs/h200_stage2a/checkpoint-final/"
echo "  - Filtered QA train (512-2048): data/qa_train_filtered_512_2048.json"
echo "  - Filtered QA dev (512-2048): data/qa_dev_filtered_512_2048.json"
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
    --config configs/h200_stage2b_medium.yaml \
    --data_path data/qa_train_filtered_512_2048.json \
    --eval_data_path data/qa_dev_filtered_512_2048.json \
    --resume_from outputs/h200_stage2a/checkpoint-final \
    --stage 2 \
    --max_eval_samples 500 \
    --wandb \
    --wandb_project deep-compressor \
    --wandb_run_name "stage2b-8xH200-0.6B-qa-medium"
