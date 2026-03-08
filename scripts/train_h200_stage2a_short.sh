#!/bin/bash
# ── Stage 2a: QA Fine-tuning on SHORT documents (< 512 tokens) ──
#
# Usage:
#   bash scripts/train_h200_stage2a_short.sh
#
# This is the first step of the two-stage length adaptation strategy.
# After this completes, run train_h200_stage2b_long.sh

set -euo pipefail

# ── Sanity checks ──
if [ ! -f "data/qa_train_filtered_512.json" ]; then
    echo "ERROR: Filtered QA data not found."
    echo "Run: python scripts/filter_qa_by_length.py --max_tokens 512"
    exit 1
fi

if [ ! -d "models/Qwen3-0.6B" ]; then
    echo "ERROR: models/Qwen3-0.6B/ not found."
    exit 1
fi

if [ ! -f "outputs/h200_stage1/checkpoint-final/trainable_weights.pt" ]; then
    echo "ERROR: Stage 1 checkpoint not found at outputs/h200_stage1/checkpoint-final/"
    exit 1
fi

echo "✓ All prerequisites satisfied"
echo "  - Stage 1 checkpoint: outputs/h200_stage1/checkpoint-final/"
echo "  - Filtered QA train: data/qa_train_filtered_512.json"
echo "  - Filtered QA dev: data/qa_dev_filtered_512.json"
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
    --config configs/h200_stage2a_short.yaml \
    --data_path data/qa_train_filtered_512.json \
    --eval_data_path data/qa_dev_filtered_512.json \
    --resume_from outputs/h200_stage1/checkpoint-final \
    --stage 2 \
    --max_eval_samples 500 \
    --wandb \
    --wandb_project deep-compressor \
    --wandb_run_name "stage2a-8xH200-0.6B-qa-short"
