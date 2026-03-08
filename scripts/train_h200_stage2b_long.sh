#!/bin/bash
# ── Stage 2b: QA Fine-tuning on LONG documents (up to 2048 tokens) ──
#
# Usage:
#   bash scripts/train_h200_stage2b_long.sh
#
# Prerequisites: Stage 2a must be completed first!
#   Checkpoint: outputs/h200_stage2a/checkpoint-final/

set -euo pipefail

# ── Sanity checks ──
if [ ! -f "data/qa_train.json" ]; then
    echo "ERROR: data/qa_train.json not found."
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

if [ ! -f "outputs/h200_stage2a/checkpoint-final/trainable_weights.pt" ]; then
    echo "ERROR: Stage 2a checkpoint not found at outputs/h200_stage2a/checkpoint-final/"
    echo "You must complete Stage 2a training first!"
    echo "Run: bash scripts/train_h200_stage2a_short.sh"
    exit 1
fi

echo "✓ All prerequisites satisfied"
echo "  - Stage 2a checkpoint: outputs/h200_stage2a/checkpoint-final/"
echo "  - Full QA train: data/qa_train.json (263K samples)"
echo "  - Full QA dev: data/qa_dev.json (35K samples)"
echo ""
echo "⚠️  max_doc_tokens extended to 2048 (covers 62% of QA data)"
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
    --config configs/h200_stage2b_long.yaml \
    --data_path data/qa_train.json \
    --eval_data_path data/qa_dev.json \
    --resume_from outputs/h200_stage2a/checkpoint-final \
    --stage 2 \
    --max_eval_samples 1000 \
    --wandb \
    --wandb_project deep-compressor \
    --wandb_run_name "stage2b-8xH200-0.6B-qa-long"
