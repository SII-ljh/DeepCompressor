#!/bin/bash
# ── Stage 2 QA Fine-tuning + Distillation on 8×H200 ──
#
# Usage:
#   bash scripts/train_h200_stage2.sh
#
# Prerequisites:
#   1. conda activate deep_compressor
#   2. pip install -r requirements.txt
#   3. Stage 1 checkpoint exists at outputs/h200_stage1/checkpoint-final/
#   4. QA data exists: data/qa_train.json and data/qa_dev.json
#   5. Model exists: models/Qwen3-0.6B/

set -euo pipefail

# ── Sanity checks ──
if [ ! -f "data/qa_train.json" ]; then
    echo "ERROR: data/qa_train.json not found. Run: python scripts/prepare_data.py"
    exit 1
fi

if [ ! -f "data/qa_dev.json" ]; then
    echo "ERROR: data/qa_dev.json not found. Run: python scripts/prepare_data.py"
    exit 1
fi

if [ ! -d "models/Qwen3-0.6B" ]; then
    echo "ERROR: models/Qwen3-0.6B/ not found. Download the model first."
    exit 1
fi

if [ ! -f "outputs/h200_stage1/checkpoint-final/trainable_weights.pt" ]; then
    echo "ERROR: Stage 1 checkpoint not found at outputs/h200_stage1/checkpoint-final/"
    echo "You must complete Stage 1 training first."
    exit 1
fi

echo "✓ All prerequisites satisfied"
echo "  - Stage 1 checkpoint: outputs/h200_stage1/checkpoint-final/"
echo "  - QA train data: data/qa_train.json"
echo "  - QA dev data: data/qa_dev.json"
echo ""

# ── NCCL tuning for H200 ──
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export TOKENIZERS_PARALLELISM=false

# ── Offline mode (no internet access on cluster) ──
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# ── Launch ──
accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision bf16 \
    -m deep_compressor.train \
    --config configs/h200_stage2.yaml \
    --data_path data/qa_train.json \
    --eval_data_path data/qa_dev.json \
    --resume_from outputs/h200_stage1/checkpoint-final \
    --stage 2 \
    --max_eval_samples 1000 \
    --wandb \
    --wandb_project deep-compressor \
    --wandb_run_name "stage2-8xH200-0.6B-qa"
