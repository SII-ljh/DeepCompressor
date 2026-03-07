#!/bin/bash
# ── Stage 1 NTP Pretraining on 8×H200 ──
#
# Usage:
#   bash scripts/train_h200_stage1.sh
#
# Prerequisites:
#   1. conda activate deep_compressor
#   2. pip install -r requirements.txt
#   3. python scripts/prepare_data.py          # full dataset (~1-2h)
#   4. Verify: ls data/ntp_train.jsonl
#   5. Verify: ls models/Qwen3-0.6B/           # local model

set -euo pipefail

# ── Sanity check ──
if [ ! -f "data/ntp_train.jsonl" ]; then
    echo "ERROR: data/ntp_train.jsonl not found. Run: python scripts/prepare_data.py"
    exit 1
fi

if [ ! -d "models/Qwen3-0.6B" ]; then
    echo "ERROR: models/Qwen3-0.6B/ not found. Download the model first."
    exit 1
fi

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
    --config configs/h200_stage1.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1 \
    --wandb \
    --wandb_project deep-compressor \
    --wandb_run_name "stage1-8xH200-0.6B-full"
