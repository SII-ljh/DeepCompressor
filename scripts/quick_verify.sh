#!/bin/bash
# 纯评估：快速验证eval loss修复（不训练）
# 验证eval loss是否从24.18降到~3.0

echo "验证eval loss修复（纯评估，不训练）"
echo ""

accelerate launch --multi_gpu --num_processes 8 \
    scripts/eval_only.py \
    --config configs/qa_q256_8gpu.yaml \
    --checkpoint outputs/qa_q256_8gpu/checkpoint-final \
    --eval_data data/qa_large_dev.json \
    --max_samples 256
