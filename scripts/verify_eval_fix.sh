#!/bin/bash
# 快速验证eval修复是否生效

echo "=========================================="
echo "验证评估loss修复"
echo "=========================================="

# 使用已有checkpoint，只运行1步训练+1次评估
accelerate launch --multi_gpu --num_processes 8 \
    -m deep_compressor.train \
    --config configs/qa_q256_8gpu.yaml \
    --data_path data/qa_large_train.json \
    --eval_data_path data/qa_large_dev.json \
    --stage 2 \
    --resume_from outputs/qa_q256_8gpu/checkpoint-final \
    --max_train_samples 8 \
    --max_eval_samples 256 2>&1 | tee verify_eval.log

echo ""
echo "=========================================="
echo "检查结果："
echo "=========================================="
echo ""
echo "训练loss (应该 ~2.8):"
grep "\[QA\]" verify_eval.log | tail -1

echo ""
echo "评估loss (修复后应该 ~2.8，修复前 ~22-24):"
grep "\[QA EVAL\]" verify_eval.log | tail -1

echo ""
echo "如果评估loss接近训练loss，则修复成功！"
