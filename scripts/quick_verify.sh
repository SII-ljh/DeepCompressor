#!/bin/bash
# 快速验证：只评估，不训练
# 验证eval loss是否从24.18降到~3.0

echo "验证eval loss修复（只运行评估，1分钟内完成）"
echo ""

accelerate launch --multi_gpu --num_processes 8 \
    -m deep_compressor.train \
    --config configs/qa_q256_8gpu.yaml \
    --data_path data/qa_large_train.json \
    --eval_data_path data/qa_large_dev.json \
    --stage 2 \
    --resume_from outputs/qa_q256_8gpu/checkpoint-final \
    --max_train_samples 16 \
    --max_eval_samples 256 \
    2>&1 | tee quick_verify.log

echo ""
echo "============================================"
echo "检查结果："
grep "\[QA\]" quick_verify.log | tail -1
grep "\[QA EVAL\]" quick_verify.log | tail -1
echo ""
echo "✅ 如果eval loss从24.18降到~3.0，则bug已修复"
echo "⚠️  EM/F1仍然低是正常的（需要更多训练）"
echo "============================================"
