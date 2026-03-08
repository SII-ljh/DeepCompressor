#!/bin/bash
# Q=128 单独训练脚本（8卡DDP）

echo "=========================================="
echo "启动 Q=128 训练（8卡DDP）"
echo "=========================================="
echo ""

# 检查数据
if [ ! -f "data/ntp_train.jsonl" ]; then
    echo "错误: data/ntp_train.jsonl 不存在"
    exit 1
fi

echo "配置: configs/stage1_q128.yaml"
echo "数据: data/ntp_train.jsonl"
echo "GPU: 0-7 (8卡)"
echo "输出: outputs/stage1_q128/"
echo ""

# 创建日志目录
mkdir -p logs

echo "开始训练..."
echo ""

# 8卡DDP训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    -m deep_compressor.train \
    --config configs/stage1_q128.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1 \
    --wandb \
    --wandb_project deep-compressor \
    2>&1 | tee logs/q128_$(date +%Y%m%d_%H%M%S).log
