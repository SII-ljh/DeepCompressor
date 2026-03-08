#!/bin/bash
# 依次训练所有Q值（8卡DDP，基于Q64的配置）
# 每个训练完成后自动开始下一个

set -e  # 遇到错误停止

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              依次训练所有Q值（16, 32, 64, 128, 256）                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 检查数据
if [ ! -f "data/ntp_train.jsonl" ]; then
    echo "错误: data/ntp_train.jsonl 不存在"
    exit 1
fi

# Q值列表
Q_VALUES=(16 32 64 128 256)

echo "配置信息："
echo "  数据: data/ntp_train.jsonl (全量)"
echo "  GPU: 0-7 (8卡DDP)"
echo "  Q值: ${Q_VALUES[@]}"
echo "  步数: 50,000 steps per Q"
echo "  模式: 离线wandb"
echo ""

read -p "按回车键开始训练（Ctrl+C取消）..."
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 依次训练每个Q值
for q in "${Q_VALUES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "开始训练 Q=${q} ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    Q_START=$(date +%s)

    # 运行训练
    bash scripts/train_q${q}_8gpu.sh

    Q_END=$(date +%s)
    Q_ELAPSED=$((Q_END - Q_START))
    Q_HOURS=$((Q_ELAPSED / 3600))
    Q_MINS=$(((Q_ELAPSED % 3600) / 60))

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✓ Q=${q} 训练完成！用时: ${Q_HOURS}小时${Q_MINS}分钟"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # 短暂休息，让GPU冷却
    if [ "$q" != "256" ]; then
        echo "等待30秒后开始下一个训练..."
        sleep 30
        echo ""
    fi
done

# 计算总时间
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINS=$(((TOTAL_ELAPSED % 3600) / 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                          ✅ 所有训练完成！                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "总用时: ${TOTAL_HOURS}小时${TOTAL_MINS}分钟"
echo ""
echo "训练结果："
for q in "${Q_VALUES[@]}"; do
    if [ -f "outputs/stage1_q${q}/checkpoint-final/trainable_weights.pt" ]; then
        size=$(ls -lh "outputs/stage1_q${q}/checkpoint-final/trainable_weights.pt" | awk '{print $5}')
        echo "  ✓ Q=${q}: outputs/stage1_q${q}/checkpoint-final/ (${size})"
    else
        echo "  ✗ Q=${q}: checkpoint未找到"
    fi
done
echo ""
echo "下一步："
echo "  1. 评估所有模型:"
echo "     bash scripts/evaluate_all_models.sh"
echo ""
echo "  2. 查看wandb离线数据:"
echo "     ls wandb/offline-run-*"
echo ""
echo "  3. 同步wandb到云端（有网时）:"
echo "     wandb sync wandb/offline-run-*/"
echo ""
