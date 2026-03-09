#!/bin/bash
# 评估所有训练好的模型

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                      评估所有训练好的模型                                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "工作目录: $(pwd)"
echo ""

# 检查是否有checkpoint
CHECKPOINT_COUNT=$(find outputs -type d -name "stage1_q*" 2>/dev/null | wc -l)

if [ "$CHECKPOINT_COUNT" -eq 0 ]; then
    echo "错误: 未找到任何checkpoint"
    echo "请先完成训练"
    exit 1
fi

echo "找到 $CHECKPOINT_COUNT 个checkpoint"
echo ""

# 创建结果目录
mkdir -p results

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_CSV="results/stage1_varying_q_${TIMESTAMP}.csv"

echo "开始评估..."
echo "输出文件: ${OUTPUT_CSV}"
echo ""

# 运行评估
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train.jsonl \
    --stage 1 \
    --show_samples 5 \
    --output "${OUTPUT_CSV}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 评估完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "结果已保存到: ${OUTPUT_CSV}"
echo ""

# 显示CSV内容（如果存在）
if [ -f "${OUTPUT_CSV}" ]; then
    echo "快速预览:"
    cat "${OUTPUT_CSV}"
    echo ""
fi
