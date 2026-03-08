#!/bin/bash
# 完整的Stage 1不同Q值实验流程
# 一键执行：准备数据 -> 训练所有模型 -> 评估所有checkpoint

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Stage 1 不同Q值实验 - 完整流程${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# ============================================================
# 步骤1: 准备数据
# ============================================================
echo -e "${YELLOW}[1/3] 检查训练数据${NC}"
echo "--------------------------------------"

if [ -f "data/ntp_train.jsonl" ]; then
    echo -e "${GREEN}✓ 训练数据已存在: data/ntp_train.jsonl${NC}"
    echo -e "${GREEN}  （数据会在加载时自动截断到512 tokens）${NC}"
else
    echo -e "${RED}✗ 错误: data/ntp_train.jsonl 不存在${NC}"
    echo "请先运行: python scripts/prepare_data.py"
    exit 1
fi

echo ""

# ============================================================
# 步骤2: 批量训练
# ============================================================
echo -e "${YELLOW}[2/3] 批量训练（Q = 16, 32, 64, 128, 256）${NC}"
echo "--------------------------------------"

read -p "选择训练模式 - [1] 全部训练  [2] 选择Q值  [3] 跳过训练: " -n 1 -r choice
echo

case $choice in
    1)
        echo -e "${GREEN}开始训练所有Q值...${NC}"
        python scripts/train_stage1_varying_q.py
        ;;
    2)
        read -p "输入Q值（用逗号分隔，如: 16,32,64）: " q_values
        echo -e "${GREEN}开始训练 Q = $q_values${NC}"
        python scripts/train_stage1_varying_q.py --q_values "$q_values"
        ;;
    3)
        echo -e "${YELLOW}跳过训练步骤${NC}"
        ;;
    *)
        echo -e "${RED}无效选择，跳过训练${NC}"
        ;;
esac

echo ""

# ============================================================
# 步骤3: 评估所有checkpoint
# ============================================================
echo -e "${YELLOW}[3/3] 评估所有训练好的模型${NC}"
echo "--------------------------------------"

# 检查是否有checkpoint
checkpoint_count=$(find outputs -type d -name "stage1_q*" 2>/dev/null | wc -l)

if [ "$checkpoint_count" -eq 0 ]; then
    echo -e "${RED}✗ 未找到任何checkpoint${NC}"
    echo "请先完成训练步骤"
    exit 1
fi

echo -e "${GREEN}找到 $checkpoint_count 个checkpoint${NC}"

read -p "是否评估所有checkpoint？(Y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    # 创建结果目录
    mkdir -p results

    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_csv="results/stage1_varying_q_${timestamp}.csv"

    echo -e "${GREEN}开始评估...${NC}"
    python scripts/evaluate_all_checkpoints.py \
        --eval_data data/ntp_train.jsonl \
        --stage 1 \
        --show_samples 5 \
        --output "$output_csv"

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}实验完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "结果已保存到: ${GREEN}$output_csv${NC}"
    echo ""

    # 如果有Python和pandas，显示简单的结果表格
    if command -v python3 &> /dev/null; then
        python3 << 'EOF'
import sys
import csv
from pathlib import Path

csv_files = sorted(Path("results").glob("stage1_varying_q_*.csv"))
if csv_files:
    latest = csv_files[-1]
    print("\n快速结果预览:")
    print("="*80)
    with open(latest) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            # 打印表头
            print(f"{'Q':<8} {'Perplexity':<15} {'Loss':<15}")
            print("-"*80)
            # 打印每一行
            for row in sorted(rows, key=lambda x: int(x.get('q_value', -1))):
                q = row.get('q_value', 'N/A')
                ppl = float(row.get('perplexity', 0))
                loss = float(row.get('loss', 0))
                print(f"{q:<8} {ppl:<15.4f} {loss:<15.4f}")
    print("="*80)
EOF
    fi
else
    echo -e "${YELLOW}跳过评估步骤${NC}"
fi

echo ""
echo -e "${GREEN}实验流程说明:${NC}"
echo "1. 查看详细指南: cat scripts/STAGE1_VARYING_Q_EXPERIMENTS.md"
echo "2. 查看wandb曲线: https://wandb.ai"
echo "3. 分析结果并选择最佳Q值"
echo "4. 使用最佳checkpoint进行Stage 2训练"
echo ""
