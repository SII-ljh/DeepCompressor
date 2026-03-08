#!/bin/bash
# 测试单个训练任务是否能正常启动

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    测试训练启动（Q=64, GPU 0）                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 检查数据
echo -e "${YELLOW}[1/5] 检查数据文件${NC}"
if [ -f "data/ntp_train.jsonl" ]; then
    echo -e "${GREEN}✓ data/ntp_train.jsonl 存在${NC}"
elif [ -f "data/ntp_tiny.jsonl" ]; then
    echo -e "${YELLOW}⚠ 使用 ntp_tiny.jsonl 进行测试${NC}"
    DATA_PATH="data/ntp_tiny.jsonl"
else
    echo -e "${RED}✗ 训练数据不存在${NC}"
    echo "请先运行: python scripts/prepare_data.py"
    exit 1
fi

DATA_PATH=${DATA_PATH:-"data/ntp_train.jsonl"}
echo ""

# 检查配置
echo -e "${YELLOW}[2/5] 检查配置文件${NC}"
if [ -f "configs/stage1_q64.yaml" ]; then
    echo -e "${GREEN}✓ configs/stage1_q64.yaml 存在${NC}"
else
    echo -e "${RED}✗ 配置文件不存在${NC}"
    exit 1
fi
echo ""

# 检查GPU
echo -e "${YELLOW}[3/5] 检查GPU可用性${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}✓ 检测到 $GPU_COUNT 个GPU${NC}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv | head -2
else
    echo -e "${RED}✗ nvidia-smi 未找到${NC}"
    exit 1
fi
echo ""

# 检查Python环境
echo -e "${YELLOW}[4/5] 检查Python环境${NC}"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python环境正常${NC}"
else
    echo -e "${RED}✗ Python环境有问题${NC}"
    exit 1
fi
echo ""

# 启动测试训练
echo -e "${YELLOW}[5/5] 启动测试训练（前台运行，10步后会停止）${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "命令："
echo "CUDA_VISIBLE_DEVICES=0 python -m deep_compressor.train \\"
echo "    --config configs/stage1_q64.yaml \\"
echo "    --data_path $DATA_PATH \\"
echo "    --stage 1 \\"
echo "    --max_train_samples 10"
echo ""
echo -e "${YELLOW}如果看到训练loss输出，说明启动成功！${NC}"
echo -e "${YELLOW}按 Ctrl+C 可以随时停止${NC}"
echo ""
read -p "按回车键开始测试..."

# 运行测试
CUDA_VISIBLE_DEVICES=0 python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path "$DATA_PATH" \
    --stage 1 \
    --max_train_samples 10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ 测试完成！如果上面看到了训练输出，说明环境配置正确${NC}"
echo ""
echo "下一步："
echo "  1. 启动8卡并行训练: bash scripts/train_parallel_8gpu.sh"
echo "  2. 查看训练状态: bash scripts/check_training_status.sh"
echo ""
