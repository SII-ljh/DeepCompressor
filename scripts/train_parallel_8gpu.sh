#!/bin/bash
# 8卡并行训练不同Q值的Stage 1模型
# 使用tmux管理多个训练任务，每个任务使用不同的GPU

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}8卡并行训练 - Stage 1 不同Q值${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查tmux是否安装
if ! command -v tmux &> /dev/null; then
    echo -e "${YELLOW}警告: tmux未安装，将使用nohup后台运行${NC}"
    USE_TMUX=false
else
    USE_TMUX=true
    echo -e "${GREEN}使用tmux管理训练任务${NC}"
fi

# 检查数据文件
if [ ! -f "data/ntp_train.jsonl" ]; then
    echo "错误: data/ntp_train.jsonl 不存在"
    echo "请先运行: python scripts/prepare_data.py"
    exit 1
fi

# 训练配置
DATA_PATH="data/ntp_train.jsonl"
SESSION_NAME="stage1_training"

echo ""
echo -e "${YELLOW}GPU分配策略:${NC}"
echo "  GPU 0-1: Q=16  (2卡)"
echo "  GPU 2-3: Q=32  (2卡)"
echo "  GPU 4-5: Q=64  (2卡)"
echo "  GPU 6:   Q=128 (1卡)"
echo "  GPU 7:   Q=256 (1卡)"
echo ""

read -p "是否继续？(Y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    exit 0
fi

if [ "$USE_TMUX" = true ]; then
    # 使用tmux
    echo -e "${GREEN}创建tmux会话: $SESSION_NAME${NC}"

    # 创建新的tmux会话
    tmux new-session -d -s "$SESSION_NAME" -n "q16"

    # 窗口1: Q=16 (GPU 0-1)
    echo "启动 Q=16 训练 (GPU 0-1)..."
    tmux send-keys -t "$SESSION_NAME:q16" "CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 -m deep_compressor.train --config configs/stage1_q16.yaml --data_path $DATA_PATH --stage 1 --wandb --wandb_project deep-compressor 2>&1 | tee logs/q16_$(date +%Y%m%d_%H%M%S).log" C-m

    # 窗口2: Q=32 (GPU 2-3)
    echo "启动 Q=32 训练 (GPU 2-3)..."
    tmux new-window -t "$SESSION_NAME" -n "q32"
    tmux send-keys -t "$SESSION_NAME:q32" "CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi_gpu --num_processes 2 -m deep_compressor.train --config configs/stage1_q32.yaml --data_path $DATA_PATH --stage 1 --wandb --wandb_project deep-compressor 2>&1 | tee logs/q32_$(date +%Y%m%d_%H%M%S).log" C-m

    # 窗口3: Q=64 (GPU 4-5)
    echo "启动 Q=64 训练 (GPU 4-5)..."
    tmux new-window -t "$SESSION_NAME" -n "q64"
    tmux send-keys -t "$SESSION_NAME:q64" "CUDA_VISIBLE_DEVICES=4,5 accelerate launch --multi_gpu --num_processes 2 -m deep_compressor.train --config configs/stage1_q64.yaml --data_path $DATA_PATH --stage 1 --wandb --wandb_project deep-compressor 2>&1 | tee logs/q64_$(date +%Y%m%d_%H%M%S).log" C-m

    # 窗口4: Q=128 (GPU 6)
    echo "启动 Q=128 训练 (GPU 6)..."
    tmux new-window -t "$SESSION_NAME" -n "q128"
    tmux send-keys -t "$SESSION_NAME:q128" "CUDA_VISIBLE_DEVICES=6 python -m deep_compressor.train --config configs/stage1_q128.yaml --data_path $DATA_PATH --stage 1 --wandb --wandb_project deep-compressor 2>&1 | tee logs/q128_$(date +%Y%m%d_%H%M%S).log" C-m

    # 窗口5: Q=256 (GPU 7)
    echo "启动 Q=256 训练 (GPU 7)..."
    tmux new-window -t "$SESSION_NAME" -n "q256"
    tmux send-keys -t "$SESSION_NAME:q256" "CUDA_VISIBLE_DEVICES=7 python -m deep_compressor.train --config configs/stage1_q256.yaml --data_path $DATA_PATH --stage 1 --wandb --wandb_project deep-compressor 2>&1 | tee logs/q256_$(date +%Y%m%d_%H%M%S).log" C-m

    echo ""
    echo -e "${GREEN}等待训练进程启动...${NC}"
    sleep 5

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}检查训练状态${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # 检查GPU使用情况
    echo "GPU使用情况："
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -8

    echo ""
    echo "训练进程："
    ps aux | grep "deep_compressor.train" | grep -v grep | wc -l | xargs -I {} echo "  找到 {} 个训练进程"

    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}查看训练进度的方法：${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo "1. 进入tmux会话（推荐）："
    echo "   tmux attach -t $SESSION_NAME"
    echo "   按 Ctrl+B 然后按 0-4 切换窗口"
    echo "   按 Ctrl+B d 退出（训练继续）"
    echo ""
    echo "2. 实时查看日志："
    echo "   tail -f logs/q64_*.log"
    echo ""
    echo "3. 监控GPU："
    echo "   watch -n 1 nvidia-smi"
    echo ""
    echo "4. 检查进程："
    echo "   ps aux | grep deep_compressor.train"
    echo ""
    echo "停止所有训练："
    echo "   tmux kill-session -t $SESSION_NAME"
    echo ""

else
    # 使用nohup后台运行
    mkdir -p logs

    echo -e "${GREEN}启动后台训练任务...${NC}"

    # Q=16 (GPU 0-1)
    CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch --multi_gpu --num_processes 2 \
        -m deep_compressor.train \
        --config configs/stage1_q16.yaml \
        --data_path $DATA_PATH --stage 1 \
        --wandb --wandb_project deep-compressor \
        > logs/q16_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    echo "Q=16 启动 (PID: $!)"

    # Q=32 (GPU 2-3)
    CUDA_VISIBLE_DEVICES=2,3 nohup accelerate launch --multi_gpu --num_processes 2 \
        -m deep_compressor.train \
        --config configs/stage1_q32.yaml \
        --data_path $DATA_PATH --stage 1 \
        --wandb --wandb_project deep-compressor \
        > logs/q32_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    echo "Q=32 启动 (PID: $!)"

    # Q=64 (GPU 4-5)
    CUDA_VISIBLE_DEVICES=4,5 nohup accelerate launch --multi_gpu --num_processes 2 \
        -m deep_compressor.train \
        --config configs/stage1_q64.yaml \
        --data_path $DATA_PATH --stage 1 \
        --wandb --wandb_project deep-compressor \
        > logs/q64_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    echo "Q=64 启动 (PID: $!)"

    # Q=128 (GPU 6)
    CUDA_VISIBLE_DEVICES=6 nohup python -m deep_compressor.train \
        --config configs/stage1_q128.yaml \
        --data_path $DATA_PATH --stage 1 \
        --wandb --wandb_project deep-compressor \
        > logs/q128_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    echo "Q=128 启动 (PID: $!)"

    # Q=256 (GPU 7)
    CUDA_VISIBLE_DEVICES=7 nohup python -m deep_compressor.train \
        --config configs/stage1_q256.yaml \
        --data_path $DATA_PATH --stage 1 \
        --wandb --wandb_project deep-compressor \
        > logs/q256_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    echo "Q=256 启动 (PID: $!)"

    echo ""
    echo -e "${GREEN}所有训练任务已在后台启动${NC}"
    echo ""
    echo "查看训练进度："
    echo "  tail -f logs/q16_*.log"
    echo "  tail -f logs/q32_*.log"
    echo "  ..."
    echo ""
    echo "查看所有训练进程："
    echo "  ps aux | grep deep_compressor.train"
    echo ""
    echo "停止所有训练："
    echo "  pkill -f 'deep_compressor.train'"
fi

echo ""
echo "监控GPU使用情况："
echo "  watch -n 1 nvidia-smi"
echo ""
echo "WandB监控："
echo "  https://wandb.ai"
echo ""
