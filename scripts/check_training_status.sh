#!/bin/bash
# 检查训练状态的脚本

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                        训练状态检查                                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 1. 检查tmux会话
echo -e "${YELLOW}[1] Tmux会话状态${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if tmux has-session -t stage1_training 2>/dev/null; then
    echo -e "${GREEN}✓ Tmux会话 'stage1_training' 正在运行${NC}"
    tmux list-windows -t stage1_training
else
    echo -e "${RED}✗ Tmux会话 'stage1_training' 未找到${NC}"
fi
echo ""

# 2. 检查训练进程
echo -e "${YELLOW}[2] 训练进程状态${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
PROCESS_COUNT=$(ps aux | grep "deep_compressor.train" | grep -v grep | wc -l)
if [ "$PROCESS_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ 找到 $PROCESS_COUNT 个训练进程${NC}"
    echo ""
    ps aux | grep "deep_compressor.train" | grep -v grep | awk '{print "  PID: "$2"  CPU: "$3"%  MEM: "$4"%  CMD: "$11" "$12" "$13}'
else
    echo -e "${RED}✗ 没有找到训练进程${NC}"
    echo "可能原因："
    echo "  1. 训练还未启动"
    echo "  2. 训练启动失败"
    echo "  3. 训练已经完成或被终止"
fi
echo ""

# 3. 检查GPU使用情况
echo -e "${YELLOW}[3] GPU使用情况${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv | head -9
    echo ""

    # 检查是否有GPU被使用
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
    if [ "${GPU_UTIL%.*}" -gt 10 ]; then
        echo -e "${GREEN}✓ GPU正在使用中 (总利用率: ${GPU_UTIL}%)${NC}"
    else
        echo -e "${RED}✗ GPU几乎空闲 (总利用率: ${GPU_UTIL}%)${NC}"
        echo "这可能表示训练未正确启动"
    fi
else
    echo -e "${RED}✗ nvidia-smi 未找到${NC}"
fi
echo ""

# 4. 检查日志文件
echo -e "${YELLOW}[4] 最新日志文件${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "logs" ]; then
    LOG_COUNT=$(ls logs/*.log 2>/dev/null | wc -l)
    if [ "$LOG_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ 找到 $LOG_COUNT 个日志文件${NC}"
        echo ""
        echo "最新的5个日志文件："
        ls -lt logs/*.log 2>/dev/null | head -5 | awk '{print "  "$9" ("$5" bytes, "$6" "$7" "$8")"}'
        echo ""
        echo "查看最新日志的最后20行："
        LATEST_LOG=$(ls -t logs/*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo "文件: $LATEST_LOG"
            echo "────────────────────────────────────────────────────────────────────────────────"
            tail -20 "$LATEST_LOG"
        fi
    else
        echo -e "${RED}✗ 没有找到日志文件${NC}"
    fi
else
    echo -e "${RED}✗ logs 目录不存在${NC}"
fi
echo ""

# 5. 检查checkpoint
echo -e "${YELLOW}[5] Checkpoint状态${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "outputs" ]; then
    CHECKPOINT_COUNT=$(find outputs -name "trainable_weights.pt" 2>/dev/null | wc -l)
    if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ 找到 $CHECKPOINT_COUNT 个checkpoint${NC}"
        echo ""
        find outputs -name "trainable_weights.pt" -exec ls -lh {} \; | awk '{print "  "$9" ("$5")"}'
    else
        echo -e "${YELLOW}⚠ 还没有checkpoint（训练刚开始是正常的）${NC}"
    fi
else
    echo -e "${YELLOW}⚠ outputs 目录不存在${NC}"
fi
echo ""

# 6. 快速操作提示
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${YELLOW}快速操作：${NC}"
echo ""
echo "查看实时日志："
echo "  tail -f logs/\$(ls -t logs/*.log | head -1)"
echo ""
echo "进入tmux查看："
echo "  tmux attach -t stage1_training"
echo ""
echo "实时监控GPU："
echo "  watch -n 1 nvidia-smi"
echo ""
echo "停止所有训练："
echo "  tmux kill-session -t stage1_training"
echo "  # 或"
echo "  pkill -f 'deep_compressor.train'"
echo ""
echo "重新检查状态："
echo "  bash scripts/check_training_status.sh"
echo ""
