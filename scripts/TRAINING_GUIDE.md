# 训练启动完整指南

## 🎯 前置准备

### 1. 检查环境

```bash
# 检查GPU
nvidia-smi

# 检查依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
```

### 2. 准备数据

```bash
# 如果还没有训练数据，先准备
python scripts/prepare_data.py

# 注意：不需要预先过滤数据！
# 配置文件中的 max_doc_tokens: 512 会在加载时自动截断
```

### 3. 创建必要目录

```bash
mkdir -p logs outputs results
```

## 🚀 启动训练

### 方式1：单卡训练

**最简单**，适合单GPU环境或测试：

```bash
# 训练Q=64
python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train_512.jsonl \
    --stage 1 \
    --wandb \
    --wandb_project deep-compressor
```

### 方式2：单个Q值多卡训练（推荐用于单个实验）

**8卡DDP训练单个Q值**：

```bash
accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train_512.jsonl \
    --stage 1 \
    --wandb \
    --wandb_project deep-compressor
```

**4卡训练**：
```bash
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train_512.jsonl \
    --stage 1
```

### 方式3：8卡并行训练多个Q值（推荐用于批量实验）⭐

**一键启动**（使用tmux管理）：

```bash
bash scripts/train_parallel_8gpu.sh
```

这会自动：
- GPU 0-1: 训练Q=16 (2卡DDP)
- GPU 2-3: 训练Q=32 (2卡DDP)
- GPU 4-5: 训练Q=64 (2卡DDP)
- GPU 6: 训练Q=128 (单卡)
- GPU 7: 训练Q=256 (单卡)

**查看训练进度**：
```bash
# 进入tmux会话
tmux attach -t stage1_training

# 在tmux中切换窗口
Ctrl+B 然后按 0/1/2/3/4 切换到不同Q值的训练

# 退出但保持训练（detach）
Ctrl+B d

# 查看日志
tail -f logs/q64_*.log
```

**停止所有训练**：
```bash
tmux kill-session -t stage1_training
```

### 方式4：自定义GPU分配

**手动指定GPU训练不同Q值**：

```bash
# 终端1: GPU 0-3训练Q=16
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu --num_processes 4 \
    -m deep_compressor.train \
    --config configs/stage1_q16.yaml \
    --data_path data/ntp_train_512.jsonl \
    --stage 1

# 终端2: GPU 4-7训练Q=32
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
    --multi_gpu --num_processes 4 \
    -m deep_compressor.train \
    --config configs/stage1_q32.yaml \
    --data_path data/ntp_train_512.jsonl \
    --stage 1
```

## 📊 训练配置优化

### 内存优化

如果遇到OOM，修改配置文件：

```yaml
# configs/stage1_q64.yaml
training:
  batch_size: 4          # 减小 (原8)
  gradient_accumulation_steps: 4  # 增大 (原2)
  mixed_precision: "fp16"         # 启用混合精度
  gradient_checkpointing: true    # 启用梯度检查点
```

### 速度优化

```yaml
training:
  mixed_precision: "bf16"  # A100推荐bf16
  batch_size: 16           # 增大batch size
  gradient_accumulation_steps: 1
```

### 调试模式（快速验证）

```bash
# 使用tiny数据集，训练100步
python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_tiny.jsonl \
    --stage 1 \
    --max_train_samples 50

# 修改config中的max_steps为100
```

## 🔍 监控训练

### 1. 实时GPU监控

```bash
# 实时查看GPU使用
watch -n 1 nvidia-smi

# 或更详细的监控
nvidia-smi dmon -i 0,1,2,3,4,5,6,7 -s pucvmet
```

### 2. 查看训练日志

```bash
# 实时查看日志
tail -f logs/q64_*.log

# 查看最近的输出
tail -100 logs/q64_*.log

# 搜索特定内容
grep "EVAL" logs/q64_*.log
```

### 3. WandB监控（推荐）

访问 https://wandb.ai 查看：
- 训练loss曲线
- 验证perplexity
- 学习率变化
- GPU利用率
- 样本预测

### 4. 检查训练进程

```bash
# 查看所有训练进程
ps aux | grep deep_compressor.train

# 查看进程GPU使用
nvidia-smi

# 查看进程详细信息
ps -p <PID> -o pid,cmd,%cpu,%mem,etime
```

## 🛠️ 常见操作

### 断点续训

```bash
# 从最新checkpoint恢复
python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train_512.jsonl \
    --stage 1 \
    --resume_from outputs/stage1_q64/checkpoint-final
```

### 后台训练（nohup）

```bash
# 使用nohup后台运行
nohup python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train_512.jsonl \
    --stage 1 \
    > logs/q64_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 查看日志
tail -f logs/q64_*.log

# 查看进程
jobs -l
```

### 停止训练

```bash
# 优雅停止（Ctrl+C）
# 会保存当前checkpoint

# 强制停止特定进程
kill <PID>

# 停止所有训练
pkill -f 'deep_compressor.train'
```

## 📈 训练时间估算

### 单个Q值（50K steps）

| 硬件配置 | 预计时间 |
|---------|---------|
| 1× A100 (40GB) | 8-12小时 |
| 1× V100 (32GB) | 12-16小时 |
| 1× RTX 3090 | 16-24小时 |
| 4× A100 (DDP) | 2-4小时 |
| 8× A100 (DDP) | 1-2小时 |

### 批量训练5个Q值

| 策略 | 时间 |
|------|------|
| 单卡依次训练 | 2-5天 |
| 8卡并行（方式3）| 8-12小时 |
| 多机并行 | 可按需缩短 |

## 🎓 Accelerate配置

### 首次使用需要配置

```bash
accelerate config
```

推荐配置（8卡训练）：
```
- 计算环境: This machine
- 机器类型: Multi-GPU
- GPU数量: 8
- 是否使用DeepSpeed: No
- 混合精度: fp16 (或 bf16 for A100)
- 其他选项: 默认
```

### 查看当前配置

```bash
accelerate env
```

### 使用配置文件

创建 `accelerate_config.yaml`：
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 8
mixed_precision: fp16
```

使用：
```bash
accelerate launch \
    --config_file accelerate_config.yaml \
    -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train_512.jsonl \
    --stage 1
```

## 🐛 常见问题

### 1. CUDA Out of Memory

**解决方案**：
- 减小 `batch_size`
- 增大 `gradient_accumulation_steps`
- 启用 `gradient_checkpointing: true`
- 使用 `mixed_precision: "fp16"`

### 2. 训练速度慢

**检查**：
- GPU利用率（`nvidia-smi`）
- 数据加载（增加`num_workers`）
- 使用混合精度
- 减小eval频率

### 3. 多卡不生效

**检查**：
```bash
# 确认accelerate已安装
pip install accelerate

# 重新配置
accelerate config

# 检查环境
accelerate env
```

### 4. WandB无法登录

```bash
# 离线模式
--wandb_offline

# 或不使用wandb
# 去掉 --wandb 参数
```

### 5. 端口冲突（多机训练）

```bash
# 指定主节点端口
accelerate launch \
    --main_process_port 29500 \
    ...
```

## 💡 最佳实践

### 1. 训练前

- ✅ 先用tiny数据集验证流程
- ✅ 检查GPU可用性和内存
- ✅ 查看数据分布
- ✅ 配置WandB（便于监控）

### 2. 训练中

- ✅ 定期查看loss曲线
- ✅ 检查eval perplexity
- ✅ 查看样本预测质量
- ✅ 监控GPU利用率

### 3. 训练后

- ✅ 评估所有checkpoint
- ✅ 对比不同Q值性能
- ✅ 保存最佳模型
- ✅ 清理中间checkpoint

## 📚 相关命令速查

```bash
# 启动训练
python -m deep_compressor.train --config <config> --data_path <data> --stage 1

# 8卡DDP训练
accelerate launch --multi_gpu --num_processes 8 -m deep_compressor.train ...

# 8卡并行训练多个Q值
bash scripts/train_parallel_8gpu.sh

# 查看GPU
nvidia-smi

# 实时监控GPU
watch -n 1 nvidia-smi

# 查看日志
tail -f logs/q64_*.log

# 后台训练
nohup python -m deep_compressor.train ... > logs/train.log 2>&1 &

# 停止训练
pkill -f 'deep_compressor.train'

# 断点续训
--resume_from outputs/stage1_q64/checkpoint-final
```

## 🎯 快速开始（推荐流程）

```bash
# 1. 准备数据
python scripts/filter_ntp_data.py \
    --input data/ntp_train.jsonl \
    --output data/ntp_train_512.jsonl \
    --max_length 512

# 2. 快速测试（可选）
python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_tiny.jsonl \
    --stage 1 \
    --max_train_samples 10

# 3. 正式训练（选择一种）

# 方式A: 8卡并行训练所有Q值（最快）
bash scripts/train_parallel_8gpu.sh

# 方式B: 8卡DDP训练单个Q值
accelerate launch --multi_gpu --num_processes 8 \
    -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train_512.jsonl \
    --stage 1 \
    --wandb

# 方式C: 依次训练所有Q值
python scripts/train_stage1_varying_q.py

# 4. 评估
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train_512.jsonl \
    --stage 1
```

---

**获取帮助**：
- 查看脚本帮助: `python scripts/train_stage1_varying_q.py --help`
- 查看配置说明: `cat scripts/STAGE1_VARYING_Q_EXPERIMENTS.md`
- 项目文档: `cat CLAUDE.md`
