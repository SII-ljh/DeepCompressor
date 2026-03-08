# 快速开始：Stage 1 不同Q值实验

一键运行完整实验流程。

## 最简单的方式

```bash
# 一键执行：数据准备 + 训练 + 评估
bash scripts/run_full_experiment.sh
```

这个脚本会引导你完成整个流程，并提供交互式选项。

## 分步执行

如果你想手动控制每一步：

### 1️⃣ 准备数据（5分钟）

```bash
# 过滤出长度 < 512 tokens的数据
python scripts/filter_ntp_data.py \
    --input data/ntp_train.jsonl \
    --output data/ntp_train_512.jsonl \
    --max_length 512
```

**可选**: 分析数据分布

```bash
# 查看数据长度分布统计
python scripts/analyze_data_distribution.py \
    --data_path data/ntp_train.jsonl

# 生成可视化图表
python scripts/analyze_data_distribution.py \
    --data_path data/ntp_train.jsonl \
    --plot results/length_distribution.png
```

### 2️⃣ 批量训练（2-5天，取决于GPU）

```bash
# 训练所有5个模型（Q = 16, 32, 64, 128, 256）
python scripts/train_stage1_varying_q.py
```

**训练特定Q值**:

```bash
# 只训练Q=64和Q=128
python scripts/train_stage1_varying_q.py --q_values 64,128
```

**断点续训**:

```bash
# 如果训练中断，使用--resume继续
python scripts/train_stage1_varying_q.py --resume
```

### 3️⃣ 评估所有模型（30分钟）

```bash
# 评估所有训练好的checkpoint
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train_512.jsonl \
    --stage 1 \
    --output results/stage1_results.csv
```

## 期望结果

训练完成后，你会得到：

1. **5个训练好的模型**
   - `outputs/stage1_q16/checkpoint-final/`
   - `outputs/stage1_q32/checkpoint-final/`
   - `outputs/stage1_q64/checkpoint-final/`
   - `outputs/stage1_q128/checkpoint-final/`
   - `outputs/stage1_q256/checkpoint-final/`

2. **评估报告**
   ```
   Q      | perplexity | loss
   --------------------------------
   16     |    28.45   |  3.35
   32     |    25.12   |  3.22
   64     |    23.57   |  3.16
   128    |    22.89   |  3.13
   256    |    22.35   |  3.11
   ```

3. **WandB曲线** - 访问 https://wandb.ai 查看训练过程

## 单独训练某个Q值

如果只想训练一个特定的Q值（比如Q=64）：

```bash
python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train_512.jsonl \
    --stage 1 \
    --wandb \
    --wandb_project deep-compressor
```

## 查看训练进度

### 实时监控

```bash
# 查看训练日志
tail -f outputs/stage1_q64/train.log

# 访问WandB查看曲线
open https://wandb.ai
```

### 检查已完成的训练

```bash
# 列出所有checkpoint
ls -lh outputs/stage1_*/checkpoint-final/trainable_weights.pt
```

## 常见用例

### 用例1：快速测试（1小时）

测试整个流程是否正常工作：

```bash
# 1. 准备小数据集
python scripts/filter_ntp_data.py \
    --input data/ntp_tiny.jsonl \
    --output data/ntp_test_512.jsonl \
    --max_length 512

# 2. 训练一个Q值（短步数）
python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_test_512.jsonl \
    --stage 1 \
    --max_train_samples 100

# 修改config中的max_steps为100即可快速测试
```

### 用例2：对比2-3个Q值

只训练几个关键的Q值进行对比：

```bash
# 训练Q=32, 64, 128
python scripts/train_stage1_varying_q.py --q_values 32,64,128

# 评估
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train_512.jsonl \
    --stage 1 \
    --checkpoints outputs/stage1_q32,outputs/stage1_q64,outputs/stage1_q128
```

### 用例3：在多台机器上并行训练

在机器A上：
```bash
python scripts/train_stage1_varying_q.py --q_values 16,32
```

在机器B上：
```bash
python scripts/train_stage1_varying_q.py --q_values 64,128
```

在机器C上：
```bash
python scripts/train_stage1_varying_q.py --q_values 256
```

然后在一台机器上收集所有checkpoint进行统一评估。

## 疑难解答

### 问题：OOM (Out of Memory)

**解决方案**：修改配置文件中的batch_size

```yaml
# configs/stage1_q64.yaml
training:
  batch_size: 4  # 改为 2
  gradient_accumulation_steps: 4  # 改为 8（保持总batch size=16）
```

### 问题：训练太慢

**解决方案1**: 启用混合精度

```yaml
training:
  mixed_precision: "fp16"  # 或 "bf16"
```

**解决方案2**: 使用多GPU

```bash
accelerate launch --multi_gpu --num_processes 4 \
    scripts/train_stage1_varying_q.py
```

### 问题：数据过滤后样本太少

**解决方案**：查看数据分布，调整阈值

```bash
python scripts/analyze_data_distribution.py \
    --data_path data/ntp_train.jsonl

# 根据结果调整max_length
python scripts/filter_ntp_data.py \
    --input data/ntp_train.jsonl \
    --output data/ntp_train_1024.jsonl \
    --max_length 1024  # 增大阈值
```

## 进阶：自定义配置

复制并修改配置文件：

```bash
# 复制基础配置
cp configs/stage1_q64.yaml configs/my_custom_q64.yaml

# 修改参数
vim configs/my_custom_q64.yaml
```

常见修改：
- `learning_rate`: 学习率
- `max_steps`: 训练步数
- `batch_size`: 批大小
- `eval_every`: 评估频率
- `save_every`: 保存频率

## 下一步

实验完成后：

1. **选择最佳Q值** - 根据perplexity和实际任务需求
2. **进入Stage 2** - QA微调

```bash
python -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/qa_train.json \
    --eval_data_path data/qa_dev.json \
    --stage 2 \
    --resume_from outputs/stage1_q64/checkpoint-final
```

## 帮助

详细文档：
- 📖 完整指南: `scripts/STAGE1_VARYING_Q_EXPERIMENTS.md`
- 🔧 训练脚本: `scripts/train_stage1_varying_q.py --help`
- 📊 评估脚本: `scripts/evaluate_all_checkpoints.py --help`
- 📈 数据分析: `scripts/analyze_data_distribution.py --help`
