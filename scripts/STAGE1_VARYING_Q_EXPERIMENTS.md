# Stage 1 不同Q值实验指南

本指南用于进行第一阶段（NTP预训练）的Q值（num_queries）对比实验。

## 实验设置

- **Q值**: 16, 32, 64, 128, 256
- **文档长度**: < 512 tokens（统一过滤）
- **训练步数**: 50,000 steps per model
- **评估**: 每500步进行评估，并显示一个样本预测

## 快速开始

### 1. 准备数据

首先过滤出长度 < 512 tokens的训练数据：

```bash
python scripts/filter_ntp_data.py \
    --input data/ntp_train.jsonl \
    --output data/ntp_train_512.jsonl \
    --max_length 512
```

**预期结果**:
- 输入: `data/ntp_train.jsonl`
- 输出: `data/ntp_train_512.jsonl`
- 终端会显示保留的样本数量和百分比

### 2. 批量训练

#### 方式1：自动训练所有5个模型

```bash
python scripts/train_stage1_varying_q.py
```

这会依次训练Q=16, 32, 64, 128, 256的5个模型。

#### 方式2：训练指定的Q值

```bash
# 只训练Q=16和Q=32
python scripts/train_stage1_varying_q.py --q_values 16,32

# 训练Q=64, 128, 256
python scripts/train_stage1_varying_q.py --q_values 64,128,256
```

#### 方式3：断点续训

如果训练中断，可以使用`--resume`继续训练：

```bash
python scripts/train_stage1_varying_q.py --resume
```

#### 其他选项

```bash
# 不使用wandb
python scripts/train_stage1_varying_q.py --no_wandb

# 仅打印命令，不执行（测试用）
python scripts/train_stage1_varying_q.py --dry_run

# 使用自定义数据路径
python scripts/train_stage1_varying_q.py --data_path data/custom_ntp.jsonl
```

### 3. 评估所有模型

训练完成后，评估所有checkpoint：

```bash
# Stage 1 NTP评估
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train_512.jsonl \
    --stage 1

# 限制评估样本数量（加快速度）
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train_512.jsonl \
    --stage 1 \
    --max_eval_samples 1000

# 显示更多样本预测
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train_512.jsonl \
    --stage 1 \
    --show_samples 10

# 保存结果到CSV
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train_512.jsonl \
    --stage 1 \
    --output results/stage1_varying_q_results.csv
```

#### 评估指定checkpoint

```bash
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train_512.jsonl \
    --stage 1 \
    --checkpoints outputs/stage1_q16,outputs/stage1_q32
```

### 4. 单独训练某个Q值

如果只想训练某个特定的Q值，可以直接使用train.py：

```bash
# 训练Q=64
python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train_512.jsonl \
    --stage 1 \
    --wandb \
    --wandb_project deep-compressor
```

## 输出目录结构

训练完成后，目录结构如下：

```
outputs/
├── stage1_q16/
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   ├── ...
│   └── checkpoint-final/
│       └── trainable_weights.pt
├── stage1_q32/
│   └── checkpoint-final/
│       └── trainable_weights.pt
├── stage1_q64/
│   └── checkpoint-final/
│       └── trainable_weights.pt
├── stage1_q128/
│   └── checkpoint-final/
│       └── trainable_weights.pt
└── stage1_q256/
    └── checkpoint-final/
        └── trainable_weights.pt
```

## 配置文件

所有配置文件位于 `configs/` 目录：

- `stage1_q16.yaml` - Q=16的配置
- `stage1_q32.yaml` - Q=32的配置
- `stage1_q64.yaml` - Q=64的配置
- `stage1_q128.yaml` - Q=128的配置
- `stage1_q256.yaml` - Q=256的配置

每个配置的主要区别：
1. `perceiver.num_queries`: 设置不同的Q值
2. `training.output_dir`: 设置不同的输出目录
3. `wandb.run_name`: 设置不同的运行名称

其他参数（学习率、batch size等）保持一致。

## 监控训练进度

### 使用wandb（推荐）

访问 https://wandb.ai 查看实时训练曲线，可以看到：
- 训练loss
- 验证perplexity
- 学习率变化
- 每次评估的样本预测

### 查看终端输出

训练过程中会定期输出：
```
[NTP] step 500/50000  loss=3.1234  ppl=22.76  lr=1.00e-04

======================================================================
  Sample NTP Prediction
======================================================================
Prediction: The company announced...
Gold:       The company announced...
======================================================================

[NTP EVAL] step 500  perplexity=25.32  loss=3.2314
```

## 评估结果解读

评估脚本会输出如下对比表：

```
================================================================================
Comparison Table
================================================================================
Q      | perplexity | loss
---------------------------------
16     |    28.4523 |    3.3476
32     |    25.1234 |    3.2234
64     |    23.5678 |    3.1589
128    |    22.8901 |    3.1312
256    |    22.3456 |    3.1078
================================================================================
```

**期望观察**:
- Q值越大，perplexity应该越低（压缩信息更丰富）
- 但Q值太大可能过拟合或训练不稳定
- 找到最佳的Q值平衡点

## 常见问题

### 1. 数据过滤后样本太少？

检查原始数据中长度分布：

```python
from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
lengths = []

with open("data/ntp_train.jsonl") as f:
    for line in f:
        item = json.loads(line)
        tokens = tokenizer.encode(item["text"], add_special_tokens=False)
        lengths.append(len(tokens))

import numpy as np
print(f"<512 tokens: {sum(l < 512 for l in lengths)} / {len(lengths)}")
print(f"Mean length: {np.mean(lengths):.0f}")
print(f"Median length: {np.median(lengths):.0f}")
```

### 2. 内存不足（OOM）？

- 减小batch_size（在配置文件中修改）
- 增加gradient_accumulation_steps
- 使用梯度检查点: 在配置中设置 `gradient_checkpointing: true`
- 使用混合精度: 设置 `mixed_precision: "fp16"`（或"bf16"）

### 3. 训练速度慢？

- 使用多GPU: `accelerate launch --multi_gpu --num_processes 4 -m deep_compressor.train ...`
- 启用混合精度训练
- 减少eval频率: 修改配置中的`eval_every`

### 4. 如何恢复中断的训练？

```bash
# 自动检测并恢复
python scripts/train_stage1_varying_q.py --resume

# 或手动恢复特定模型
python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train_512.jsonl \
    --stage 1 \
    --resume_from outputs/stage1_q64/checkpoint-final
```

## 时间估算

根据硬件配置，每个模型的训练时间：
- **单张A100 (40GB)**: 约8-12小时
- **单张V100 (32GB)**: 约12-16小时
- **单张RTX 3090**: 约16-24小时
- **多张GPU**: 时间成比例减少

总时间（5个模型）：
- **单GPU**: 约2-5天
- **4×GPU并行**: 约12-30小时

## 下一步

训练和评估完成后，可以：

1. **分析结果**: 比较不同Q值的性能
2. **选择最佳Q**: 根据perplexity和实际任务需求
3. **进入Stage 2**: 使用最佳checkpoint进行QA微调

```bash
# Stage 2训练示例
python -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/qa_train.json \
    --eval_data_path data/qa_dev.json \
    --stage 2 \
    --resume_from outputs/stage1_q64/checkpoint-final \
    --wandb \
    --wandb_project deep-compressor
```

## 参考

- 配置文件: `configs/stage1_q*.yaml`
- 训练脚本: `scripts/train_stage1_varying_q.py`
- 评估脚本: `scripts/evaluate_all_checkpoints.py`
- 数据过滤: `scripts/filter_ntp_data.py`
