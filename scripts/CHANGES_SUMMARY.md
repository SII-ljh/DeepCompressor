# Stage 1 不同Q值实验 - 功能实现总结

本次更新为Deep Compressor项目添加了完整的Stage 1不同Q值（num_queries）对比实验流程。

## 📦 新增文件

### 配置文件（5个）

1. **`configs/stage1_q16.yaml`** - Q=16的训练配置
2. **`configs/stage1_q32.yaml`** - Q=32的训练配置
3. **`configs/stage1_q64.yaml`** - Q=64的训练配置
4. **`configs/stage1_q128.yaml`** - Q=128的训练配置
5. **`configs/stage1_q256.yaml`** - Q=256的训练配置

**特点**：
- 统一的max_doc_tokens=512
- 各自独立的output_dir
- wandb集成，带标签区分
- batch_size=8, gradient_accumulation_steps=2（总batch=16）

### 核心脚本（5个）

1. **`scripts/filter_ntp_data.py`** - 数据过滤工具
   - 按token长度过滤训练数据
   - 支持自定义阈值
   - 显示过滤统计信息

2. **`scripts/train_stage1_varying_q.py`** - 批量训练脚本
   - 自动训练多个Q值
   - 支持断点续训
   - 检测已存在checkpoint避免重复训练
   - dry_run模式用于测试

3. **`scripts/evaluate_all_checkpoints.py`** - 自动评估脚本
   - 自动发现outputs/下的所有checkpoint
   - 支持Stage 1 NTP和Stage 2 QA评估
   - 生成对比表格和CSV报告
   - 显示样本预测

4. **`scripts/analyze_data_distribution.py`** - 数据分析工具
   - 分析文档长度分布
   - 显示各种统计指标（均值、中位数、百分位数）
   - 按阈值统计样本数量
   - 可选生成可视化图表

5. **`scripts/run_full_experiment.sh`** - 一键执行脚本
   - 交互式引导完整流程
   - 数据准备 → 训练 → 评估
   - 颜色输出，用户友好
   - 自动生成结果摘要

### 文档（3个）

1. **`scripts/STAGE1_VARYING_Q_EXPERIMENTS.md`** - 完整实验指南
   - 详细的步骤说明
   - 配置文件解释
   - 常见问题解答
   - 时间估算
   - 疑难解答

2. **`scripts/QUICKSTART_STAGE1_Q_EXPERIMENTS.md`** - 快速开始指南
   - 最简单的使用方式
   - 常见用例
   - 快速参考命令

3. **`scripts/CHANGES_SUMMARY.md`** - 本文档
   - 功能总结
   - 文件清单
   - 修改说明

## 🔧 修改的文件

### 1. `deep_compressor/eval.py`

**修改内容**：
```python
@torch.no_grad()
def evaluate_ntp(model, eval_loader, accelerator,
                 tokenizer=None,
                 show_sample: bool = True):
```

**新增功能**：
- NTP评估时显示样本预测
- 展示压缩后的prefix生成的文本 vs 真实文本
- 只在主进程显示，避免重复
- 只显示第一个batch的第一个样本

### 2. `deep_compressor/train.py`

**修改内容**：
```python
metrics = evaluate_ntp(model, eval_loader,
                       accelerator, tokenizer=tokenizer,
                       show_sample=True)
```

**新增功能**：
- 将tokenizer传递给evaluate_ntp
- 启用样本显示功能

### 3. `CLAUDE.md`

**新增章节**：
- "Stage 1 Varying Query (Q) Experiments"
- 快速开始说明
- 文件清单
- 评估输出示例

## ✨ 核心功能

### 1. 数据准备

```bash
python scripts/filter_ntp_data.py \
    --input data/ntp_train.jsonl \
    --output data/ntp_train_512.jsonl \
    --max_length 512
```

**功能**：
- 过滤出指定长度以下的文档
- 实时显示进度条
- 输出过滤统计

### 2. 批量训练

```bash
python scripts/train_stage1_varying_q.py
```

**功能**：
- 依次训练Q=16,32,64,128,256
- 自动跳过已存在的checkpoint
- 记录训练时间
- 支持中断后继续

### 3. 自动评估

```bash
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train_512.jsonl \
    --stage 1
```

**功能**：
- 自动发现所有checkpoint
- 统一评估标准
- 生成对比表格
- 保存CSV报告

### 4. 数据分析

```bash
python scripts/analyze_data_distribution.py \
    --data_path data/ntp_train.jsonl
```

**功能**：
- 长度分布统计
- 百分位数分析
- 阈值样本统计
- 可视化图表（可选）

## 📊 输出结果

### 训练输出

```
outputs/
├── stage1_q16/checkpoint-final/trainable_weights.pt
├── stage1_q32/checkpoint-final/trainable_weights.pt
├── stage1_q64/checkpoint-final/trainable_weights.pt
├── stage1_q128/checkpoint-final/trainable_weights.pt
└── stage1_q256/checkpoint-final/trainable_weights.pt
```

### 评估输出

**终端输出**：
```
================================================================================
Comparison Table
================================================================================
Q      | perplexity | loss
--------------------------------
16     |    28.4523 |    3.3476
32     |    25.1234 |    3.2234
64     |    23.5678 |    3.1589
128    |    22.8901 |    3.1312
256    |    22.3456 |    3.1078
================================================================================
```

**CSV文件**：
```csv
checkpoint,q_value,perplexity,loss
stage1_q16,16,28.4523,3.3476
stage1_q32,32,25.1234,3.2234
stage1_q64,64,23.5678,3.1589
stage1_q128,128,22.8901,3.1312
stage1_q256,256,22.3456,3.1078
```

### 样本预测输出

**NTP评估**：
```
================================================================================
  Sample NTP Prediction
================================================================================
Prediction: The company announced quarterly earnings...
Gold:       The company announced quarterly earnings...
================================================================================
```

**QA评估**（已有功能，现在更突出）：
```
================================================================================
  Sample Predictions (first 5 examples)
================================================================================

[Sample 1]
Question:   What was the revenue in Q3?
Prediction: $1.2 billion
Gold:       $1.2 billion
EM: 1  F1: 1.0000
--------------------------------------------------------------------------------
...
================================================================================
```

## 🚀 使用流程

### 最简单方式（一键执行）

```bash
bash scripts/run_full_experiment.sh
```

### 分步执行

```bash
# 1. 准备数据
python scripts/filter_ntp_data.py \
    --input data/ntp_train.jsonl \
    --output data/ntp_train_512.jsonl \
    --max_length 512

# 2. （可选）分析数据分布
python scripts/analyze_data_distribution.py \
    --data_path data/ntp_train.jsonl

# 3. 批量训练
python scripts/train_stage1_varying_q.py

# 4. 评估所有模型
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train_512.jsonl \
    --stage 1 \
    --output results/stage1_results.csv
```

## 🎯 设计亮点

### 1. 模块化设计

每个脚本都可以独立使用：
- 数据过滤 → 单独工具
- 训练 → 可批量可单独
- 评估 → 可全部可指定

### 2. 容错能力

- 自动跳过已完成的训练
- 支持断点续训
- 异常处理不中断整体流程

### 3. 用户友好

- 交互式引导（run_full_experiment.sh）
- 详细的进度显示
- 清晰的错误提示
- 丰富的文档支持

### 4. 灵活配置

- 支持自定义Q值列表
- 支持自定义数据路径
- 支持自定义评估样本数
- 支持dry_run测试

### 5. 完整的观察性

- WandB集成
- 样本预测展示
- 对比表格
- CSV导出

## 📈 预期效果

通过这套工具，用户可以：

1. **快速启动实验** - 一键或几个命令完成
2. **系统对比不同Q值** - 自动化训练和评估
3. **直观看到效果** - 表格、曲线、样本预测
4. **灵活调整实验** - 模块化设计便于修改
5. **复现和扩展** - 清晰的文档和代码结构

## 🔍 技术细节

### 评估机制改进

- **NTP评估**：新增样本预测显示，直观看到压缩效果
- **QA评估**：强化样本展示，更清晰的性能指标

### 配置一致性

所有Q值配置保持参数一致，唯一差异：
- `perceiver.num_queries`
- `training.output_dir`
- `wandb.run_name`

### 数据处理优化

- 流式处理，不占用大量内存
- 进度条显示，实时反馈
- 异常处理，跳过错误行

### 并行友好

- 各Q值独立输出目录
- 可在不同机器并行训练
- 支持统一收集评估

## 📝 后续优化方向

1. **支持多GPU并行训练每个Q值**
2. **增加更多评估指标**（如压缩率、推理速度）
3. **自动选择最佳Q值**（基于多指标综合）
4. **结果可视化增强**（训练曲线对比图）
5. **实验配置管理**（实验追踪和版本控制）

## 🤝 使用建议

1. **首次运行**：使用`run_full_experiment.sh`熟悉流程
2. **数据分析**：运行`analyze_data_distribution.py`了解数据
3. **分步调试**：先单独训练一个Q值确认无误
4. **长期训练**：使用`nohup`或`screen`后台运行
5. **结果分析**：结合WandB曲线和评估表格综合判断

## 📚 参考文档

- 完整指南：`scripts/STAGE1_VARYING_Q_EXPERIMENTS.md`
- 快速开始：`scripts/QUICKSTART_STAGE1_Q_EXPERIMENTS.md`
- 项目说明：`CLAUDE.md`（已更新）

---

**版本**：v1.0
**日期**：2026-03-08
**作者**：Claude Code
