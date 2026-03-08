# Scripts 目录

本目录包含Deep Compressor的各种实验脚本和工具。

## 🚀 快速开始

### Stage 1 不同Q值实验（推荐）

```bash
# 一键运行完整流程
bash scripts/run_full_experiment.sh
```

📖 详细文档：
- **训练启动指南**：[TRAINING_GUIDE.md](TRAINING_GUIDE.md) ⭐ **新增**
- **快速参考**：[QUICKSTART_STAGE1_Q_EXPERIMENTS.md](QUICKSTART_STAGE1_Q_EXPERIMENTS.md)
- **完整指南**：[STAGE1_VARYING_Q_EXPERIMENTS.md](STAGE1_VARYING_Q_EXPERIMENTS.md)
- **功能总结**：[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)

## 📂 脚本分类

### 数据处理

- **`filter_ntp_data.py`** - 按文档长度过滤NTP训练数据
  ```bash
  python scripts/filter_ntp_data.py \
      --input data/ntp_train.jsonl \
      --output data/ntp_train_512.jsonl \
      --max_length 512
  ```

- **`analyze_data_distribution.py`** - 分析文档长度分布
  ```bash
  python scripts/analyze_data_distribution.py \
      --data_path data/ntp_train.jsonl \
      --plot results/distribution.png
  ```

- **`prepare_data.py`** - 准备训练数据（下载和预处理）
  ```bash
  python scripts/prepare_data.py              # 完整数据
  python scripts/prepare_data.py --test       # 测试子集
  python scripts/prepare_data.py --make-tiny  # 极小子集
  ```

### 训练

- **`train_stage1_varying_q.py`** - 批量训练不同Q值的Stage 1模型
  ```bash
  python scripts/train_stage1_varying_q.py
  python scripts/train_stage1_varying_q.py --q_values 16,32,64
  python scripts/train_stage1_varying_q.py --resume
  ```

- **`train_parallel_8gpu.sh`** - **8卡并行训练脚本（最快）** ⭐
  ```bash
  # 同时训练5个Q值，GPU自动分配
  bash scripts/train_parallel_8gpu.sh

  # 使用tmux管理，支持实时查看
  tmux attach -t stage1_training
  ```

### 评估

- **`evaluate_all_checkpoints.py`** - 自动评估所有checkpoint
  ```bash
  # Stage 1 NTP评估
  python scripts/evaluate_all_checkpoints.py \
      --eval_data data/ntp_train_512.jsonl \
      --stage 1 \
      --output results/stage1_results.csv

  # Stage 2 QA评估
  python scripts/evaluate_all_checkpoints.py \
      --eval_data data/qa_dev.json \
      --stage 2 \
      --show_samples 10
  ```

- **`benchmark.py`** - 基准测试（对比不同baseline）
  ```bash
  python scripts/benchmark.py \
      --config configs/benchmark.yaml \
      --checkpoint outputs/checkpoint-final/trainable_weights.pt \
      --eval_data data/qa_dev.json
  ```

### 诊断实验

- **`diagnostics/pre_training.py`** - 预训练阶段诊断（Exp 1-3）
  ```bash
  python scripts/diagnostics/pre_training.py \
      --config configs/macbook_debug.yaml \
      --data_path data/ntp_tiny.jsonl \
      --experiments 1,2,3
  ```

- **`diagnostics/mid_training.py`** - 训练中诊断（Exp 4-5）
  ```bash
  python scripts/diagnostics/mid_training.py \
      --config configs/macbook_debug.yaml \
      --data_path data/ntp_tiny.jsonl \
      --experiments 4,5
  ```

- **`diagnostics/post_training.py`** - 训练后诊断（Exp 6-9）
  ```bash
  python scripts/diagnostics/post_training.py \
      --config configs/benchmark.yaml \
      --checkpoint outputs/checkpoint-final/trainable_weights.pt \
      --eval_data data/qa_dev.json \
      --experiments 6,7,8,9
  ```

### 消融实验

- **`ablation.py`** - 17种消融实验
  ```bash
  python scripts/ablation.py --list  # 列出所有实验

  python scripts/ablation.py --stage 1 \
      --config configs/ablation_base.yaml \
      --data_path data/ablation/ntp_ablation.jsonl \
      --ablation full_pipeline,no_stage_c
  ```

### 超参数搜索

- **`hp_search.py`** - Optuna超参数搜索
  ```bash
  python scripts/hp_search.py --n_trials 50 --stage 1 \
      --config configs/hp_search.yaml \
      --data_path data/ntp_tiny.jsonl
  ```

### 渐进式过拟合测试

- **`overfitting/step1_single_sample.py`** - 单样本过拟合
- **`overfitting/step2_memorize_tiny.py`** - 记忆小数据集
- **`overfitting/step3_ablation_full.py`** - 完整消融实验

### 一键执行

- **`run_full_experiment.sh`** - Stage 1不同Q值完整流程
  ```bash
  bash scripts/run_full_experiment.sh
  ```

## 📊 工作流示例

### 工作流1：首次训练

```bash
# 1. 准备数据
python scripts/prepare_data.py

# 2. 过滤数据
python scripts/filter_ntp_data.py \
    --input data/ntp_train.jsonl \
    --output data/ntp_train_512.jsonl \
    --max_length 512

# 3. 批量训练
python scripts/train_stage1_varying_q.py

# 4. 评估
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train_512.jsonl \
    --stage 1
```

### 工作流2：调试和验证

```bash
# 1. 分析数据
python scripts/analyze_data_distribution.py \
    --data_path data/ntp_train.jsonl

# 2. 准备测试数据
python scripts/prepare_data.py --make-tiny

# 3. 快速训练测试
python -m deep_compressor.train \
    --config configs/tiny_subset.yaml \
    --data_path data/ntp_tiny.jsonl \
    --stage 1

# 4. 运行诊断
python scripts/diagnostics/pre_training.py \
    --config configs/macbook_debug.yaml \
    --data_path data/ntp_tiny.jsonl \
    --experiments 1,2,3
```

### 工作流3：完整实验

```bash
# Stage 1: 不同Q值实验
bash scripts/run_full_experiment.sh

# Stage 2: QA微调（使用最佳Stage 1 checkpoint）
python -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/qa_train.json \
    --eval_data_path data/qa_dev.json \
    --stage 2 \
    --resume_from outputs/stage1_q64/checkpoint-final

# 最终评估
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/qa_dev.json \
    --stage 2
```

## 🔍 脚本速查

| 任务 | 脚本 | 时间 |
|------|------|------|
| 准备全量数据 | `prepare_data.py` | 1-2小时 |
| 准备测试数据 | `prepare_data.py --test` | 5分钟 |
| 过滤数据 | `filter_ntp_data.py` | 5-10分钟 |
| 分析数据分布 | `analyze_data_distribution.py` | 5分钟 |
| 训练单个Stage 1模型（单卡） | `train.py` | 8-16小时 |
| 训练单个Stage 1模型（8卡） | `accelerate launch --num_processes 8` | 1-2小时 |
| 训练5个Q值模型（单卡依次） | `train_stage1_varying_q.py` | 2-5天 |
| **训练5个Q值模型（8卡并行）** ⭐ | `train_parallel_8gpu.sh` | **8-12小时** |
| 评估所有checkpoint | `evaluate_all_checkpoints.py` | 30分钟 |
| 运行诊断实验 | `diagnostics/*.py` | 10-30分钟 |
| 超参数搜索 | `hp_search.py` | 数小时-数天 |

## 📚 文档索引

### 新手入门
1. **[训练启动指南](TRAINING_GUIDE.md)** - **如何启动训练，8卡并行** ⭐ **推荐**
2. [快速开始](QUICKSTART_STAGE1_Q_EXPERIMENTS.md) - 最简单的使用方式
3. [完整指南](STAGE1_VARYING_Q_EXPERIMENTS.md) - 详细的步骤和说明

### 深入了解
4. [功能总结](CHANGES_SUMMARY.md) - 新增功能详解
5. [项目文档](../CLAUDE.md) - 整体架构和使用

### 问题排查
- 完整指南中的"常见问题"章节
- 快速开始中的"疑难解答"部分

## 💡 提示

- **首次使用**：推荐从`run_full_experiment.sh`开始
- **快速测试**：使用`--make-tiny`创建小数据集
- **长时间训练**：使用`nohup`或`screen`后台运行
- **多机并行**：各Q值可在不同机器上训练
- **结果分析**：结合WandB曲线和CSV报告

## 🤝 贡献

如果你添加了新的脚本，请：
1. 在此README中添加说明
2. 在脚本开头添加docstring
3. 支持`--help`参数
4. 添加使用示例

## 📞 帮助

- 查看脚本帮助：`python scripts/<script>.py --help`
- 查看项目文档：`cat CLAUDE.md`
- 提交issue：https://github.com/your-repo/issues

---

**最后更新**：2026-03-08
