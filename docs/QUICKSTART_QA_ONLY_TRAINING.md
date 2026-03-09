# Quick Start: Pure QA Training (Skip Stage 1)

**新方法**: 直接在QA任务上训练，跳过Stage 1 NTP预训练。

---

## 🎯 为什么直接训练QA？

基于之前的分析，Stage 1 NTP训练存在信息丢失问题（用零向量训练导致模型丢失关键信息）。

**新方案**：直接在大规模QA数据集上训练，让模型学习如何压缩文档并回答问题，避免Stage 1的信息丢失问题。

---

## 📦 1. 准备大规模QA数据集

### 下载完整数据集（~800K训练样本）

```bash
# 下载所有QA数据集（英文+中文）
python scripts/prepare_large_qa_data.py
```

**数据集包括**：
- SQuAD v1.1 + v2.0 (217K)
- TriviaQA (95K)
- Natural Questions (150K, long context)
- HotpotQA (90K, multi-hop)
- CMRC2018 (10K, Chinese)
- DuReader (15K, Chinese)
- DRCD (27K, Traditional Chinese)
- WebQA (42K, Chinese web)

**总计**: ~800K train, ~60K dev

### 仅下载中文数据集

```bash
python scripts/prepare_large_qa_data.py --only-chinese
```

### 仅下载英文数据集

```bash
python scripts/prepare_large_qa_data.py --only-english
```

### 测试模式（快速验证）

```bash
python scripts/prepare_large_qa_data.py --test
```

输出：~10K train, ~2K dev

---

## 🧪 2. 过拟合实验（验证模型容量）

**目的**: 验证模型能够记住小数据集（loss应降至<0.01）

```bash
# 快速过拟合测试（10个样本，1000步）
python scripts/quick_overfit_qa.py

# 自定义参数
python scripts/quick_overfit_qa.py --samples 20 --steps 2000 --q_length 64
```

**成功标准**:
- ✅ Loss < 0.01: 模型成功记住数据
- ✅ Loss < 0.1: 模型基本收敛
- ⚠️ Loss < 0.5: 模型正在收敛，需要更多步数
- ❌ Loss >= 0.5: 检查模型架构或训练代码

---

## 🚀 3. 全量训练：对比不同Q长度

训练多个模型（Q=16, 32, 64, 128, 256），找到最优压缩比。

### 方法1: 批量训练（推荐）

```bash
# 训练所有Q值（使用大规模数据集）
python scripts/train_qa_varying_q.py --use_large_data

# 训练指定Q值
python scripts/train_qa_varying_q.py --q_values 64 128 256

# 预览命令（不实际运行）
python scripts/train_qa_varying_q.py --dry_run
```

### 方法2: 手动训练单个模型

```bash
# Q=128示例
python -m deep_compressor.train \
    --config configs/qa_q128.yaml \
    --data_path data/qa_large_train.json \
    --eval_data_path data/qa_large_dev.json \
    --stage 2 \
    --wandb --wandb_project deep-compressor-qa
```

---

## 📊 4. 评估所有模型

训练完成后，使用统一脚本评估所有checkpoint：

```bash
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/qa_large_dev.json \
    --stage 2 \
    --output results/qa_varying_q_results.csv
```

**输出**: 包含所有模型的对比表格（EM, F1, PPL, Token Accuracy）

---

## 📁 文件结构

### 数据文件
```
data/
├── qa_large_train.json   # ~800K样本，大规模QA训练集
├── qa_large_dev.json     # ~60K样本，评估集
├── qa_train.json         # 原始标准QA数据（~234K）
└── qa_dev.json           # 原始标准QA评估集（~31K）
```

### 配置文件（自动生成）
```
configs/
├── qa_q16.yaml
├── qa_q32.yaml
├── qa_q64.yaml
├── qa_q128.yaml
└── qa_q256.yaml
```

### 输出目录
```
outputs/
├── qa_q16/checkpoint-final/
├── qa_q32/checkpoint-final/
├── qa_q64/checkpoint-final/
├── qa_q128/checkpoint-final/
└── qa_q256/checkpoint-final/
```

---

## ⚙️ 训练参数

### 默认配置（configs/default.yaml）

```yaml
training:
  stage: 2                        # QA训练
  learning_rate: 1.0e-4
  batch_size: 4
  gradient_accumulation_steps: 4
  max_steps: 50000
  warmup_steps: 1000
  eval_every: 500
  save_every: 1000

perceiver:
  num_queries: 128                # 可通过ablation.override_num_queries覆盖

ablation:
  enable_kl_distillation: false   # 纯QA训练不需要蒸馏
  enable_hidden_mse_distillation: false
```

---

## 📈 预期结果

### 不同Q长度的性能对比

| Q值 | 压缩比 | EM (预期) | F1 (预期) | 训练时间(H100) |
|-----|--------|-----------|-----------|----------------|
| 16  | ~32:1  | 20-30%    | 0.30-0.40 | ~3小时         |
| 32  | ~16:1  | 30-40%    | 0.40-0.50 | ~3.5小时       |
| 64  | ~8:1   | 40-50%    | 0.50-0.60 | ~4小时         |
| 128 | ~4:1   | 50-60%    | 0.60-0.70 | ~5小时         |
| 256 | ~2:1   | 60-70%    | 0.70-0.80 | ~6小时         |

**注意**: 这些是基于~800K样本训练的预期值，实际结果可能因数据质量和训练时长而异。

---

## 🎓 训练策略

### 策略1: 快速迭代（推荐新手）

```bash
# 1. 过拟合实验（5分钟）
python scripts/quick_overfit_qa.py --samples 10 --steps 500

# 2. 小规模训练（使用标准数据集，~3小时）
python scripts/train_qa_varying_q.py --q_values 64 128

# 3. 评估
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/qa_dev.json \
    --stage 2
```

### 策略2: 完整训练（推荐最终模型）

```bash
# 1. 下载大规模数据集（1-2小时）
python scripts/prepare_large_qa_data.py

# 2. 过拟合验证（确保模型正常）
python scripts/quick_overfit_qa.py

# 3. 全量训练所有Q值（~24小时）
python scripts/train_qa_varying_q.py --use_large_data

# 4. 评估对比
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/qa_large_dev.json \
    --stage 2 \
    --output results/qa_final_results.csv
```

---

## 🔍 调试技巧

### Loss不下降

1. **检查过拟合实验**:
   ```bash
   python scripts/quick_overfit_qa.py --samples 5 --steps 1000
   ```
   如果过拟合失败，说明模型或训练代码有问题。

2. **降低Q值**: Q=128或256可能过大，试试Q=32或64。

3. **增加学习率**: 在config中设置`training.learning_rate: 5e-4`。

### 内存不足（OOM）

1. **减小batch_size**: 设置`training.batch_size: 2`。
2. **启用gradient_checkpointing**: 设置`training.gradient_checkpointing: true`。
3. **减小Q值**: 使用Q=64而不是Q=256。

### 数据下载失败

```bash
# 重试单独数据集
python scripts/prepare_large_qa_data.py --only-english
python scripts/prepare_large_qa_data.py --only-chinese
```

---

## 📊 评估指标

### EM (Exact Match)
完全匹配正确答案的比例（0-100%）

### F1 Score
Token级别的F1分数（0-1.0）

### Perplexity (PPL)
预测不确定性（越低越好）

### Token Accuracy
Top-1 token预测准确率（0-100%）

---

## 🎯 下一步

完成训练后：

1. **分析结果**: 查看wandb曲线，对比不同Q值
2. **选择最优Q**: 根据EM/F1权衡压缩比和性能
3. **生成示例**: 使用最优模型在测试集上生成答案
4. **发布模型**: 保存最终checkpoint供部署使用

---

## 📚 相关文档

- **完整指南**: [`docs/QUESTION_GUIDED_COMPRESSION.md`](./QUESTION_GUIDED_COMPRESSION.md)
- **项目文档**: [`CLAUDE.md`](../CLAUDE.md)
- **评估改进**: [`docs/EVALUATION_IMPROVEMENTS.md`](./EVALUATION_IMPROVEMENTS.md)

---

## 💡 FAQ

### Q: 为什么跳过Stage 1？
A: Stage 1用零向量训练导致信息丢失，直接QA训练可避免这个问题。

### Q: 需要多少数据？
A: 最少~10K样本可训练，建议>100K样本以获得良好性能。

### Q: 训练需要多久？
A: 在8×H100上，~800K样本训练~5小时（Q=128）。

### Q: 是否需要NTP数据？
A: 不需要。纯QA训练只需要QA数据集。

---

**开始训练！ 🚀**
