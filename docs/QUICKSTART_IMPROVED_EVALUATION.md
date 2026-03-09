# 快速上手：改进的评估系统

## 📊 新功能概览

**统一输出 + 增强指标 + 样本预测**

从之前的混乱逐个输出，升级为清晰的统一对比表格，并新增了多个有用指标。

---

## 🚀 快速使用

### 基本用法（与之前完全相同）

```bash
# 使用 shell 脚本（推荐）
bash scripts/evaluate_all_models.sh

# 或直接运行 Python 脚本
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train.jsonl \
    --stage 1
```

---

## 📈 输出示例

### 1. 对比表格（所有模型一起显示）

```
====================================================================================================
EVALUATION RESULTS - COMPARISON TABLE
====================================================================================================
Model                | perplexity     | loss           | token_accuracy | top5_accuracy  | Retention
----------------------------------------------------------------------------------------------------
Qwen (no compress)   |      18.234567 |       2.903456 |       0.456789 |       0.678901 |         —
Q=16                 |      45.678901 |       3.821456 |       0.234567 |       0.456789 |     76.0%
Q=32                 |      32.123456 |       3.469789 |       0.345678 |       0.567890 |     83.7%
====================================================================================================
```

### 2. 样本预测（每个模型都有）

```
====================================================================================================
SAMPLE PREDICTIONS
====================================================================================================

────────────────────────────────────────────────────────────────────────────────────────────────────
  Direct Qwen (No Compression - Baseline)
────────────────────────────────────────────────────────────────────────────────────────────────────

  [Sample 1/3]
  Doc Preview: The black-tailed jackrabbit plays host to many ectoparasites...
  Prediction:  fleas, ticks, lice, and mites, and many endoparasites...
  Gold:        fleas, ticks, lice, and mites, and many endoparasites...

────────────────────────────────────────────────────────────────────────────────────────────────────
  Model: Q=16 (stage1_q16)
────────────────────────────────────────────────────────────────────────────────────────────────────

  [Sample 1/3]
  Doc Preview: The black-tailed jackrabbit plays host to many ectoparasites...
  Prediction:  the disease is a common cause of death in the United States...
  Gold:        fleas, ticks, lice, and mites, and many endoparasites...
```

---

## 📊 新增指标说明

| 指标 | 说明 | 理想值 |
|------|------|--------|
| **perplexity** | 困惑度，衡量预测不确定性 | 越低越好 |
| **loss** | 交叉熵损失 | 越低越好 |
| **token_accuracy** | Top-1 token 预测准确率 | 越高越好 |
| **top5_accuracy** | 正确 token 在 top-5 的比例 | 越高越好 |
| **Retention** | 相对 baseline 的质量保持率 | 100% = baseline 性能 |

**示例解读**：
- `token_accuracy = 0.456789` → 45.7% 的 token 预测完全正确
- `top5_accuracy = 0.678901` → 67.9% 的情况下正确 token 在 top-5 中
- `Retention = 83.7%` → 模型达到了 baseline 83.7% 的性能

---

## ⚙️ 高级选项

### 调整样本数量

```bash
# 收集更多样本（默认 3 个）
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train.jsonl \
    --stage 1 \
    --show_samples 5
```

### 保存结果到 CSV

```bash
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train.jsonl \
    --stage 1 \
    --output results/stage1_results_$(date +%Y%m%d).csv
```

CSV 格式：
```csv
checkpoint,q_value,perplexity,loss,token_accuracy,top5_accuracy
direct_qwen (baseline),baseline,18.234567,2.903456,0.456789,0.678901
stage1_q16,16,45.678901,3.821456,0.234567,0.456789
...
```

### 快速测试（少量数据）

```bash
# 只评估 100 个样本快速验证
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_tiny.jsonl \
    --stage 1 \
    --max_eval_samples 100 \
    --show_samples 2
```

---

## 🔍 与之前的区别

| 方面 | 之前 | 现在 |
|------|------|------|
| 输出方式 | 逐个输出，混乱 | 统一表格，清晰 |
| 指标数量 | 2 个 | 4 个 + 样本 |
| 样本展示 | 随机1个 | 每个模型N个 |
| Baseline | 无 | 自动评估 |
| 文档预览 | 无 | 显示前50 tokens |

---

## ✅ 测试验证

```bash
# 快速功能测试（~30 秒）
python scripts/test_improved_evaluation.py

# 输出示例：
# ✅ Test PASSED: All new features working correctly!
#   ✓ Token accuracy calculation
#   ✓ Top-5 accuracy calculation
#   ✓ Sample collection (with doc preview)
#   ✓ Unified metrics dictionary
```

---

## 📚 详细文档

完整的技术细节和设计说明见：
- `docs/EVALUATION_IMPROVEMENTS.md` - 详细的改进说明
- `scripts/TEST_BASELINE_COMPARISON.md` - Baseline 对比功能说明

---

## 💡 使用建议

1. **首次使用**：用 tiny 数据集快速测试，确保一切正常
2. **正式评估**：使用完整数据集，保存结果到 CSV
3. **模型选择**：根据 Retention 和 token_accuracy 权衡压缩比
4. **问题诊断**：查看样本预测，直观了解模型的生成质量

---

**核心改进**：更清晰、更全面、更易用！🎉
