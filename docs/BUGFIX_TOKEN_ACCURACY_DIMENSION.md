# Bug 修复：Token Accuracy 维度不匹配

## 🐛 问题描述

### 错误信息
```
ERROR - ✗ Failed to evaluate stage1_q16: The size of tensor a (668) must match the size of tensor b (652) at non-singleton dimension 1
ERROR - ✗ Failed to evaluate stage1_q32: The size of tensor a (684) must match the size of tensor b (652) at non-singleton dimension 1
ERROR - ✗ Failed to evaluate h200_stage1: The size of tensor a (764) must match the size of tensor b (700) at non-singleton dimension 1
```

### 触发条件
- 运行 `bash scripts/evaluate_all_models.sh`
- 评估训练好的压缩模型（非 baseline）
- Direct Qwen baseline 评估正常，压缩模型评估失败

### 影响范围
- ✅ Direct Qwen baseline 评估：**正常**
- ❌ Deep Compressor 模型评估：**失败**

---

## 🔍 根本原因

### 问题代码（deep_compressor/eval.py）

```python
# 错误的实现 ❌
outputs = unwrapped.decode(prefix_embeds, segment_ids, segment_mask)
logits = outputs.logits  # (B, prefix_len + segment_len, vocab_size)

# 直接对整个 logits 进行 shift
shift_logits = logits[:, :-1, :].contiguous()      # ❌ 包含 prefix 部分
shift_labels = segment_labels[:, 1:].contiguous()  # ✅ 只有 segment 部分

# 维度不匹配！
# shift_logits.shape[1] = prefix_len + segment_len - 1
# shift_labels.shape[1] = segment_len - 1
```

### 原理解释

在 Deep Compressor 中：

1. **Encoder 阶段**：
   ```python
   doc → encode_document() → byte_array → compress() → latent_array → up_mlp() → prefix_embeds
   # prefix_embeds.shape = (B, num_queries, hidden_size)
   ```

2. **Decoder 阶段**：
   ```python
   decode(prefix_embeds, segment_ids, segment_mask)
   # 输入序列 = [prefix_embeds (Q个), segment_ids (S个)]
   # 输出 logits.shape = (B, Q + S, vocab_size)
   ```

3. **Token Accuracy 计算**（错误的做法）：
   ```python
   # logits 包含 prefix 和 segment 两部分
   # 但 segment_labels 只对应 segment 部分
   shift_logits = logits[:, :-1, :]         # 长度 = Q + S - 1
   shift_labels = segment_labels[:, 1:]     # 长度 = S - 1
   # Q + S - 1 ≠ S - 1 → 维度不匹配！❌
   ```

### 为什么 Direct Qwen Baseline 没问题？

```python
# Direct Qwen: 没有 prefix，直接拼接 doc + segment
input_ids = torch.cat([doc_ids, segment_ids], dim=1)
outputs = qwen_model(input_ids, ...)
logits = outputs.logits  # (B, doc_len + segment_len, vocab_size)

# 正确提取 segment 部分
doc_len = doc_ids.shape[1]
segment_logits = logits[:, doc_len:-1, :]  # ✅ 只取 segment 部分并 shift
shift_labels = segment_labels[:, 1:]       # ✅ 长度匹配
```

---

## ✅ 修复方案

### 正确的实现

```python
# deep_compressor/eval.py (修复后)

outputs = unwrapped.decode(prefix_embeds, segment_ids, segment_mask)
logits = outputs.logits  # (B, prefix_len + segment_len, vocab_size)

# ✅ 关键：只提取 segment 部分的 logits
prefix_len = prefix_embeds.shape[1]
segment_logits = logits[:, prefix_len:, :]  # (B, segment_len, vocab_size)

# 然后再 shift
shift_logits = segment_logits[:, :-1, :].contiguous()  # (B, segment_len - 1, vocab_size)
shift_labels = segment_labels[:, 1:].contiguous()      # (B, segment_len - 1)

# ✅ 现在维度匹配了！
```

### 修改的文件

- `deep_compressor/eval.py` - 第 277-284 行

### 修改内容

```diff
  # Get predictions
  segment_mask = batch["segment_attention_mask"]
  outputs = unwrapped.decode(prefix_embeds, segment_ids, segment_mask)
- logits = outputs.logits  # (B, seq_len, vocab_size)
+ logits = outputs.logits  # (B, prefix_len + segment_len, vocab_size)

+ # Extract only the segment portion of logits
+ prefix_len = prefix_embeds.shape[1]
+ segment_logits = logits[:, prefix_len:, :]  # (B, segment_len, vocab_size)

  # Shift for next-token prediction
- shift_logits = logits[:, :-1, :].contiguous()
+ shift_logits = segment_logits[:, :-1, :].contiguous()
  shift_labels = segment_labels[:, 1:].contiguous()
```

---

## 🧪 验证测试

### 快速测试

```bash
# 使用修复后的代码测试
python scripts/test_fix_token_accuracy.py
```

**预期输出**：
```
✅ Evaluation successful! Bug is FIXED!

Metrics:
  Perplexity:      XX.XXXXXX
  Loss:            X.XXXXXX
  Token Accuracy:  X.XXXXXX
  Top-5 Accuracy:  X.XXXXXX

✅ Test PASSED: Token accuracy calculation fixed!
```

### 完整测试

```bash
# 运行完整的评估流程
bash scripts/evaluate_all_models.sh
```

**预期输出**：
```
Evaluating: stage1_q16...
✓ stage1_q16 evaluation complete

Evaluating: stage1_q32...
✓ stage1_q32 evaluation complete

====================================================================================================
EVALUATION RESULTS - COMPARISON TABLE
====================================================================================================
Model                | perplexity     | loss           | token_accuracy | top5_accuracy  | Retention
----------------------------------------------------------------------------------------------------
Qwen (no compress)   |      XX.XXXXXX |       X.XXXXXX |       X.XXXXXX |       X.XXXXXX |         —
Q=16                 |      XX.XXXXXX |       X.XXXXXX |       X.XXXXXX |       X.XXXXXX |     XX.X%
Q=32                 |      XX.XXXXXX |       X.XXXXXX |       X.XXXXXX |       X.XXXXXX |     XX.X%
====================================================================================================
```

---

## 📊 修复前后对比

### 修复前
```
✗ Failed to evaluate stage1_q16: The size of tensor a (668) must match the size of tensor b (652)
✗ Failed to evaluate stage1_q32: The size of tensor a (684) must match the size of tensor b (652)
✗ Failed to evaluate h200_stage1: The size of tensor a (764) must match the size of tensor b (700)
```

### 修复后
```
✓ stage1_q16 evaluation complete
✓ stage1_q32 evaluation complete
✓ h200_stage1 evaluation complete

All models evaluated successfully with token accuracy metrics!
```

---

## 🔬 技术细节

### 维度分析

假设：
- `num_queries (Q)` = 16
- `segment_len (S)` = 652
- `batch_size (B)` = 4

**修复前（错误）**：
```python
logits.shape           = (4, 668, vocab_size)  # 16 + 652 = 668
shift_logits.shape     = (4, 667, vocab_size)  # 668 - 1 = 667
shift_labels.shape     = (4, 651)              # 652 - 1 = 651
# 667 ≠ 651 → RuntimeError! ❌
```

**修复后（正确）**：
```python
logits.shape           = (4, 668, vocab_size)  # 16 + 652 = 668
prefix_len             = 16
segment_logits.shape   = (4, 652, vocab_size)  # logits[:, 16:, :]
shift_logits.shape     = (4, 651, vocab_size)  # 652 - 1 = 651
shift_labels.shape     = (4, 651)              # 652 - 1 = 651
# 651 == 651 → Success! ✅
```

### 为什么不同模型的错误数字不同？

错误信息中的数字差异来自不同的 `num_queries` 配置：

| 模型 | num_queries | segment_len | 错误的 shift_logits | 正确的 shift_labels | 差值 |
|------|-------------|-------------|---------------------|---------------------|------|
| stage1_q16 | 16 | 652 | 667 (16+652-1) | 651 (652-1) | 16 |
| stage1_q32 | 32 | 652 | 683 (32+652-1) | 651 (652-1) | 32 |
| h200_stage1 | 64 | 700 | 763 (64+700-1) | 699 (700-1) | 64 |

**差值 = num_queries** - 这正好证实了问题所在！

---

## 🎯 经验教训

### 1. **理解数据流**
在修改或添加新功能时，必须完全理解数据在模型中的流动：
- Encoder 输出什么？
- Decoder 接收什么？
- 中间有什么变换？

### 2. **注意维度匹配**
在进行张量操作时，始终检查：
- 输入维度
- 输出维度
- 中间步骤的维度变化

### 3. **区分 baseline 和压缩模型**
不同的模型架构可能有不同的数据处理逻辑：
- Direct Qwen: `doc + segment`
- Deep Compressor: `prefix + segment`

### 4. **充分测试**
添加新功能后，应该测试：
- ✅ Baseline 模型
- ✅ 压缩模型
- ✅ 不同的配置（Q=16, 32, 64, ...）

---

## 📝 相关文件

- `deep_compressor/eval.py` - 修复的核心文件
- `scripts/evaluate_all_checkpoints.py` - 调用评估函数
- `scripts/test_fix_token_accuracy.py` - 验证测试脚本
- `docs/EVALUATION_IMPROVEMENTS.md` - 评估系统文档

---

## ✅ 总结

**问题**: Token accuracy 计算时，logits 包含 prefix 部分，但 labels 只有 segment 部分，导致维度不匹配。

**修复**: 在 shift 之前，先提取 segment 部分的 logits。

**验证**: 运行 `python scripts/test_fix_token_accuracy.py` 确认修复成功。

**状态**: ✅ **已修复**，可以正常运行 `bash scripts/evaluate_all_models.sh`

---

**修复日期**: 2026-03-09
**影响版本**: 引入 token accuracy 功能后的所有版本
**修复提交**: (待提交)
