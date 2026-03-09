# Epoch-Based训练配置

## ✅ 问题修复

### 原问题
- 所有Q值都用相同的max_steps=50,000
- 导致不同Q值覆盖的epoch数差异巨大（3-13个epoch）
- 训练不公平，结果难以对比

### 修复后
- **统一为2.5个epoch**
- 根据effective batch size调整max_steps
- 所有模型看到相同数量的数据

---

## 📊 新的训练配置

### 数据集规模
- **训练样本**: 483,813
- **评估样本**: 54,529 (限制5,000用于训练中评估)

### 统一的训练量（2.5 epochs）

| Q值 | 有效Batch | Steps/Epoch | Max Steps | Warmup | Eval每 | Save每 |
|-----|-----------|-------------|-----------|--------|--------|--------|
| **64**  | 128 | 3,780  | **9,450**  | 473  | 945  | 1,890 |
| **128** | 96  | 5,040  | **12,600** | 630  | 1,260 | 2,520 |
| **256** | 64  | 7,560  | **18,900** | 945  | 1,890 | 3,780 |
| **512** | 32  | 15,120 | **37,800** | 1,890 | 3,780 | 7,560 |

**关键改进**:
- ✅ 所有模型训练2.5个epoch（数据覆盖一致）
- ✅ Warmup = 5% of max_steps
- ✅ Eval频率 = 每个epoch 4次
- ✅ Save频率 = 每个epoch 2次

---

## 🎯 为什么选2.5个epoch？

### Epoch数的权衡

| Epoch数 | 优点 | 缺点 | 适用 |
|---------|------|------|------|
| 1 epoch | 快速 | 可能欠拟合 | 快速实验 |
| 2-3 epochs | **平衡** | - | **推荐** ✅ |
| 5+ epochs | 充分训练 | 过拟合风险 | 小数据集 |

**选择2.5的理由**:
1. ✅ 数据集够大（48万）→ 不需要太多epoch
2. ✅ 避免过拟合
3. ✅ 训练时间合理（17.5小时）

---

## 📈 训练时间预估

### 按Q值分解

```
Q=64:  9,450 steps  →  ~2 hours
Q=128: 12,600 steps →  ~3 hours
Q=256: 18,900 steps →  ~4.5 hours
Q=512: 37,800 steps →  ~8 hours
----------------------------------------
Total:              →  ~17.5 hours
```

### 对比之前（50K steps统一）

| 配置 | Q=64 | Q=128 | Q=256 | Q=512 | 总计 |
|------|------|-------|-------|-------|------|
| **之前** | 13 epochs | 10 epochs | 7 epochs | 3 epochs | 不统一 ❌ |
| **现在** | 2.5 epochs | 2.5 epochs | 2.5 epochs | 2.5 epochs | **统一** ✅ |

---

## 🔢 计算公式

### Epoch计算

```python
num_samples = 483,813
effective_batch = num_gpus × batch_size × grad_accum

steps_per_epoch = ceil(num_samples / effective_batch)
max_steps = steps_per_epoch × target_epochs
```

### 示例：Q=128

```python
effective_batch = 8 × 6 × 2 = 96
steps_per_epoch = ceil(483813 / 96) = 5,040
max_steps = 5,040 × 2.5 = 12,600  ✓
```

---

## 📝 配置文件更新清单

所有配置文件已更新：

```yaml
# configs/qa_q64_8gpu.yaml
training:
  max_steps: 9450
  warmup_steps: 473
  eval_every: 945
  save_every: 1890

# configs/qa_q128_8gpu.yaml
training:
  max_steps: 12600
  warmup_steps: 630
  eval_every: 1260
  save_every: 2520

# configs/qa_q256_8gpu.yaml
training:
  max_steps: 18900
  warmup_steps: 945
  eval_every: 1890
  save_every: 3780

# configs/qa_q512_8gpu.yaml
training:
  max_steps: 37800
  warmup_steps: 1890
  eval_every: 3780
  save_every: 7560
```

---

## 🎯 训练策略

### 推荐顺序

```bash
# 1. 先训练Q=64（最快，2小时）
bash scripts/train_qa_q64_8gpu.sh

# 2. 训练Q=128（平衡，3小时）
bash scripts/train_qa_q128_8gpu.sh

# 3. 根据效果决定是否训练Q=256/512
```

### 批量训练

```bash
# 一次性训练全部（17.5小时）
bash scripts/train_all_q_values_8gpu.sh
```

---

## 📊 完整配置总览

### Q=64（最快）
```
Document: 4096 tokens
Queries: 64
Batch: 8 × 2 = 128 effective
Steps: 9,450 (2.5 epochs)
Memory: ~60GB (42%)
Time: ~2 hours
```

### Q=128（推荐）
```
Document: 4096 tokens
Queries: 128
Batch: 6 × 2 = 96 effective
Steps: 12,600 (2.5 epochs)
Memory: ~90GB (63%)
Time: ~3 hours
```

### Q=256（高性能）
```
Document: 4096 tokens
Queries: 256
Batch: 4 × 2 = 64 effective
Steps: 18,900 (2.5 epochs)
Memory: ~120GB (84%)
Time: ~4.5 hours
```

### Q=512（最佳性能）
```
Document: 4096 tokens
Queries: 512
Batch: 2 × 2 = 32 effective
Steps: 37,800 (2.5 epochs)
Memory: ~140GB (98%)
Gradient Ckpt: ON
Time: ~8 hours
```

---

## ✅ 优势

1. **公平对比**: 所有模型训练量相同
2. **数据充分**: 2.5个epoch足够学习
3. **避免过拟合**: 不会训练过多
4. **时间合理**: 总计17.5小时

---

**配置已优化，准备上传！** 🚀
