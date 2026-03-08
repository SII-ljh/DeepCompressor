# Stage 2 Three-Phase Training Strategy (No Truncation)

## 策略概述

为了避免截断损失，将 Stage 2 分为三个阶段，每个阶段只训练对应长度范围内的样本：

| 阶段 | 长度范围 | 占比 | 样本数 | 目的 |
|------|---------|------|--------|------|
| **Stage 2a** | 0-512 | 48.4% | ~127K | 与 Stage 1 相同长度，平滑过渡 |
| **Stage 2b** | 512-2048 | 13.6% | ~36K | 中长文档，渐进适应 |
| **Stage 2c** | 0-2048 | **62%** | ~163K | 混合训练，巩固全长度范围 |

**关键优势**：
- ✅ **零截断** — 每个样本都在其完整长度范围内训练
- ✅ **渐进适应** — 512 → 512-2048 → 全混合，避免分布突变
- ✅ **可比性** — Dev set (平均 168 tokens) 与训练长度匹配

## 数据统计

基于分析结果：
- **qa_train**: 263,047 samples
  - 0-512 tokens: 127,259 samples (48.4%)
  - 512-2048 tokens: 35,788 samples (13.6%)
  - 0-2048 tokens: 163,047 samples (62.0%)
  - > 2048 tokens: 100,000 samples (38.0%) — **不参与训练**

- **qa_dev**: 35,251 samples
  - 平均 168 tokens，99.5% < 512 tokens

---

## 训练流程

### Step 0: 准备三个过滤数据集

```bash
# 过滤 0-512 tokens（Stage 2a）
python scripts/filter_qa_by_length.py --max_tokens 512

# 过滤 512-2048 tokens（Stage 2b，新增）
python scripts/filter_qa_by_length.py --min_tokens 512 --max_tokens 2048

# 过滤 0-2048 tokens（Stage 2c，新增）
python scripts/filter_qa_by_length.py --max_tokens 2048
```

生成：
- `data/qa_train_filtered_512.json` (~127K, Stage 2a)
- `data/qa_train_filtered_512_2048.json` (~36K, Stage 2b)
- `data/qa_train_filtered_2048.json` (~163K, Stage 2c)
- `data/qa_dev_filtered_512.json` (Dev set, 99.5% samples)

---

### Step 1: Stage 2a - 短文档（0-512 tokens, 5K 步）

与 Stage 1 相同长度，平滑过渡。

```bash
bash scripts/train_h200_stage2a_short.sh
```

**配置**：
- `max_doc_tokens: 512`
- `batch_size: 8, gradient_accumulation: 2` (effective=128)
- `max_steps: 5000` (~5 epochs over 127K samples)
- `learning_rate: 5e-5`

**预计时长**: 2-3 小时

**输出**: `outputs/h200_stage2a/checkpoint-final/`

---

### Step 2: Stage 2b - 中长文档（512-2048 tokens, 3K 步，新增）

专注训练中长文档，渐进适应更长序列。

```bash
bash scripts/train_h200_stage2b_medium.sh
```

**配置**：
- `max_doc_tokens: 2048`
- `batch_size: 4, gradient_accumulation: 2` (effective=64)
- `max_steps: 3000` (~5 epochs over 36K samples)
- `learning_rate: 3e-5`
- **Resume from Stage 2a**

**预计时长**: 1.5-2 小时

**输出**: `outputs/h200_stage2b/checkpoint-final/`

---

### Step 3: Stage 2c - 全长度混合（0-2048 tokens, 8K 步，新增）

混合训练所有长度范围，巩固全能力。

```bash
bash scripts/train_h200_stage2c_mixed.sh
```

**配置**：
- `max_doc_tokens: 2048`
- `batch_size: 4, gradient_accumulation: 2` (effective=64)
- `max_steps: 8000` (~3 epochs over 163K samples)
- `learning_rate: 2e-5` (更低，稳定收敛)
- **Resume from Stage 2b**

**预计时长**: 3-4 小时

**输出**: `outputs/h200_stage2c/checkpoint-final/` (**最终模型**)

---

## 关键参数对比

| 参数 | Stage 1 | Stage 2a | Stage 2b | Stage 2c |
|------|---------|----------|----------|----------|
| **长度范围** | 0-512 | 0-512 | 512-2048 | 0-2048 |
| **样本数** | 2.72M | 127K | 36K | 163K |
| **max_doc_tokens** | 512 | 512 | 2048 | 2048 |
| **batch_size** | 16 | 8 | 4 | 4 |
| **gradient_acc** | 1 | 2 | 2 | 2 |
| **effective_batch** | 128 | 128 | 64 | 64 |
| **learning_rate** | 2.5e-4 | 5e-5 | 3e-5 | 2e-5 |
| **max_steps** | 50K | 5K | 3K | 8K |
| **预计时长** | 8-12h | 2-3h | 1.5-2h | 3-4h |

---

## 监控指标

### Stage 2a（0-512 tokens）

**预期**：
- EM: 10% → 45-50%
- F1: 0.3 → 0.65-0.70
- Dev set 完全匹配（都是短文档）

### Stage 2b（512-2048 tokens）

**预期**：
- Loss 初期可能略微上升（文档变长）
- EM: 20% → 35-40%（中长文档更难）
- F1: 0.4 → 0.60-0.65

### Stage 2c（0-2048 混合）

**预期**：
- 短文档保持 2a 水平
- 长文档达到 2b 水平
- 整体 EM: 40-50%, F1: 0.65-0.75

---

## 总训练时长

| 阶段 | 步数 | 时长 | 累计 |
|------|------|------|------|
| Stage 1 | 50K | 8-12h | 8-12h |
| Stage 2a | 5K | 2-3h | 10-15h |
| Stage 2b | 3K | 1.5-2h | 11.5-17h |
| Stage 2c | 8K | 3-4h | **14.5-21h** |

---

## 优势对比

### 方案 A（两阶段 + 截断）
- ❌ 38% 数据被截断
- ❌ 答案可能在截断部分
- ✅ 时长短（6-9h）

### 方案 B（三阶段 + 零截断）✅
- ✅ **零截断**，62% 数据完整训练
- ✅ 渐进适应，更稳定
- ✅ Dev set 完全匹配
- ⚠️ 时长稍长（7-8h）

---

## 与完全重训 Stage 1 的对比

### 重训 Stage 1（max_doc_tokens=2048）
- 优点：真正学会压缩 2048 tokens
- 缺点：需要重跑 50K 步（8-12h）
- 总时长：20-30h

### 三阶段 Stage 2（当前方案）
- 优点：利用现有 Stage 1，7-8h 完成
- 缺点：Perceiver 在 Stage 1 只见过 512 tokens，对 2048 是"外推"
- 总时长：14.5-21h（含已完成的 Stage 1）

---

## 建议

**如果 Stage 1 已经跑完**：
→ 用三阶段 Stage 2（当前方案），快速验证效果

**如果实验效果不理想**（2048 tokens 压缩质量差）：
→ 证明 Perceiver 需要在 Stage 1 就见过长文档，再重训 Stage 1

**如果实验效果好**：
→ 证明 Perceiver 有足够的长度泛化能力，可以发论文了！

---

## 剩余 38% 超长样本怎么办？

如果三阶段效果好，可以考虑：

### 方案 1: Stage 2d（2048-8192 tokens）
再加一个阶段，专门训练超长文档。

### 方案 2: 重训 Stage 1（max_doc_tokens=4096）
覆盖 72% 数据，留 28% 截断。

### 方案 3: 动态长度训练
修改数据加载器，支持变长 batch（技术难度较高）。

但这些都可以等三阶段跑完，看效果再决定。
