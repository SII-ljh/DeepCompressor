# Stage 2 Two-Phase Training Strategy

## 背景

Stage 1 用 `max_doc_tokens=512` 训练（覆盖 82% NTP 数据），但 QA 数据中 51.6% 的样本超过 512 tokens（平均 2427 tokens）。直接用 2048 tokens 训练会导致严重的分布 mismatch。

**解决方案：两阶段长度适应**

## 训练流程

### Step 0: 准备过滤数据（一次性）

```bash
# 过滤出 doc_len <= 512 的 QA 样本（~48% 数据，~126K 样本）
python scripts/filter_qa_by_length.py --max_tokens 512
```

生成：
- `data/qa_train_filtered_512.json`
- `data/qa_dev_filtered_512.json`

---

### Step 1: Stage 2a - 短文档适应（5K 步，~2-3 小时）

**目标**：在与 Stage 1 相同的长度（512 tokens）上进行 QA fine-tuning，平滑过渡。

```bash
bash scripts/train_h200_stage2a_short.sh
```

**配置**：
- `max_doc_tokens: 512`（与 Stage 1 一致）
- `batch_size: 8, gradient_accumulation: 2`（有效 batch=128）
- `max_steps: 5000`（~5 epochs over 126K short samples）
- `learning_rate: 5e-5`

**输出**：`outputs/h200_stage2a/checkpoint-final/`

---

### Step 2: Stage 2b - 长文档扩展（10K 步，~4-6 小时）

**目标**：从 Stage 2a checkpoint 继续，扩展到 2048 tokens，覆盖 62% 的 QA 数据。

```bash
bash scripts/train_h200_stage2b_long.sh
```

**配置**：
- `max_doc_tokens: 2048`（扩展 4 倍）
- `batch_size: 4, gradient_accumulation: 2`（有效 batch=64，降低以应对长序列）
- `max_steps: 10000`（~2.4 epochs over 263K full samples）
- `learning_rate: 3e-5`（比 2a 更低，保持稳定）

**输出**：`outputs/h200_stage2b/checkpoint-final/`（最终模型）

---

## 关键参数对比

| 参数 | Stage 1 | Stage 2a | Stage 2b | 说明 |
|------|---------|----------|----------|------|
| `max_doc_tokens` | 512 | **512** | **2048** | 逐步扩展 |
| `batch_size` | 16 | 8 | 4 | 随序列长度减半 |
| `gradient_accumulation` | 1 | 2 | 2 | 保持有效 batch |
| `有效 batch size` | 128 | 128 | 64 | 2b 降低避免 OOM |
| `learning_rate` | 2.5e-4 | **5e-5** | **3e-5** | Fine-tuning 用更低 LR |
| `max_steps` | 50K | 5K | 10K | 总 15K QA 步 |

## 监控指标

### Stage 2a（短文档）

日志示例：
```
[QA] step 500/5000  loss=1.234  ppl=3.44  lr=5.00e-05
[QA EVAL] step 500  EM=45.2%  F1=0.678
```

**预期**：
- EM (Exact Match) 应该从 ~10% 升到 40-50%
- F1 应该从 ~0.3 升到 0.65-0.70

### Stage 2b（长文档）

日志示例：
```
[QA] step 1000/10000  loss=1.456  ppl=4.29  lr=3.00e-05
[QA EVAL] step 1000  EM=38.5%  F1=0.652
```

**预期**：
- 初期 loss 可能上升（因为文档变长 4 倍）
- 训练 2-3K 步后应该恢复到 2a 的水平
- 最终 EM ~40-50%, F1 ~0.65-0.75

---

## 注意事项

### 1. Checkpoint 依赖

- Stage 2a 依赖 `outputs/h200_stage1/checkpoint-final/`
- Stage 2b 依赖 `outputs/h200_stage2a/checkpoint-final/`

如果你的 Stage 1 checkpoint 在其他位置，修改 `train_h200_stage2a_short.sh` 中的 `--resume_from` 路径。

### 2. 显存不足

如果 Stage 2b OOM，降低 batch_size：

```yaml
# 编辑 configs/h200_stage2b_long.yaml
training:
  batch_size: 2                # 从 4 降到 2
  gradient_accumulation_steps: 4  # 从 2 升到 4
```

### 3. 关闭蒸馏（节省显存）

如果仍然 OOM，可以关闭 teacher 蒸馏：

```yaml
# 编辑 configs/h200_stage2b_long.yaml
loss:
  kl_weight: 0.0
  hidden_mse_weight: 0.0
```

### 4. 跳过 Stage 2a（不推荐）

如果你确定 Stage 1 已经学到足够泛化能力，可以直接跳到 Stage 2b。但这样会有分布 shift 风险。

---

## 总训练时长

| 阶段 | 步数 | 预计时长 | 累计时长 |
|------|------|---------|---------|
| Stage 1 | 50K | 8-12 小时 | 8-12 小时 |
| Stage 2a | 5K | 2-3 小时 | 10-15 小时 |
| Stage 2b | 10K | 4-6 小时 | 14-21 小时 |

## 完成后

最终模型在 `outputs/h200_stage2b/checkpoint-final/`，支持：
- 短文档（< 512 tokens）：充分训练
- 中长文档（512-2048 tokens）：Stage 2b 覆盖
- 超长文档（> 2048 tokens）：会被截断，需要 Stage 1 重训才能支持

## 数据覆盖率

| 长度范围 | 占比 | 覆盖策略 |
|---------|------|---------|
| 0-512 | 48.4% | Stage 2a + 2b 充分训练 |
| 512-2048 | 13.6% | Stage 2b 训练 |
| 2048-8192 | 37.5% | **截断到 2048** |
| > 8192 | 0.5% | 截断到 2048 |

总覆盖率：**62%** 不截断，**38%** 截断。
