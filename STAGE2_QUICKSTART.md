# Stage 2 QA Fine-tuning Quick Start

## Prerequisites

确认 Stage 1 训练已完成，checkpoint 已保存：

```bash
ls outputs/h200_stage1/checkpoint-final/trainable_weights.pt
```

如果你的 checkpoint 在其他位置（比如 `checkpoint-50000`），需要修改启动脚本中的 `--resume_from` 路径。

## Stage 2 配置说明

**与 Stage 1 的区别：**

| 参数 | Stage 1 | Stage 2 | 原因 |
|------|---------|---------|------|
| `learning_rate` | 2.5e-4 | **5e-5** | Fine-tuning 用更小 LR |
| `batch_size` | 16 | **4** | 蒸馏需要 teacher logits，显存更紧张 |
| `gradient_accumulation` | 1 | **2** | 保持有效 batch=64 |
| `max_steps` | 50K | **20K** | QA 数据量 ~230K，20K 步 ≈ 5.7 epochs |
| `kl_weight` | 1.0 | **0.5** | 蒸馏权重 |
| `hidden_mse_weight` | 0.0 | **0.1** | Hidden state 对齐 |

## 启动训练

```bash
# 在服务器上执行
cd /path/to/DeepCompressor
git pull
bash scripts/train_h200_stage2.sh
```

## 训练监控

日志输出会包含：

```
[QA] step 340/20000  loss=1.2345  ppl=3.44  lr=4.50e-05
```

每 500 步会评估一次：

```
[QA EVAL] step 500  EM=23.45%  F1=0.6789
```

- **EM (Exact Match)**: 答案完全匹配的比例
- **F1**: token-level F1 score

## 注意事项

### 1. Checkpoint 路径

如果你的 Stage 1 最终 checkpoint 不在 `checkpoint-final`，需要修改：

```bash
# 编辑 scripts/train_h200_stage2.sh
--resume_from outputs/h200_stage1/checkpoint-50000  # 改成你的实际路径
```

### 2. 显存不足

如果 OOM，可以进一步降低 batch_size：

```yaml
# 编辑 configs/h200_stage2.yaml
training:
  batch_size: 2                # 从 4 降到 2
  gradient_accumulation_steps: 4  # 从 2 升到 4
```

### 3. 不需要蒸馏

如果不想用 teacher 蒸馏（节省显存），可以关闭：

```yaml
# 编辑 configs/h200_stage2.yaml
loss:
  kl_weight: 0.0
  hidden_mse_weight: 0.0
```

这样 Stage 2 就是纯 QA fine-tuning，不做知识蒸馏。

### 4. 评估样本数限制

如果 dev set 太大导致评估很慢，可以限制评估样本数：

```bash
# 在启动脚本中添加
--max_eval_samples 5000  # 只用 5000 个样本评估
```

## 输出

训练完成后：

- **Checkpoint**: `outputs/h200_stage2/checkpoint-final/trainable_weights.pt`
- **Wandb logs**: `wandb/offline-run-*`（用 `wandb sync` 上传）

## 下一步

训练完成后，可以用以下方式评估：

```bash
# Benchmark 评估（对比 baseline）
python scripts/benchmark.py \
    --config configs/benchmark.yaml \
    --checkpoint outputs/h200_stage2/checkpoint-final/trainable_weights.pt \
    --eval_data data/qa_dev.json
```
