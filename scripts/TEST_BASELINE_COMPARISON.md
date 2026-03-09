# Baseline Comparison Testing

## 新功能：Direct Qwen Baseline 对比

`evaluate_all_checkpoints.py` 和 `evaluate_all_models.sh` 现在会自动评估 **direct_qwen baseline**（Qwen3-0.6B 直接读取完整文档进行预测）作为上界对比。

### 评估内容

对于 Stage 1 (NTP) 评估，现在会包括：

1. **Direct Qwen (Baseline)** - Qwen3-0.6B 直接读取完整文档
   - 这是理论上的性能上界（upper bound）
   - 不进行任何压缩，文档完整输入

2. **Deep Compressor Models** - 不同 Q 值的压缩模型
   - Q=16, 32, 64, 128, 256
   - 文档压缩为 Q 个 prefix vectors

### 新增指标

- **Retention（质量保持率）**: 衡量压缩模型相对于 baseline 的性能
  - 计算公式：`retention = (baseline_loss / model_loss) × 100%`
  - 100% 表示压缩模型达到了与完整文档相同的性能
  - 50% 表示模型的 loss 是 baseline 的 2 倍

### 使用方法

```bash
# 自动评估所有模型 + baseline
bash scripts/evaluate_all_models.sh

# 或手动运行
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train.jsonl \
    --stage 1 \
    --show_samples 5
```

### 输出示例

```
╔════════════════════════════════════════════════════════════════════════════╗
║                           Comparison Table                                  ║
╚════════════════════════════════════════════════════════════════════════════╝

Model                     | perplexity   | loss         | Retention
------------------------------------------------------------------------------
Direct Qwen (baseline)    |      18.2345 |       2.9034 |         —
Q=16                      |      45.6789 |       3.8214 |      76.0%
Q=32                      |      32.1234 |       3.4698 |      83.7%
Q=64                      |      24.5678 |       3.2012 |      90.7%
Q=128                     |      20.9876 |       3.0432 |      95.4%
Q=256                     |      19.4567 |       2.9678 |      97.8%
==============================================================================

Summary:
  - Direct Qwen reads full document (512 tokens)
  - Compressed models use Q prefix vectors
  - Retention = (baseline_loss / model_loss) × 100%
```

### 快速测试（使用 tiny 数据集）

```bash
# 测试 baseline 评估（只使用少量数据验证功能）
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_tiny.jsonl \
    --stage 1 \
    --max_eval_samples 10 \
    --checkpoints outputs/stage1_q16  # 如果有训练好的模型
```

### 技术细节

**Direct Qwen Baseline 实现**：
- 使用 `AutoModelForCausalLM.from_pretrained("models/Qwen3-0.6B")`
- 输入：完整文档 + 后续片段（doc + segment）
- 标签：文档部分标记为 -100（不计算 loss），只对后续片段计算 loss
- 与压缩模型使用相同的验证集分割（最后 10%）

**对比公平性**：
- 所有模型（baseline + compressed）使用相同的验证数据
- 相同的 tokenizer 和数据预处理
- 相同的 batch size 和评估设置

### 注意事项

1. **内存占用**：Direct Qwen baseline 会加载完整的 Qwen3-0.6B 模型（~600M 参数），评估完成后会自动释放内存
2. **评估时间**：Baseline 评估会增加总体评估时间（~1-2 分钟，取决于数据集大小）
3. **仅 Stage 1**：目前只在 Stage 1 (NTP) 评估时添加 baseline 对比，Stage 2 (QA) 可以使用 `scripts/benchmark.py`

### 与 benchmark.py 的区别

- `benchmark.py`：用于 Stage 2 (QA) 的综合对比，包含多种 baseline（random_prefix, mean_pool 等）
- `evaluate_all_checkpoints.py`：用于 Stage 1 (NTP) 的模型性能对比，现在加入 direct_qwen baseline
- 两者互补，分别服务于不同的评估场景
