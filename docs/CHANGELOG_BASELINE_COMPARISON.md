# Changelog: 添加 Direct Qwen Baseline 对比

## 日期
2026-03-09

## 变更摘要
在 `evaluate_all_checkpoints.py` 和 `evaluate_all_models.sh` 中添加了 **Direct Qwen baseline** 评估，用于对比压缩模型与直接读取完整文档的性能差异。

## 问题描述
原先的 `bash scripts/evaluate_all_models.sh` 只评估不同 Q 值训练出的 Deep Compressor 模型，缺少与 Qwen3-0.6B 直接读取完整文档的对比基线。这使得无法直观地看出：
- 压缩造成的性能损失有多大
- 不同压缩比（Q 值）的性能权衡
- 压缩模型相对于理论上界的质量保持率

## 解决方案

### 1. 新增功能
在 `scripts/evaluate_all_checkpoints.py` 中添加了 `evaluate_direct_qwen_ntp()` 函数：
- 加载 Qwen3-0.6B 模型
- 让模型直接读取完整文档（无压缩）
- 在相同验证集上计算 perplexity 和 loss
- 作为性能上界（upper bound）

### 2. 修改的文件

#### `scripts/evaluate_all_checkpoints.py`
- **新增导入**: `AutoModelForCausalLM`
- **新增函数**: `evaluate_direct_qwen_ntp()` - 评估 direct_qwen baseline
- **修改主流程**: 在评估训练模型之前先评估 baseline（仅 Stage 1）
- **改进输出表格**:
  - 添加 "Retention" 列，显示质量保持率
  - baseline 显示在表格顶部
  - 更清晰的模型名称格式
- **更新文档**: 添加 baseline 评估说明

#### `CLAUDE.md`
- 更新了 Stage 1 评估输出示例
- 添加了 Retention 指标说明
- 说明了 baseline 对比的作用

### 3. 新增文件

#### `scripts/TEST_BASELINE_COMPARISON.md`
详细的功能说明文档，包括：
- 使用方法
- 输出示例
- 技术细节
- 与 benchmark.py 的区别

#### `scripts/test_baseline_only.py`
独立的测试脚本，验证 baseline 评估功能：
- 使用 tiny 数据集快速测试
- 只评估 baseline（不需要训练模型）
- ✅ 测试通过（Perplexity: 66.60, Loss: 4.20 on 10 samples）

## 使用示例

### 完整评估（包含 baseline）
```bash
# 自动发现所有模型 + baseline 评估
bash scripts/evaluate_all_models.sh

# 输出示例：
# Model                     | perplexity   | loss       | Retention
# ----------------------------------------------------------------------
# Direct Qwen (baseline)    |      18.23   |     2.90   |     —
# Q=16                      |      45.68   |     3.82   |   76.0%
# Q=32                      |      32.12   |     3.47   |   83.7%
# Q=64                      |      24.57   |     3.20   |   90.7%
# Q=128                     |      20.99   |     3.04   |   95.4%
# Q=256                     |      19.46   |     2.97   |   97.8%
```

### 只测试 baseline
```bash
# 快速验证 baseline 功能
python scripts/test_baseline_only.py
```

## 技术实现细节

### Direct Qwen Baseline 评估
```python
def evaluate_direct_qwen_ntp(eval_loader, qwen_model, accelerator):
    """
    输入：完整文档 + 后续片段（doc + segment）
    标签：文档部分标记为 -100（忽略），只对后续片段计算 loss
    输出：perplexity 和 loss
    """
```

### 质量保持率（Retention）计算
```python
retention = (baseline_loss / model_loss) × 100%
```
- 100% = 压缩模型达到与完整文档相同的性能
- 90% = 模型 loss 比 baseline 高 11%
- 50% = 模型 loss 是 baseline 的 2 倍

### 对比公平性保证
- ✓ 相同的验证数据集（最后 10%）
- ✓ 相同的 tokenizer 和数据预处理
- ✓ 相同的 batch size
- ✓ 相同的评估指标（perplexity, loss）

## 性能影响

### 内存占用
- 需要临时加载 Qwen3-0.6B 模型（~600M 参数）
- 评估完成后自动释放内存（`del qwen_model; torch.cuda.empty_cache()`）

### 时间开销
- 增加约 1-2 分钟（取决于验证集大小）
- 只运行一次 baseline 评估，后续评估所有压缩模型

## 与 benchmark.py 的区别

| 特性 | evaluate_all_checkpoints.py | benchmark.py |
|------|----------------------------|--------------|
| 用途 | Stage 1 (NTP) 模型对比 | Stage 2 (QA) 综合对比 |
| Baseline | direct_qwen | direct_qwen, random_prefix, mean_pool 等 |
| 评估指标 | perplexity, loss | EM, F1, loss |
| 使用场景 | 比较不同压缩比的模型 | 比较不同压缩方法 |

两者互补，分别服务于不同的评估需求。

## 测试验证

✅ **语法检查**: `python -m py_compile scripts/evaluate_all_checkpoints.py` - 通过
✅ **功能测试**: `python scripts/test_baseline_only.py` - 通过
✅ **输出格式**: 表格和 CSV 格式正确

### 测试结果（10 样本）
```
Perplexity: 66.6028
Loss:       4.1987
```

## 后续改进建议

1. **Stage 2 集成**: 考虑在 Stage 2 (QA) 评估中也添加类似的 baseline 对比
2. **更多 baseline**: 可以添加其他 baseline，如 random_prefix, mean_pool（类似 benchmark.py）
3. **可视化**: 生成 loss vs Q 的曲线图
4. **缓存 baseline**: 如果经常运行评估，可以缓存 baseline 结果避免重复计算

## 向后兼容性

✓ 完全向后兼容：
- 如果不需要 baseline，可以传入 `--checkpoints` 参数指定特定模型
- Stage 2 评估不受影响
- 现有的 CSV 输出格式兼容（只是多了一行 baseline）

## 文档更新

- ✅ `CLAUDE.md` - 更新了 Stage 1 评估输出示例
- ✅ `scripts/TEST_BASELINE_COMPARISON.md` - 新增详细功能说明
- ✅ `scripts/evaluate_all_checkpoints.py` - 更新了 docstring

## 总结

这个改进解决了原先缺少对比基线的问题，让用户能够：
1. 直观看到压缩的代价（loss 增加多少）
2. 评估不同 Q 值的性能权衡
3. 计算质量保持率（Retention），量化压缩效果
4. 快速识别最优的压缩比

这对于理解和优化 Deep Compressor 的压缩策略非常重要。
